"""The audited dense, batch-one recurrent execution policy for MoM 1.3B."""
from collections import deque
from contextlib import ExitStack, contextmanager
import inspect
from threading import RLock
from unittest.mock import patch

import torch
from torch.nn import functional as F

EXECUTION_LOCK = RLock()
MAX_RECURRENT_TOKENS = 64
PREVIEW_PROFILE = "t4-fp16-recurrent-v1"


def stable_transform(original):
    source = inspect.getsource(original)
    needle = "sorted_indices = combined.argsort()"
    if source.count(needle) != 1:
        raise RuntimeError("Expected the pinned MoM sort statement")
    namespace = dict(original.__globals__)
    code = source.replace(needle, "sorted_indices = combined.argsort(stable=True)")
    exec(compile(code, inspect.getfile(original), "exec"), namespace)
    return namespace[original.__name__]


def convolution_history(original):
    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs):
        if self.use_fast_conv1d or cu_seqlens is not None or kwargs.get("seq_idx") is not None:
            raise RuntimeError("Developer Preview requires dense FLA Triton convolution")
        if cache is None or x.shape[1] == 1:
            return original(self, x, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
        # The last W-1 inputs supply the causal context at this chunk boundary.
        history = cache.transpose(1, 2)[:, 1:]
        prefix = history.shape[1]
        joined = torch.cat((history, x), dim=1)
        if residual is not None:
            residual = torch.nn.functional.pad(residual, (0, 0, prefix, 0))
        if mask is not None:
            mask = torch.nn.functional.pad(mask, (prefix, 0), value=1)
        output, updated = original(self, joined, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
        return output[:, prefix:].contiguous(), updated
    return forward


def skip_padding(forward, conv, x, valid, *, cache=None, output_final_state=False):
    """Compute padded outputs on scratch state and persist only valid inputs."""
    width = x.shape[1]
    lengths = valid.sum(dim=1).tolist()
    if all(length == width for length in lengths):
        return forward(conv, x, cache=cache, output_final_state=output_final_state)
    outputs, states = [], []
    for lane, count in enumerate(lengths):
        split = width - count
        inputs = x[lane:lane + 1]
        state = None if cache is None else cache[lane:lane + 1]
        parts = []
        if split:
            scratch = None if state is None else state.clone()
            padded, _ = forward(conv, inputs[:, :split].contiguous(), cache=scratch,
                                output_final_state=output_final_state)
            parts.append(padded)
        if count:
            result, state = forward(conv, inputs[:, split:].contiguous(), cache=state,
                                    output_final_state=output_final_state)
            parts.append(result)
        elif output_final_state and state is None:
            state = x.new_zeros(1, x.shape[-1], conv.kernel_size[0])
        outputs.append(torch.cat(parts, dim=1))
        if state is not None:
            states.append(state)
    if cache is None and states:
        cache = torch.cat(states)
    return torch.cat(outputs), cache


def no_padding_transitions(arguments, valid):
    gate = valid[None, :, None]
    return dict(arguments, g=arguments["g"].masked_fill(~gate, 0),
                beta=arguments["beta"].masked_fill(~gate, 0))


class RecurrentExecution:
    """Scope the A.3.2 correctness interventions to an SDK forward."""
    def __init__(self, shared_memory):
        from fla.modules import convolution
        from mom.layers import mom_gated_deltanet as ops

        self.ops = ops
        self.convolution = convolution
        self.short_conv = convolution.ShortConvolution
        self.transform = stable_transform(ops.transform)
        self.shared_memory = shared_memory

    @contextmanager
    def scope(self):
        conv = self.convolution
        short_conv = self.short_conv
        original_forward = convolution_history(short_conv.forward)
        original_delta = self.ops.fused_recurrent_gated_delta_rule
        pending = deque()
        valid = None

        def transform(x, routing_mask, num_memories, selected_memories, capacity):
            nonlocal valid
            packed, _, _, _, mask = result = self.transform(
                x, routing_mask, num_memories, selected_memories, capacity)
            valid = F.pad(mask, (packed.shape[2] - mask.shape[1], 0), value=False)
            # The native batch-one recurrent branch omits entirely empty experts.
            active = packed.reshape(num_memories, -1).any(dim=1).tolist()
            pending.extend(valid[i] for i, enabled in enumerate(active) if enabled)
            if self.shared_memory:
                pending.append(None)
            return result

        def forward(module, x, residual=None, mask=None, cache=None, output_final_state=False,
                    cu_seqlens=None, **kwargs):
            if x.shape[0] == 1:
                return original_forward(module, x, residual, mask, cache, output_final_state, cu_seqlens, **kwargs)
            if (x.shape[:2] != valid.shape or residual is not None or mask is not None
                    or cu_seqlens is not None or kwargs.get("seq_idx") is not None):
                raise ValueError("Developer Preview expects dense batch-one expert convolution")
            return skip_padding(original_forward, module, x, valid, cache=cache,
                                output_final_state=output_final_state)

        def step(module, x, residual, cache, cu_seqlens=None):
            return conv.causal_conv1d_update(
                x=x, cache=cache, residual=residual, weight=module.weight.squeeze(1),
                bias=module.bias, activation=module.activation)

        def recurrent(**arguments):
            mask = pending.popleft()
            if mask is not None:
                arguments = no_padding_transitions(arguments, mask)
            return original_delta(**arguments)

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.ops, "transform", transform))
            stack.enter_context(patch.object(short_conv, "forward", forward))
            stack.enter_context(patch.object(short_conv, "step", step))
            stack.enter_context(patch.object(self.ops, "fused_recurrent_gated_delta_rule", recurrent))
            yield
