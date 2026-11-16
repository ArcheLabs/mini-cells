"""Compare actual writes, frozen-state reads and next-token logits."""
import torch


def rms_delta(left, right):
    return (left.float() - right.float()).square().mean().sqrt().item()


@torch.no_grad()
def compare(model, wrapper, correct, counterfactual, prompt, layer):
    block = model.model.layers[layer]
    inputs = torch.tensor([prompt], device=model.get_input_embeddings().weight.device)
    captured = []

    def capture(_module, _args, kwargs):
        captured.append(kwargs["hidden_states"])

    block.self_attn = wrapper.attention
    handle = wrapper.attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(input_ids=inputs, use_cache=False)
    finally:
        handle.remove()
        block.self_attn = wrapper
    x = captured[0].float()
    from torch.nn import functional as F
    q = F.normalize(wrapper.query(x), dim=-1) * 512 ** -0.25

    def read(saved):
        cell, _ = saved
        value, _ = wrapper.memory.read_only((q, q, q), cell)
        return value

    def logits(saved):
        wrapper.reset_episode()
        wrapper.cell, wrapper.position = saved
        wrapper.write_enabled = False
        return model(input_ids=inputs, use_cache=False).logits[0, -1]

    try:
        a, b = read(correct), read(counterfactual)
        return {
            "fast_weight_rms_delta": [
                rms_delta(left, right)
                for left, right in zip(correct[0].states[0], counterfactual[0].states[0])
            ],
            "read_rms_delta": rms_delta(a, b),
            "projected_read_rms_delta": rms_delta(wrapper.output(a), wrapper.output(b)),
            "logit_rms_delta": rms_delta(logits(correct), logits(counterfactual)),
        }
    finally:
        wrapper.reset_episode()
