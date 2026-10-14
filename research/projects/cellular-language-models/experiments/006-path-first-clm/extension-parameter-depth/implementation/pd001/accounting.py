"""Actual module-row accounting; incomplete total FLOPs are explicitly labeled."""

from torch.utils._python_dispatch import TorchDispatchMode
from torch import nn
from weakref import WeakKeyDictionary
from exp006.reuse import ObservedGemmFlops
from exp006.accounting import parameter_catalog


class Counter(ObservedGemmFlops):
    def __init__(self, model):
        super().__init__(model)
        self.calls, self.work = {}, 0
        self.last = {}
        self.module_sizes = {
            id(m): sum(p.numel() for p in m.parameters(recurse=False)) for m in model.modules()
        }

    def __enter__(self):
        super().__enter__()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):

                def count(m, inputs, output, key=name):
                    if not self.enabled:
                        return
                    rows = (
                        inputs[0].numel()
                        if isinstance(m, nn.Embedding)
                        else inputs[0].numel() // inputs[0].shape[-1]
                    )
                    self.calls[key] = self.calls.get(key, 0) + rows
                    size = m.embedding_dim if isinstance(m, nn.Embedding) else self.module_sizes[id(m)]
                    self.work += rows * size

                self.handles.append(module.register_forward_hook(count))
        if self.model.arm == "M0":
            for block in self.model.blocks:
                self.handles.append(
                    block.register_forward_hook(
                        lambda m, i, o: self.add("attention.qk_av", 4 * i[0].numel() * o[1][0].shape[-2])
                    )
                )
        else:
            self.handles.append(
                self.model.context.register_forward_hook(
                    lambda m, i, o: self.add("context.qk_av", 4 * i[0].shape[0] * i[1].key.shape[1] * m.width)
                )
            )

            def edge(m, i, o):
                if self.enabled and hasattr(m, "source") and i[3] != "zero":
                    self.add("connection.edge_dot", 2 * i[0].shape[0] * m.actions * m.source.embedding_dim)
                    self.work += i[0].shape[0] * m.target.weight.numel()

            self.handles.append(self.model.controller.register_forward_hook(edge))

        def before(m, i):
            if not self.enabled:
                return
            self.before = dict(self.calls)

        def after(m, i, o):
            if not self.enabled:
                return
            tokens = o.logits.shape[0] * o.logits.shape[1]
            self.add("tied_head", 2 * tokens * m.config["width"] * m.config["vocab_size"])
            self.work += tokens * m.embedding.weight.numel()
            if m.arm != "M0":
                actual = {
                    name: rows - self.before.get(name, 0)
                    for name, rows in self.calls.items()
                    if name.startswith("cells.") and name.endswith(".up") or name == "context.query"
                }
                expected = {
                    f"cells.{a}.up": int((o.actions == a).sum()) for a in range(m.config["cell_count"])
                }
                # context_zero replay intentionally removes the read operator.
                expected["context.query"] = (
                    0 if o.context_zero else int((o.actions == m.config["cell_count"]).sum())
                )
                if any(actual.get(k, 0) != v for k, v in expected.items()):
                    raise AssertionError("selected action hooks disagree with hard path")
                self.last = {"expected": expected, "actual": actual, "verified": True}
                self.touched = touched_weights(m, o)

        self.handles.append(self.model.register_forward_pre_hook(before))
        self.handles.append(self.model.register_forward_hook(after))
        return self

    def report(self, tokens):
        inventory = parameter_catalog(self.model)
        return {
            **super().report(tokens),
            "parameter_catalog": inventory,
            "resident_parameters": sum(v["numel"] for v in inventory.values()),
            "resident_weight_bytes": sum(v["bytes"] for v in inventory.values()),
            "executed_parameter_work": self.work,
            "actual_module_token_rows": self.calls,
            "trace_verification": self.last,
            "unique_touched_weights_last_rollout": getattr(self, "touched", None),
            "total_flops_status": "INCOMPLETE_ACCOUNTING",
            "non_gemm_flops": None,
            "non_gemm_note": "normalization, sampling, softmax and scheduling are not inferred as zero",
            "training_backward_flops": None,
            "backward_note": "forward ledger is not total training FLOPs",
        }


_inventories = WeakKeyDictionary()


def touched_weights(model, out):
    from exp006.accounting import module_size as original_module_size

    if model not in _inventories:
        _inventories[model] = {id(module): original_module_size(module) for module in model.modules()}
    inventory = _inventories[model]

    def module_size(module):
        return inventory[id(module)]

    c = model.config
    base = (
        model.embedding.weight.numel()
        + model.position.embedding_dim
        + module_size(model.init)
        + module_size(model.memory)
        + module_size(model.final)
    )
    result = []
    for row in out.actions.detach().reshape(-1, c["budget"]).cpu().tolist():
        actions = [a for a in row if a >= 0]
        used = set(a for a in actions if a <= c["cell_count"])
        value = base + sum(module_size(model.cells[a]) for a in used if a < c["cell_count"])
        if used:
            value += module_size(model.norm)
        if c["cell_count"] in used and not out.context_zero:
            value += module_size(model.context)
        if actions:
            ctrl = model.controller
            value += module_size(ctrl.scorer) + len(actions) * ctrl.step.embedding_dim
            if model.training:
                value += module_size(ctrl.critic)
            if hasattr(ctrl, "source") and out.edge_mode != "zero":
                value += (
                    len(set([c["cell_count"] + 2] + actions[:-1])) * ctrl.source.embedding_dim
                    + ctrl.target.weight.numel()
                )
        result.append(value)
    return {
        "mean": sum(result) / len(result),
        "min": min(result),
        "max": max(result),
        "scope": "module inventories plus executed unique actions/step rows; current-token static memory preparation; full tied head counted once",
        "note": "distinct parameter visits are not executed work, resident memory or DRAM traffic",
    }


class TrainingGemmCounter(TorchDispatchMode):
    """Observe executed mm/addmm/bmm in both forward and autograd backward.

    Fused SDPA kernels and non-GEMM operators remain explicitly unaccounted.
    This is a supported-operator lower bound, never a complete training budget.
    """

    def __init__(self):
        super().__init__()
        self.flops = 0
        self.by_operator = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        flops = 0
        if name == "aten.mm.default":
            a, b = args[:2]
            flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif name == "aten.addmm.default":
            a, b = args[1:3]
            flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif name == "aten.bmm.default":
            a, b = args[:2]
            flops = 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        if flops:
            self.flops += int(flops)
            self.by_operator[name] = self.by_operator.get(name, 0) + int(flops)
        return func(*args, **(kwargs or {}))
