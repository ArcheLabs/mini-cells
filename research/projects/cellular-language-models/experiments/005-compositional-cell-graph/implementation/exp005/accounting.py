"""Observed GEMM FLOPs and per-token unique touched-weight accounting."""
from __future__ import annotations

import contextvars
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor, nn


_ACTIVE_COUNTER: contextvars.ContextVar["ObservedGemmFlops | None"] = contextvars.ContextVar(
    "exp005_gemm_counter", default=None
)


def record_attention_flops(query_shape: torch.Size, key_length: int) -> None:
    counter = _ACTIVE_COUNTER.get()
    if counter is None:
        return
    batch, heads, query_length, head_width = map(int, query_shape)
    # QK^T and AV are both counted as executed dense matmuls, including causal
    # prefill. Decode has query_length=1 and cached key_length=prefix+1.
    flops = 4 * batch * heads * query_length * int(key_length) * head_width
    counter.add("attention.qk_av", flops)


def record_linear_flops(name: str, rows: int, input_width: int, output_width: int) -> None:
    counter = _ACTIVE_COUNTER.get()
    if counter is not None:
        counter.add(name, 2 * int(rows) * int(input_width) * int(output_width))


class ObservedGemmFlops:
    """Count the actual shapes passed through Linear modules and SDPA matmuls."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.by_module: dict[str, int] = defaultdict(int)
        self.handles = []
        self._token = None
        self.enabled = True

    def __enter__(self) -> "ObservedGemmFlops":
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            def hook(layer: nn.Linear, _inputs: tuple[Tensor, ...], output: Tensor,
                     module_name: str = name) -> None:
                rows = output.numel() // max(1, layer.out_features)
                self.add(module_name or "linear", 2 * rows * layer.in_features * layer.out_features)

            self.handles.append(module.register_forward_hook(hook))
        self._token = _ACTIVE_COUNTER.set(self)
        return self

    def add(self, name: str, flops: int) -> None:
        if self.enabled:
            self.by_module[name] += int(flops)

    @property
    def total(self) -> int:
        return sum(self.by_module.values())

    def report(self, token_count: int) -> dict[str, Any]:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        return {"observed_forward_gemm_flops": self.total,
                "observed_gemm_flops_per_token": self.total / token_count,
                "by_module": dict(sorted(self.by_module.items())),
                "scope": "Linear matmuls plus explicit SDPA QK^T/AV; excludes non-matmul operations"}

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._token is not None:
            _ACTIVE_COUNTER.reset(self._token)
            self._token = None


def _as_path_tensor(trace: dict[str, Tensor] | None) -> Tensor:
    if not trace or "paths" not in trace:
        raise ValueError("cell execution trace is required for active-weight accounting")
    paths = trace["paths"]
    if paths.ndim != 3:
        raise ValueError("paths must have shape (batch, sequence, executions)")
    return paths.detach().cpu()


def unique_weight_counts(model: nn.Module) -> tuple[Tensor, dict[str, int]]:
    """Return one exact touched-weight count per input token for the last forward.

    The tied embedding/output matrix is counted in full because the output
    projection reads every vocabulary row. Router address tables count only rows
    actually gathered; dense linear controller weights count once when touched.
    Repeated Cell execution affects GEMM work, but a Cell's weights count once.
    """
    traces = getattr(model, "last_execution", None)
    if not traces or any(trace is None for trace in traces):
        raise ValueError("run a traced forward pass before active-weight accounting")
    breakdown = model.parameter_breakdown()
    shared = int(breakdown["shared"])
    cell_params = int(breakdown["ffn_cells"])
    controller_total = int(breakdown["controller"])
    arm = model.arm
    batch, sequence = traces[0]["paths"].shape[:2]
    paths_all = [_as_path_tensor(trace) for trace in traces]
    counts = torch.empty((batch, sequence), dtype=torch.float64)

    for batch_index in range(batch):
        for token_index in range(sequence):
            count = shared
            for layer_index, path_tensor in enumerate(paths_all):
                path = path_tensor[batch_index, token_index].tolist()
                if arm == "M0":
                    unique_cells = set(path)
                    experts = len(model.blocks[0].ffn.cells)
                    count += len(unique_cells) * (cell_params // (len(model.blocks) * experts))
                    count += controller_total // len(model.blocks)
                    continue
                if arm == "P0":
                    cells = len(model.blocks[0].ffn.cells)
                    count += len(set(path)) * (cell_params // (len(model.blocks) * cells))
                    count += controller_total // len(model.blocks)
                    continue

                block = model.blocks[layer_index].ffn
                router = block.router
                assert router is not None
                unique_cells = set(path)
                count += len(unique_cells) * sum(p.numel() for p in block.cells[0].parameters())
                # Query weights are dense and reused; address tables are gathered
                # only for the sources and hard-constrained candidate IDs.
                sources = [router.cell_count, *path[:-1]]
                candidates = traces[layer_index]["candidates"][batch_index, token_index]
                target_ids = set(int(x) for x in candidates.reshape(-1).tolist())
                source_ids = set(int(x) for x in sources)
                count += router.query.weight.numel()
                count += len(source_ids) * (router.source_address.embedding_dim +
                                             router.edge_source.embedding_dim)
                count += len(target_ids) * (router.target_address.embedding_dim +
                                            router.edge_target.embedding_dim)
            counts[batch_index, token_index] = count

    # Each per-layer controller is charged directly above; controller_total is
    # used only by the M0/P0 dense-controller branches.
    return counts.reshape(-1), {key: int(value) for key, value in breakdown.items()}


def execution_summary(model: nn.Module) -> dict[str, Any]:
    traces = getattr(model, "last_execution", None)
    if not traces or any(trace is None for trace in traces):
        raise ValueError("run a traced forward pass before execution accounting")
    paths = [_as_path_tensor(trace) for trace in traces]
    all_visits = torch.cat([path.reshape(-1, path.shape[-1]) for path in paths], dim=0)
    executed = int((all_visits >= 0).sum())
    unique = 0
    for path in all_visits.tolist():
        unique += len(set(int(cell) for cell in path if int(cell) >= 0))
    token_count = paths[0].shape[0] * paths[0].shape[1]
    layers = len(paths)
    return {"executed_cell_visits_per_token": executed / (token_count * layers),
            "unique_cells_per_token_layer": unique / (token_count * layers),
            "train_backup_cell_visits_per_token":
                int(getattr(model, "last_backup_cell_visits", 0)) / token_count,
            "token_count": token_count,
            "cell_execution_scope": "forward Cell module calls; repeated Cell visits are counted repeatedly"}


def active_weight_summary(model: nn.Module) -> dict[str, Any]:
    values, breakdown = unique_weight_counts(model)
    values = values.float()
    total = sum(parameter.numel() for parameter in model.parameters())
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.95, 0.99])).tolist()
    return {"unique_weight_params_per_token": {
                "mean": float(values.mean()), "p50": float(quantiles[0]),
                "p95": float(quantiles[1]), "p99": float(quantiles[2])},
            "unique_weight_fraction_of_model": float(values.mean()) / total,
            "model_total_params": total, "parameter_breakdown": breakdown,
            "interpretation": "touched-weight working-set proxy; not RAM residency or DRAM bytes"}
