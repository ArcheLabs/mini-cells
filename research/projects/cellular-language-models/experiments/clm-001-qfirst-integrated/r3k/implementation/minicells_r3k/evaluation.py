"""Use the R3j four-grid evaluator, collecting existing counters without profiling hooks."""

import copy
from minicells_r3j.evaluation import evaluate as previous
from minicells_r3h.evaluation import quality
from .reuse import bind


def evaluate(*args, **kwargs):
    work = {}

    def observed(model, *inputs, **options):
        before = copy.deepcopy(model.work)
        row = quality(model, *inputs, **options)
        for key, value in model.work.items():
            work[key] = work.get(key, 0) + value - before.get(key, 0)
        return row

    result = bind(previous, quality=observed)(*args, **kwargs)
    result.update(
        executed_operator_rows=work,
        partial_forward_gemm_flops=partial_forward_flops(work, args[2]),
        hardware_flops=None,
        accounting_scope="Existing forward operator counters; no TorchDispatch, profiler or wall-time-to-FLOPs conversion",
    )
    return result


def partial_forward_flops(work, runtime):
    """Count only FFN, tied output head and input projection GEMMs; not total FLOPs."""
    m = runtime["engine"]["model"]
    d = m["width"]
    return (
        4 * d * m["cell_hidden"] * work.get("cell_rows", 0)
        + 2 * d * m["vocab_size"] * work.get("head_rows", 0)
        + 2 * d * d * work.get("input_rows", 0)
    )
