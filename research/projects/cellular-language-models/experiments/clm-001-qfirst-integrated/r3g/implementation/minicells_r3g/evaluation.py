"""Same historical evaluator and windows; hooks only hash executed paths."""

import platform
import math
import torch
from minicells_r3c_moe_bridge.evaluation import evaluate as historical_evaluate
from minicells_r3e.evaluation import windows
from minicells_r3d.evaluation import mean_nll
from .common import fingerprint


@torch.no_grad()
def measure(model, runtime, data, repeats):
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("UNEXPECTED_EVALUATION_PRECISION")
    before = fingerprint(model.state_dict())
    runs = []
    original_mode = model.training
    try:
        model.eval()
        for _ in range(repeats):
            paths = []

            def capture(module, args, output):
                if output.logits.requires_grad:
                    raise RuntimeError("READONLY_EVAL_CREATED_GRADIENT_GRAPH")
                paths.append(output.actions.detach().cpu())

            hook = model.register_forward_hook(capture)
            try:
                rows = historical_evaluate(
                    model,
                    windows(data, runtime, "calibration", next(model.parameters()).device),
                    model.seed,
                )
            finally:
                hook.remove()
            runs.append(
                {"nll": mean_nll(rows), "story_rows": rows, "path_hash": fingerprint(paths)}
            )
    finally:
        model.train(original_mode)
    if before != fingerprint(model.state_dict()):
        raise RuntimeError("AUDIT_MUTATED_MODEL_PARAMETERS")
    if len({r["path_hash"] for r in runs}) != 1:
        raise RuntimeError("NONDETERMINISTIC_READONLY_PATHS")
    nlls = [r["nll"] for r in runs]
    if not all(math.isfinite(v) for v in nlls):
        raise FloatingPointError("NONFINITE_CROSSOVER_NLL")
    return {
        "nll": nlls[0],
        "raw_nll_repeats": nlls,
        "max_repeat_error": max(nlls) - min(nlls),
        "path_hash": runs[0]["path_hash"],
        "model_hash": before,
        "story_rows": runs[0]["story_rows"],
        "precision": str(next(model.parameters()).dtype),
        "epsilon": 0,
        "deterministic_route": True,
    }


def environment(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else platform.processor() or "CPU",
        "dtype": "torch.float32",
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
