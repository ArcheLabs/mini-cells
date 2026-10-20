from pathlib import Path
from copy import deepcopy
from minicells_runtime.store import read, ContinuationError
from minicells_runtime.store import canonical as canonical, atomic as atomic, digest as digest
from clm001_r3b.analysis_runner import fingerprint as fingerprint

BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ID = "CLM-001-R3c-MOE-Bridge-001"
ARMS = ("A", "B", "C", "D", "E")


def configuration():
    c = read(BASE / "config/proposed.json")
    validate(c)
    return c


def validate(c):
    r, m = c["routing"], c["model"]
    if c["experiment_id"] != ID or tuple(c["arms"]) != ARMS:
        raise ContinuationError("R3C_IDENTITY_CHANGED")
    if (
        r["stages"] != 4
        or r["top_k"] != 2
        or r["fixed_context_reads"] != 1
        or not r["shared_cells"]
        or r["cells"] < 2
    ):
        raise ContinuationError("R3C_ROUTING_CONTRACT_CHANGED")
    if (
        m["width"] % m["heads"]
        or m["memory_mode"] != "static_input_readonly"
        or r["temperature"] <= 0
    ):
        raise ContinuationError("R3C_MODEL_CONTRACT_CHANGED")
    if not c.get("synthetic", False) and r["cells"] != 6:
        raise ContinuationError("R3C_SIX_CELLS_REQUIRED")
    if c["release"]["confirmation"] != "SEALED" or c["scientific_decision"]:
        raise ContinuationError("R3C_CONFIRMATION_SEALED")
    if not 0 < r["fixed_pair_offset"] < r["cells"]:
        raise ContinuationError("R3C_DISTINCT_PAIR_REQUIRED")
    if c["training"]["effective_batch"] % c["training"]["microbatch"]:
        raise ContinuationError("R3C_BATCH_MISMATCH")
    return c


def synthetic_configuration():
    c = deepcopy(configuration())
    c["synthetic"] = True
    c["model"].update(
        width=8,
        heads=2,
        cell_hidden=12,
        vocab_size=16,
        max_sequence_length=8,
        router_hidden=8,
        step_width=4,
    )
    c["routing"].update(cells=4, fixed_pair_offset=2)
    c["training"].update(
        effective_batch=2,
        microbatch=1,
        sequence_length=4,
        warmup_steps=1,
        learning_rate=0.01,
        connection_learning_rate=0.01,
    )
    c["balance"]["coefficient"] = 0.01
    return c


def source_sha():
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
