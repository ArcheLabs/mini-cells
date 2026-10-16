"""CPU exact control; only real Kaggle mounted-input evidence can validate persistence."""

from pathlib import Path
from .store import atomic, read, digest, ContinuationError
from . import r3


def compare_control(checkpoint_input, output, parent_version=None, notebook_version=None):
    import tempfile

    source = Path(checkpoint_input)
    meta = r3.verify(source)
    c = read(source / "runtime-config.json")
    with tempfile.TemporaryDirectory() as tmp:
        control = Path(tmp) / "control"
        # Identical identity is deliberate: scientific comparison is not run-ID noise.
        r3.new_run(c, source / "assets", control, source=meta["source_sha"], run_id=meta["run_id"])
        for arm in c["arms"]:
            r3.segment_arm(control, arm, c["training"]["steps"])
        reference = r3.package(control)
        rows = {
            arm: {
                "continued": meta["arms"][arm]["training_state_sha256"],
                "uninterrupted": reference["arms"][arm]["training_state_sha256"],
                "exact_equal": meta["arms"][arm]["training_state_sha256"]
                == reference["arms"][arm]["training_state_sha256"],
            }
            for arm in c["arms"]
        }
    if not all(row["exact_equal"] for row in rows.values()):
        raise ContinuationError("CONTINUOUS_RESUME_STATE_MISMATCH")
    mounted = (
        source.is_absolute()
        and "/kaggle/input" in str(source)
        and str(source).startswith("/kaggle/input/")
    )
    parent = read(source / "parent-input.json") if (source / "parent-input.json").exists() else None
    result = {
        "status": "EXACT_CPU_CONTROL_PASS",
        "state_comparison": rows,
        "scope": "SYNTHETIC_ENGINEERING_ONLY",
        "scientific_decision": False,
        "kaggle_input_path": str(source),
        "actual_mounted_input": mounted,
        "parent_input": parent,
        "parent_version": parent_version,
        "notebook_version": notebook_version,
        "input_continuation_sha256": digest(source / "continuation.json"),
        "KAGGLE_CROSS_SESSION_RESUME": "NOT_VERIFIED",
        "reason": "A/B/C exact version identities, parent mount and read-only historical evaluation must all be collated",
    }
    Path(output).mkdir(parents=True, exist_ok=False)
    atomic(Path(output) / "control-comparison.json", result)
    return result
