"""One-engineering-segment entry. No formal R3/R3b training is exposed here."""

import argparse
import json
from pathlib import Path
import tempfile
from .store import atomic, read, ContinuationError
from . import r3
from .smoke import configuration, prepare


def boolean(value):
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("use true or false")
    return value == "true"


def _main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "auto",
            "verify-checkpoint",
            "evaluate-checkpoint",
            "fork-experiment",
            "compare-control",
            "publish-only",
            "import-r3-output",
            "segment-worker",
        ),
        default="auto",
    )
    p.add_argument("--checkpoint-input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--source-sha", default=r3.source_sha())
    p.add_argument("--stop-after-step", type=int, default=6)
    p.add_argument("--require-resume", type=boolean, default=False)
    p.add_argument("--arm", default="QC-R3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--step", type=int, default=6)
    p.add_argument("--revision", type=int)
    p.add_argument("--evaluation-config", type=Path)
    p.add_argument("--new-config", type=Path)
    p.add_argument("--new-run-id")
    p.add_argument("--optimizer-policy", choices=("inherit", "reset"))
    p.add_argument("--parent-version", help="exact Kaggle parent Notebook URL with scriptVersionId")
    p.add_argument(
        "--notebook-version", help="exact current Kaggle Notebook version identity if known"
    )
    a = p.parse_args()
    if a.mode == "segment-worker":
        result = r3.segment_arm(a.output, a.arm, a.stop_after_step, device=a.device)
    elif a.mode == "import-r3-output":
        result = r3.import_legacy(
            a.checkpoint_input,
            a.assets,
            a.output,
            expected_source=a.source_sha,
            candidate_config=a.config,
        )
    elif a.mode == "verify-checkpoint":
        result = r3.verify(a.checkpoint_input, a.source_sha)
    elif a.mode == "evaluate-checkpoint":
        result = r3.evaluate_checkpoint(
            a.checkpoint_input,
            a.output,
            a.arm,
            a.step,
            read(a.evaluation_config)
            if a.evaluation_config
            else {"policies": ["greedy", "sampled", "uniform"]},
            revision=a.revision,
        )
    elif a.mode == "fork-experiment":
        if not all((a.new_config, a.new_run_id, a.optimizer_policy)):
            p.error("fork requires new-config, new-run-id and optimizer-policy")
        result = r3.fork_experiment(
            a.checkpoint_input,
            a.output,
            a.arm,
            a.step,
            read(a.new_config),
            run_id=a.new_run_id,
            optimizer_policy=a.optimizer_policy,
            revision=a.revision,
        )
    elif a.mode == "compare-control":
        from .validation import compare_control

        result = compare_control(a.checkpoint_input, a.output, a.parent_version, a.notebook_version)
    elif a.mode == "publish-only":
        from .publication import publish

        result = publish(a.checkpoint_input or a.output)
    else:
        if a.stop_after_step < 0:
            p.error("stop-after-step must be nonnegative")
        r3.require_source(a.source_sha)
        if a.require_resume and a.checkpoint_input is None:
            raise ContinuationError("RESUME_INPUT_MISSING", "require-resume=true")
        if a.checkpoint_input is not None:
            result = r3.resume(
                a.checkpoint_input,
                a.output,
                a.stop_after_step,
                require_resume=a.require_resume,
                source=a.source_sha,
            )
        else:
            if a.require_resume:
                raise ContinuationError("RESUME_INPUT_MISSING")
            c = read(a.config) if a.config else configuration()
            if not c.get("synthetic"):
                raise ContinuationError(
                    "FORMAL_TRAINING_NOT_AUTHORIZED",
                    "this launcher is the engineering acceptance fixture",
                )
            with tempfile.TemporaryDirectory() as tmp:
                assets = prepare(Path(tmp) / "assets", c)
                r3.new_run(
                    c,
                    assets,
                    a.output,
                    source=a.source_sha,
                    run_id="engineering-" + a.source_sha[:12],
                )
            r3.segments(a.output, a.stop_after_step)
            result = r3.package(a.output)
        result = {
            "status": result["status"],
            "run_id": result["run_id"],
            "source_sha": result["source_sha"],
            "complete": result["complete"],
            "arms": {k: v["last_committed_step"] for k, v in result["arms"].items()},
            "scientific_decision": False,
            "formal_training_authorized": False,
        }
    print(json.dumps(result, indent=2), flush=True)


def main():
    try:
        return _main()
    except Exception as error:
        import sys

        args = argparse.ArgumentParser(add_help=False)
        args.add_argument("--output", type=Path)
        args.add_argument("--checkpoint-input", type=Path)
        known, _ = args.parse_known_args()
        out = known.output
        if (
            out is not None
            and out.is_dir()
            and not str(out).startswith("/kaggle/input/")
            and (
                known.checkpoint_input is None or out.resolve() != known.checkpoint_input.resolve()
            )
        ):
            code = getattr(error, "code", "ENGINEERING_EXECUTION_FAILED")
            atomic(
                out / "failure.json",
                {"failure_code": code, "type": type(error).__name__, "scientific_decision": False},
            )
            r3.state(out, "FAILED", failure_code=code)
        print(
            "ENGINEERING_FAILURE: " + getattr(error, "code", type(error).__name__),
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
