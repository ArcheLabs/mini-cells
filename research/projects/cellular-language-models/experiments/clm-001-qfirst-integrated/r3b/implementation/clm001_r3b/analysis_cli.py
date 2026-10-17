"""One read-only entry for listing, bounded tasks, recovery and aggregate-only publication."""

import argparse
from pathlib import Path
from minicells_runtime.store import atomic, canonical, read, ContinuationError
from minicells_artifacts.hf_storage import HFCheckpointStore, HubBackend
from clm001_r3.common import source_sha
from .common import ARMS, safe_error
from .cli import checkpoint_ref
from .diagnostic_schedule import load_plan
from .analysis_store import AnalysisStore
from . import analysis_runner as ar
from .analysis_aggregate import aggregate_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "list-checkpoints",
            "analyze-checkpoint",
            "analyze-batch",
            "aggregate-analysis",
            "publish-only",
        ),
        default="list-checkpoints",
    )
    p.add_argument("--checkpoint-ref", required=True)
    p.add_argument("--hf-repo", default="libingjiang/minicells-research-checkpoints")
    p.add_argument("--analysis-plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("quality", "functional", "action"), default="quality")
    p.add_argument("--role", choices=("calibration", "development"), default="development")
    p.add_argument("--arm", choices=ARMS, default="QC-R3b")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--event-revision", type=int)
    p.add_argument("--phase", choices=("pre_event", "post_event"), default="post_event")
    p.add_argument(
        "--policies",
        nargs="+",
        choices=("greedy", "uniform", "sampled", "constant"),
        default=["greedy", "uniform"],
    )
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    p.add_argument("--publish-github", action="store_true")
    a = p.parse_args()
    if a.hf_repo != "libingjiang/minicells-research-checkpoints":
        p.error("only the trusted existing private repository is allowed")
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        ref = checkpoint_ref(a.checkpoint_ref)
        if ref.get("hf_repo") != a.hf_repo:
            raise ContinuationError("TRUSTED_HF_REPOSITORY_MISMATCH")
        backend = HubBackend(a.hf_repo)
        store = HFCheckpointStore(backend)
        root = a.output / "immutable-training-input"
        if not root.exists():
            store.fetch_verified_checkpoint(ref, root)
            atomic(a.output / "input-reference.json", ref)
        elif read(a.output / "input-reference.json") != ref:
            raise ContinuationError("ANALYSIS_INPUT_REFERENCE_CHANGED")
        from .runner import verify

        meta = verify(root)
        if a.mode == "list-checkpoints":
            result = ar.list_checkpoints(root)
            atomic(a.output / "checkpoints.json", result)
            print(result, flush=True)
            return
        plan = load_plan(a.analysis_plan)
        analysis_source = source_sha()
        storage = AnalysisStore(backend)
        if a.mode == "analyze-checkpoint":
            if a.event_revision is None:
                raise ContinuationError("EXPLICIT_EVENT_REVISION_REQUIRED")
            task = ar.identity(
                root,
                ref,
                plan,
                analysis_source,
                a.arm,
                a.step,
                a.event_revision,
                a.phase,
                a.stage,
                a.role,
                a.policies if a.stage == "quality" else [],
            )
            selected_tasks = [task]
            tasks, missing = ar.expand(root, ref, plan, analysis_source)
            if canonical(task) not in {canonical(t) for t in tasks}:
                # Explicit additional tasks are a separately identified scope,
                # never silently relabelled as scheduled primary observations.
                tasks.append(task)
                missing.append({"status": "EXPLORATORY_EXTENSION_AFTER_INITIAL_RUN"})
        else:
            tasks, missing = ar.expand(root, ref, plan, analysis_source)
            selected_tasks = [t for t in tasks if t["stage"] == a.stage]
        if a.mode in ("aggregate-analysis", "publish-only"):
            # Recovery reads the authoritative private index; it never infers
            # completion from a local file or executes any diagnostic/model call.
            results = [r for r in storage.collect(tasks) if r is not None]
            ledger = []
            for t in tasks:
                _, index = storage.index(t)
                entry = index["tasks"].get(canonical(t))
                ledger.append(
                    {
                        "task_key": canonical(t),
                        "status": "COMPLETE" if entry else "NOT_EVALUATED",
                        **({"entry": entry} if entry else {}),
                    }
                )
        else:
            results, ledger = ar.batch(
                root,
                a.output / "analysis",
                selected_tasks,
                plan,
                storage,
                max_tasks=a.max_tasks,
                device=a.device,
            )
        # Include all previously committed stages without re-running them.
        results = [r for r in storage.collect(tasks) if r is not None]
        ledger = []
        for t in tasks:
            _, ix = storage.index(t)
            entry = ix["tasks"].get(canonical(t))
            ledger.append(
                {
                    "task_key": canonical(t),
                    "status": "COMPLETE" if entry else "NOT_EVALUATED",
                    **({"entry": entry} if entry else {}),
                }
            )
        c = read(root / "runtime-config.json")
        values = aggregate_tasks(c, meta, tasks, results, ledger)
        values["experiment-manifest.json"]["unavailable_checkpoints"] = missing
        if missing:
            values["experiment-manifest.json"]["analysis_status"] = "PARTIAL"
            values["experiment-manifest.json"]["status"] = (
                "TRAINING_COMPLETE_" if meta["complete"] else "TRAINING_PARTIAL_"
            ) + "ANALYSIS_PARTIAL"
        for name, value in values.items():
            atomic(a.output / "public" / name, value)
        from .plots import render_analysis

        render_analysis(values, a.output / "figures", plan)
        if a.publish_github:
            from .publication import publish_documents

            publish_documents(
                root,
                values,
                scope={
                    "analysis_source_sha": analysis_source,
                    "analysis_protocol_sha256": canonical(plan),
                },
            )
        print(
            {
                "status": values["experiment-manifest.json"]["status"],
                "TRAINING_SOURCE_SHA": meta["source_sha"],
                "ANALYSIS_SOURCE_SHA": analysis_source,
                "analysis_protocol_sha256": canonical(plan),
                "hf_revision": ref["hf_revision"],
                "completed_tasks": len(results),
                "training_updates": 0,
            },
            flush=True,
        )
    except Exception as error:
        atomic(
            a.output / "analysis-failure.json",
            {"status": "ANALYSIS_BLOCKED", **safe_error(error), "training_status_modified": False},
        )
        print({"status": "ANALYSIS_BLOCKED", **safe_error(error)}, flush=True)
        raise
