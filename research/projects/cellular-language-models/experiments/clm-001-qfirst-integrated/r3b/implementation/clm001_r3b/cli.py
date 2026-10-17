"""Automatic preflight and private Pilot; formal execution requires Gate A–E."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from minicells_runtime.store import atomic, canonical, read, ContinuationError, MountedInputStore
from minicells_artifacts.hf_storage import HubBackend, HFCheckpointStore
from .common import BASE, ARMS, load, runtime, require_training_release, safe_error
from clm001_r3.common import source_sha


def checkpoint_ref(value):
    if not value:
        return None
    if str(value).startswith("https://"):
        from urllib.parse import urlsplit
        from urllib.request import urlopen

        url = urlsplit(value)
        if (
            url.netloc != "raw.githubusercontent.com"
            or not url.path.startswith("/ArcheLabs/mini-cells/")
            or url.query
        ):
            raise ContinuationError("GITHUB_INDEX_URL_REJECTED")
        with urlopen(value, timeout=30) as stream:
            content = stream.read(2 * 1024 * 1024 + 1)
        if len(content) > 2 * 1024 * 1024:
            raise ContinuationError("GITHUB_INDEX_TOO_LARGE")
        document = json.loads(content)
    else:
        document = read(value)
    return document.get("checkpoint_ref", document)


def resolved(store, source, run_id, ref=None):
    return checkpoint_ref(ref) if ref else store.locate(source, run_id)


def private_pilot(
    candidate, source, workspace, store, preflight_report, *, devices, publish_github
):
    from . import runner, coordinator, pilot
    from .publication import publish_documents

    workspace = Path(workspace)
    run_id = "r3b-pilot-" + source[:12] + "-" + canonical(candidate)[:8]
    root = workspace / "pilot"
    cached = None
    try:
        cached = store.locate(source, run_id)
    except ContinuationError as error:
        if error.code != "HF_RUN_NOT_FOUND":
            raise
    if cached:
        if root.exists():
            raise ContinuationError("OUTPUT_ALREADY_EXISTS")
        store.fetch_verified_checkpoint(cached, root, source)
        if (root / "pilot-summary.json").exists():
            report = read(root / "pilot-summary.json")
            if report["candidate_sha256"] != canonical(candidate):
                raise ContinuationError("CACHED_PILOT_IDENTITY_MISMATCH")
            print("R3b Pilot reused at pinned HF revision; no rehearsal repeated", flush=True)
            return report, root
        raise ContinuationError(
            "PILOT_PARTIAL_REQUIRES_REVIEW", "do not repeat partial GPU work silently"
        )
    c = runtime(
        candidate,
        source,
        beta=candidate["engine"]["competition"]["pilot_beta"],
        pilot_hash="PRIVATE_REHEARSAL_ONLY",
        run_id=run_id,
        assets_hash=preflight_report["assets_manifest_sha256"],
    )
    c["r3b"]["experiment_id"] = "CLM-001-R3b-GPU-PILOT"
    c["pilot_rehearsal"] = True
    c["execution"]["configuration_sha256"] = canonical(
        {k: v for k, v in c.items() if k != "execution"}
    )
    runner.new_run(
        candidate,
        c,
        workspace / "assets",
        root,
        release={
            "passed": True,
            "scope": "PRIVATE_GPU_PILOT_NOT_SCIENTIFIC_TRAINING",
            "acceptance": preflight_report["acceptance"],
        },
    )
    live = workspace / "pilot-live"
    shutil.copytree(root, live)
    index = coordinator.run(
        root,
        live,
        candidate["engine"]["resources"]["pilot_steps"],
        store,
        devices=devices,
        worker_target=pilot.worker_entry,
        final_readback=False,
    )
    report = pilot.summarize(candidate, root, source)
    atomic(root / "pilot-summary.json", report)
    index = coordinator.Publisher(root, store).publish(readback=True)
    report["final_storage_verification"] = read(
        root.parent / (root.name + "-evidence") / "resource-ledger.json"
    )["uploads"][-1]
    if publish_github:
        from minicells_runtime.publication import storage_summary

        publish_documents(
            root,
            {
                "experiment-manifest.json": {
                    "experiment_id": c["r3b"]["experiment_id"],
                    "run_id": run_id,
                    "scope": "PRIVATE_GPU_PILOT_ONLY",
                    "formal_training_started": False,
                    "scientific_decision": False,
                },
                "resource-ledger.json": report,
                "checkpoint-index.json": storage_summary("checkpoint-index.json", index),
                "storage-status.json": read(
                    root.parent / (root.name + "-evidence") / "storage-status.json"
                ),
            },
        )
    return report, root


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "auto",
            "preflight",
            "gpu-pilot",
            "train",
            "resume",
            "evaluate",
            "fork",
            "publish-only",
        ),
        default="auto",
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hf-repo", default="libingjiang/minicells-research-checkpoints")
    p.add_argument("--checkpoint-ref")
    p.add_argument("--assets-ref")
    p.add_argument("--run-id")
    p.add_argument("--segment-end-step", type=int, default=200)
    p.add_argument("--acceptance", type=Path)
    p.add_argument("--pilot-report", type=Path)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--publish-github", action="store_true")
    p.add_argument("--arm", default="QC-R3b", choices=ARMS)
    p.add_argument("--step", type=int, default=400)
    p.add_argument("--event-revision", type=int)
    p.add_argument("--new-run-id")
    p.add_argument("--new-experiment-id")
    p.add_argument("--optimizer-policy", choices=("inherit", "reset"))
    a = p.parse_args()
    if not 0 <= a.segment_end_step <= 500:
        p.error("segment-end-step must be within the fixed 0..500 budget")
    candidate, source = load(a.config), source_sha()
    if a.hf_repo != candidate["execution"]["hf_repo"]:
        raise ContinuationError("TRUSTED_HF_REPOSITORY_MISMATCH")
    devices = tuple(
        os.environ.get(
            "MINICELLS_WORKER_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
        ).split(",")
    )
    if len(devices) < 2:
        devices = ("0", "1")
    # The host is CPU only. Fresh children restore visibility before any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    workspace = a.output
    workspace.mkdir(parents=True, exist_ok=True)
    run_id = a.run_id or "r3b-" + source[:12] + "-" + canonical(candidate)[:8]
    try:
        from . import runner, preflight
        from .publication import publish

        store = HFCheckpointStore(HubBackend(a.hf_repo))
        if a.mode in ("evaluate", "fork", "publish-only", "resume"):
            reference = resolved(store, source, run_id, a.checkpoint_ref)
            with tempfile.TemporaryDirectory() as tmp:
                fetched = Path(tmp) / "fetched"
                store.fetch_verified_checkpoint(
                    reference, fetched, source if a.mode == "resume" else None
                )
                meta = runner.verify(fetched)
                if a.mode == "evaluate":
                    from .evaluation import evaluate_checkpoint

                    result = evaluate_checkpoint(
                        fetched, workspace / "evaluation", a.arm, a.step, a.event_revision
                    )
                    if a.publish_github:
                        # Public projection excludes private per-story arrays.
                        safe = {
                            k: result[k]
                            for k in (
                                "experiment_id",
                                "run_id",
                                "arm",
                                "source_sha",
                                "evaluation_source_sha",
                                "evaluation_protocol",
                                "retrospective",
                                "exploratory",
                                "optimizer_updates",
                                "scientific_decision",
                            )
                        }
                        safe["quality"] = {
                            p: {k: v for k, v in row.items() if k not in ("nll", "cost", "groups")}
                            for p, row in result["quality"].items()
                        }
                        safe["checkpoint_sha256"] = result["checkpoint"]["checkpoint_sha256"]
                        safe["hf_reference"] = {
                            k: reference[k]
                            for k in ("hf_repo", "hf_revision", "bundle_sha256", "bundle_path")
                        }
                        from .publication import publish_documents

                        safe["step"] = result["checkpoint"]["step"]
                        safe["event_revision"] = result["checkpoint"]["revision"]
                        publish_documents(
                            fetched,
                            {"metrics-summary.json": safe},
                            scope="historical-evaluation-"
                            + canonical(
                                {
                                    "checkpoint": result["checkpoint"],
                                    "evaluation_source_sha": source,
                                    "protocol": result["evaluation_protocol"],
                                }
                            )[:16],
                        )
                    print(
                        json.dumps(
                            {
                                "status": "EVALUATED",
                                "optimizer_updates": 0,
                                "step": result["checkpoint"]["step"],
                                "event_revision": result["checkpoint"]["revision"],
                            }
                        ),
                        flush=True,
                    )
                    return
                if a.mode == "fork":
                    from .evaluation import fork_checkpoint

                    child = workspace / "fork"
                    fork_checkpoint(
                        fetched,
                        child,
                        a.arm,
                        a.step,
                        revision=a.event_revision,
                        new_run_id=a.new_run_id,
                        new_experiment_id=a.new_experiment_id,
                        optimizer_policy=a.optimizer_policy,
                    )
                    from .coordinator import Publisher

                    print(json.dumps(Publisher(child, store).publish(readback=True)), flush=True)
                    return
                root = workspace / "run"
                MountedInputStore(fetched).materialize(root, meta["source_sha"])
            if a.mode == "publish-only":
                # Recover the actual HF reference as authoritative index even when
                # the prior Kaggle local receipt/output is absent.
                evidence = root.parent / (root.name + "-evidence")
                evidence.mkdir(exist_ok=True)
                snapshot = read(
                    store.backend.get(reference["bundle_path"], reference["hf_revision"])
                )
                index = store._index(meta, snapshot, reference, snapshot["files"])
                atomic(evidence / "checkpoint-index.json", index)
                atomic(
                    evidence / "storage-status.json",
                    {
                        "status": index["status"],
                        "hf_revision": reference["hf_revision"],
                        "hf_repo": store.repo,
                        "scientific_decision": False,
                    },
                )
                print(json.dumps(publish(root)), flush=True)
                return
            c = read(root / "runtime-config.json")
            preflight.cpu_evidence(read(root / "candidate-config.json"))
            release = read(root / "preflight.json")
            require_training_release(
                read(root / "candidate-config.json"),
                source,
                release["acceptance"],
                release["authorization"],
                release["pilot"],
            )
            store.preflight()
            if a.publish_github:
                __import__("clm001_r3b.publication", fromlist=["preflight"]).preflight(
                    workspace, source, meta["run_id"]
                )
        else:
            # Formal training never starts because a GPU exists or Pilot completes.
            if a.mode == "train":
                acceptance = preflight.engineering_evidence(a.acceptance)
                approval = read(a.authorization or BASE / "config/authorization.json")
                pilot_report = read(a.pilot_report or BASE / "reports/gpu-pilot.json")
                require_training_release(candidate, source, acceptance, approval, pilot_report)
            cached = None
            try:
                cached = store.locate(
                    source, "r3b-pilot-" + source[:12] + "-" + canonical(candidate)[:8]
                )
            except ContinuationError as error:
                if error.code != "HF_RUN_NOT_FOUND":
                    raise
            if cached is None and not a.assets_ref:
                # Reuse trusted accepted assets; never repeat a successful Pilot
                # merely to obtain data/tokenizer bytes on a new source revision.
                baseline = read(BASE / "reports/diagnostic-optimization/baseline-evidence.json")
                cached = baseline["evidence"]["checkpoint-index.json"]["checkpoint_ref"]
            report = preflight.run(
                candidate,
                source,
                workspace,
                store,
                acceptance_path=a.acceptance,
                assets_ref=checkpoint_ref(a.assets_ref),
                cached_ref=cached,
                publish_github=a.publish_github,
            )
            if a.mode == "preflight":
                print(json.dumps(report), flush=True)
                return
            if a.mode == "auto":
                print(
                    json.dumps(
                        {
                            "status": "ENGINEERING_READY_FORMAL_TRAINING_HOLD",
                            "diagnostic_execution": "DEFER_EXPENSIVE_ANALYSIS",
                            "prior_pilot_repeated": False,
                            "formal_training_authorized": False,
                            "reason": "Targeted delta verification and explicit release remain required",
                        }
                    ),
                    flush=True,
                )
                return
            if a.mode in ("gpu-pilot",):
                raise ContinuationError("FULL_PILOT_REPLAY_NOT_AUTHORIZED")
            c = runtime(
                candidate,
                source,
                beta=pilot_report["beta"],
                pilot_hash=canonical(pilot_report),
                run_id=run_id,
                assets_hash=report["assets_manifest_sha256"],
            )
            if report["assets_manifest_sha256"] != pilot_report["assets_manifest_sha256"]:
                raise ContinuationError("REGISTERED_ASSET_IDENTITY_MISMATCH")
            root = workspace / "run"
            runner.new_run(
                candidate,
                c,
                workspace / "assets",
                root,
                release={
                    "passed": True,
                    "acceptance": acceptance,
                    "authorization": approval,
                    "pilot": pilot_report,
                },
            )
        live = workspace / "live"
        shutil.copytree(root, live)
        from .coordinator import run

        published = None
        try:
            index = run(
                root, live, a.segment_end_step, store, devices=devices[:2], final_readback=False
            )
            from .plots import render

            render(root)
            from .coordinator import Publisher

            index = Publisher(root, store).publish(readback=True)
        finally:
            if a.publish_github and (root / "continuation.json").exists():
                published = publish(root)
        print(
            json.dumps(
                {
                    "EXPERIMENT": "CLM-001-R3b",
                    "RUN_ID": read(root / "experiment.lock.json")["run_id"],
                    "SOURCE_SHA": source,
                    "ARM_PROGRESS": {
                        arm: row["step"]
                        for arm, row in read(root / "continuation.json")["arms"].items()
                    },
                    "HF_REPO": store.repo,
                    "HF_REVISION": index["checkpoint_ref"]["hf_revision"],
                    "CHECKPOINT_INDEX_SHA256": canonical(index),
                    "GITHUB_RESULTS_BRANCH": published["branch"] if published else None,
                    "GITHUB_RESULTS_COMMIT": published.get("commit_sha") if published else None,
                    "STORAGE_STATUS": index["status"],
                    "SCIENTIFIC_STATUS": "SINGLE_SEED_EXPLORATORY",
                }
            ),
            flush=True,
        )
    except Exception as error:
        # Return trustworthy failure evidence even if the first worker failed
        # before a complete continuation bundle. Never fabricate an HF index.
        if a.publish_github and safe_error(error)["failure_code"] != "GITHUB_PUBLICATION_FAILED":
            for target in (workspace / "run", workspace / "pilot"):
                if (target / "experiment.lock.json").exists():
                    try:
                        from .publication import publish_documents

                        lock = read(target / "experiment.lock.json")
                        publish_documents(
                            target,
                            {
                                "failure-index.json": {
                                    "status": "FAILED_OR_PARTIAL",
                                    "failures": read(
                                        target.parent
                                        / (target.name + "-evidence")
                                        / "failure-index.json"
                                    )["failures"]
                                    if (
                                        target.parent
                                        / (target.name + "-evidence")
                                        / "failure-index.json"
                                    ).exists()
                                    else [safe_error(error)],
                                    "scientific_decision": False,
                                },
                                "experiment-manifest.json": {
                                    "experiment_id": lock["experiment_id"],
                                    "run_id": lock["run_id"],
                                    "source_sha": source,
                                    "status": "FAILED_OR_PARTIAL",
                                    "scientific_decision": False,
                                },
                            },
                        )
                    except Exception as publication_error:
                        print(
                            json.dumps(
                                {"publication_status": "FAILED", **safe_error(publication_error)}
                            ),
                            flush=True,
                        )
                    break
        atomic(
            workspace / "failure.json",
            {
                **safe_error(error),
                "scientific_decision": False,
                "formal_training_authorized": False,
            },
        )
        print(json.dumps({"status": "STOPPED", **safe_error(error)}), flush=True)
        raise ContinuationError(safe_error(error)["failure_code"]) from None
