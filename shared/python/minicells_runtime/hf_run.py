"""Synthetic engineering launcher with mandatory durable HF boundaries.

No changes to frozen R3. Formal R3b configuration remains outside this launcher.
"""

import argparse
import json
from pathlib import Path
import tempfile
import time
from . import r3
from .smoke import configuration, prepare
from .store import ContinuationError, MountedInputStore, atomic, read
from minicells_artifacts.hf_storage import HFCheckpointStore, HubBackend, error_code


def persist(store, root, *, readback=False):
    # Storage evidence lives outside bundle: don't recursively upload upload receipts.
    evidence = Path(root).parent / (Path(root).name + "-storage")
    evidence.mkdir(exist_ok=True)
    started = time.monotonic()
    atomic(
        evidence / "storage-status.json",
        {"status": "UPLOAD_IN_PROGRESS", "scientific_decision": False},
    )
    try:
        r3.package(root)
        index = store.publish_committed_checkpoint(root, readback=readback)
        atomic(evidence / "checkpoint-index.json", index)
        status = {
            "status": index["status"],
            "hf_repo": store.repo,
            "hf_revision": index["checkpoint_ref"]["hf_revision"],
            "elapsed_seconds": time.monotonic() - started,
            "scientific_decision": False,
            "HF_CROSS_SESSION_RESUME": "NOT_VERIFIED",
        }
        atomic(evidence / "storage-status.json", status)
        print(
            json.dumps(
                {
                    "storage_status": status["status"],
                    "hf_repo": store.repo,
                    "hf_revision": status["hf_revision"],
                }
            ),
            flush=True,
        )
        return index
    except Exception as error:
        code = getattr(error, "code", "STORAGE_BLOCKED")
        atomic(
            evidence / "storage-status.json",
            {
                "status": "VERIFICATION_FAILED" if "MISMATCH" in code else "STORAGE_BLOCKED",
                "failure_code": code,
                **error_code(error),
                "scientific_decision": False,
            },
        )
        raise ContinuationError("STORAGE_BLOCKED", code) from None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=(
            "auto",
            "train",
            "resume",
            "evaluate",
            "fork",
            "storage-preflight",
            "publish-only",
        ),
        default="auto",
    )
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint-ref", type=Path)
    p.add_argument("--github-index-url")
    p.add_argument("--run-key")
    p.add_argument("--run-id", default="hf-engineering-" + r3.source_sha()[:12])
    p.add_argument("--segment-end-step", type=int, default=6)
    p.add_argument("--source-sha", default=r3.source_sha())
    p.add_argument("--arm", default="QC-R3")
    p.add_argument("--step", type=int, default=6)
    p.add_argument("--revision", type=int)
    p.add_argument("--new-run-id")
    p.add_argument("--new-config", type=Path)
    p.add_argument("--optimizer-policy", choices=("inherit", "reset"))
    p.add_argument("--publish-github", action="store_true")
    a = p.parse_args()
    if a.segment_end_step < 0:
        p.error("segment-end-step must be nonnegative")
    store = HFCheckpointStore(HubBackend(a.hf_repo))
    if a.mode == "storage-preflight":
        print(json.dumps(store.preflight()))
        return
    evidence = a.output.parent / (a.output.name + "-storage")
    if a.mode == "publish-only":
        from .publication import publish

        print(json.dumps(publish(a.output, storage=evidence)))
        return
    # Read-only modes need read permission, never a mandatory remote write probe.
    if a.mode in ("auto", "train", "resume", "fork"):
        store.preflight()
    ref = read(a.checkpoint_ref) if a.checkpoint_ref else None
    if a.github_index_url:
        from urllib.request import urlopen
        from urllib.parse import urlsplit

        url = urlsplit(a.github_index_url)
        if (
            url.scheme != "https"
            or url.netloc != "raw.githubusercontent.com"
            or not url.path.startswith("/ArcheLabs/mini-cells/")
            or not url.path.endswith("/checkpoint-index.json")
            or url.query
        ):
            raise ContinuationError("GITHUB_INDEX_URL_REJECTED")
        try:
            with urlopen(a.github_index_url, timeout=30) as stream:
                raw = stream.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ContinuationError("GITHUB_INDEX_TOO_LARGE")
            ref = json.loads(raw)
            if ref.get("schema") != "minicells.checkpoint-index.v1":
                raise ContinuationError("CHECKPOINT_INDEX_SCHEMA_INVALID")
        except ContinuationError:
            raise
        except Exception as error:
            raise ContinuationError(
                "GITHUB_INDEX_FETCH_FAILED", json.dumps(error_code(error))
            ) from None
    if ref and "checkpoint_ref" in ref:
        ref = ref["checkpoint_ref"]
    if not ref and a.run_key:
        ref = store.lookup(a.run_key)
    mode = a.mode
    if ref is None and mode in ("resume", "evaluate", "fork"):
        ref = store.locate(a.source_sha, a.run_id)
    if a.publish_github and mode in ("auto", "train", "resume", "fork"):
        from clm001_r3 import publication as existing

        namespace = {"engineering": "kaggle-continuation", "run_id": a.run_id}
        existing.prepare(a.output.parent / "engineering-result-checkout", a.source_sha, namespace)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if mode in ("auto", "train"):
            r3.require_source(a.source_sha)
            try:
                existing = store.locate(a.source_sha, a.run_id)
            except ContinuationError as error:
                if error.code != "HF_RUN_NOT_FOUND":
                    raise
                existing = None
            if mode == "train" and existing:
                raise ContinuationError("HF_RUN_ALREADY_EXISTS", "use resume or a new run-id")
            if ref or existing:
                ref = ref or existing
                mode = "resume"
            else:
                c = configuration()
                assets = prepare(tmp / "assets", c)
                r3.new_run(c, assets, a.output, source=a.source_sha, run_id=a.run_id)
                mode = "train"
        if mode in ("resume", "evaluate", "fork"):
            if not ref:
                raise ContinuationError("HF_CHECKPOINT_REF_REQUIRED")
            fetched = tmp / "fetched"
            meta = store.fetch_verified_checkpoint(
                ref, fetched, a.source_sha if mode == "resume" else None
            )
            print(
                json.dumps({"HF_RESTORED_REVISION": ref["hf_revision"], "run_id": meta["run_id"]}),
                flush=True,
            )
            if mode == "evaluate":
                result = r3.evaluate_checkpoint(
                    fetched,
                    a.output,
                    a.arm,
                    a.step,
                    {"policies": ["greedy", "sampled", "uniform"]},
                    revision=a.revision,
                )
                # Control is separate from the evaluated historical weights.
                from .validation import compare_control

                control = compare_control(fetched, a.output.with_name(a.output.name + "-control"))
                atomic(
                    a.output / "hf-provenance.json",
                    {"checkpoint_ref": ref, "control": control, "scientific_decision": False},
                )
                if a.publish_github:
                    from .publication import publish

                    print(
                        json.dumps(
                            publish(
                                fetched,
                                evaluation={
                                    "result": result,
                                    "control": control,
                                    "checkpoint_ref": ref,
                                },
                            )
                        ),
                        flush=True,
                    )
                print(json.dumps(result))
                return
            if mode == "fork":
                if not all((a.new_run_id, a.new_config, a.optimizer_policy)):
                    p.error("fork requires new-run-id, new-config and optimizer-policy")
                result = r3.fork_experiment(
                    fetched,
                    a.output,
                    a.arm,
                    a.step,
                    read(a.new_config),
                    run_id=a.new_run_id,
                    optimizer_policy=a.optimizer_policy,
                    revision=a.revision,
                )
                persist(store, a.output, readback=True)
                print(json.dumps(result))
                return
            r3.require_source(meta["source_sha"])
            MountedInputStore(fetched).materialize(a.output, a.source_sha)
            atomic(
                a.output / "parent-input.json",
                {"checkpoint_ref": ref, "parent_run_id": meta["run_id"]},
            )
        # Single remote publisher, serial arm boundary commits. This CPU launcher
        # does not introduce a second competing publisher or GPU training.
        try:
            c = read(a.output / "runtime-config.json")
            if not c.get("synthetic"):
                raise ContinuationError("FORMAL_TRAINING_NOT_AUTHORIZED")
            for arm in c["arms"]:
                r3.segment_arm(
                    a.output, arm, a.segment_end_step, committed=lambda root: persist(store, root)
                )
            index = persist(store, a.output, readback=True)
            print(
                json.dumps(
                    {
                        "status": index["status"],
                        "run_key": index["checkpoint_ref"]["run_key"],
                        "HF_CROSS_SESSION_RESUME": "NOT_VERIFIED",
                        "scientific_decision": False,
                    }
                ),
                flush=True,
            )
        finally:
            if a.publish_github and (a.output / "continuation.json").exists():
                from .publication import publish

                try:
                    print(json.dumps(publish(a.output, storage=evidence)), flush=True)
                except Exception as error:
                    print(
                        json.dumps(
                            {
                                "github_status": "FAILED",
                                **error_code(error),
                                "local_checkpoint_retained": True,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
