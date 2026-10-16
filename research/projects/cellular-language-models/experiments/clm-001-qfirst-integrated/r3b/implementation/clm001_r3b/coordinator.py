"""Host-only serialized publisher, immutable verified handoffs and graceful failure.

No worker touches the publish tree. Queued workers block until exact HF commit ACK.
On a failure all workers finish their current local transaction then stop; never
kill a GPU process or continue indefinitely past a failed remote safe boundary.
"""

import multiprocessing as mp
from pathlib import Path
import os
import queue
import shutil
import time
from minicells_runtime.store import atomic, digest, read, safe, ContinuationError


def merge_handoff(root, arm, source):
    from .common import ARMS

    if arm not in ARMS:
        raise ContinuationError("ARM_NOT_FOUND")
    root, source = Path(root), Path(source)
    witness = read(safe(source, "handoff.json"))
    files = witness["files"]
    if "latest.json" not in files or "identity.json" not in files:
        raise ContinuationError("HANDOFF_INCOMPLETE")
    for name, row in files.items():
        path = safe(source, name)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ContinuationError("HANDOFF_HASH_MISMATCH", name)
    folder = root / "arms" / arm
    latest = read(source / "latest.json")
    if (
        latest["sha256"] != witness["checkpoint_sha256"]
        or latest["identity"]["arm"] != arm
        or latest["identity"]["source_commit"] != read(root / "experiment.lock.json")["source_sha"]
    ):
        raise ContinuationError("HANDOFF_IDENTITY_MISMATCH")
    if (folder / "latest.json").exists():
        old = read(folder / "latest.json")
        import re

        def rank(s):
            return tuple(
                int(x or 0)
                for x in re.fullmatch(r"checkpoint-step-(\d{6})(?:-event-(\d{6}))?\.pt", s).groups()
            )

        if rank(latest["file"]) < rank(old["file"]):
            raise ContinuationError("HANDOFF_ROLLBACK_REJECTED")
        if rank(latest["file"]) == rank(old["file"]) and latest["sha256"] != old["sha256"]:
            raise ContinuationError("CHECKPOINT_ID_COLLISION")
    folder.mkdir(parents=True, exist_ok=True)
    # Install latest last. A failed merge cannot turn a half-copied arm into a
    # valid published selector. The previous HF index remains authoritative.
    for name in [n for n in files if n != "latest.json"] + ["latest.json"]:
        target = folder / name
        incoming = safe(source, name)
        if target.exists() and digest(target) != files[name]["sha256"] and name != "latest.json":
            raise ContinuationError("IMMUTABLE_HANDOFF_COLLISION", name)
        if target.exists() and digest(target) == files[name]["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".receiving")
        if target.suffix == ".pt":
            os.link(incoming, temporary)
        else:
            shutil.copyfile(incoming, temporary)
        os.replace(temporary, target)
    return witness


class Publisher:
    def __init__(self, root, store):
        self.root = Path(root)
        self.store = store
        self.evidence = self.root.parent / (self.root.name + "-evidence")
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.ledger = (
            read(self.evidence / "resource-ledger.json")
            if (self.evidence / "resource-ledger.json").exists()
            else {"uploads": []}
        )

    def publish(self, arm=None, source=None, *, readback=False):
        from .runner import package
        from .common import safe_error

        begin = time.perf_counter()
        try:
            if arm is not None:
                merge_handoff(self.root, arm, source)
            meta = package(self.root)
            atomic(
                self.evidence / "storage-status.json",
                {"status": "UPLOAD_IN_PROGRESS", "scientific_decision": False},
            )
            index = self.store.publish_committed_checkpoint(self.root, readback=readback)
            atomic(self.evidence / "checkpoint-index.json", index)
            self.ledger["uploads"].append(
                {
                    "seconds": time.perf_counter() - begin,
                    "hf_revision": index["checkpoint_ref"]["hf_revision"],
                    "logical_bundle_bytes": meta["output_capacity"]["bytes"],
                    "file_count": len(read(self.root / meta["files_manifest"])["files"]),
                    "transport": index.get("transport", {}),
                    "full_readback": readback,
                    "arm_progress": {a: p["step"] for a, p in meta["arms"].items()},
                }
            )
            atomic(self.evidence / "resource-ledger.json", self.ledger)
            atomic(
                self.evidence / "storage-status.json",
                {
                    "status": index["status"],
                    "hf_repo": self.store.repo,
                    "hf_revision": index["checkpoint_ref"]["hf_revision"],
                    "scientific_decision": False,
                },
            )
            print(
                {
                    "storage_status": index["status"],
                    "hf_revision": index["checkpoint_ref"]["hf_revision"],
                },
                flush=True,
            )
            return index
        except Exception as error:
            atomic(
                self.evidence / "storage-status.json",
                {"status": "STORAGE_BLOCKED", **safe_error(error), "scientific_decision": False},
            )
            raise ContinuationError(
                "STORAGE_BLOCKED", getattr(error, "code", type(error).__name__)
            ) from None


def worker_entry(root, arms, end, physical_device, messages, acknowledgements, stop, *, cpu=False):
    # Set visibility before the first CUDA call in this fresh spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(physical_device)
    from .runner import segment, handoff
    from .common import safe_error

    root = Path(root)
    try:
        for arm in arms:
            if stop.is_set():
                break
            counter = [0]

            def boundary(folder, phase):
                frozen = handoff(folder, root.parent / "handoffs" / arm, phase=phase)
                token = str(physical_device) + ":" + arm + ":" + str(counter[0])
                counter[0] += 1
                messages.put(
                    {
                        "kind": "checkpoint",
                        "arm": arm,
                        "path": str(frozen),
                        "token": token,
                        "queue": physical_device,
                    }
                )
                ack = acknowledgements.get()
                if ack.get("token") != token or not ack.get("ok"):
                    raise ContinuationError("STORAGE_BLOCKED")

            result = segment(
                root,
                arm,
                end,
                device="cpu" if cpu else "cuda:0",
                committed=boundary,
                stop_requested=stop.is_set,
            )
            messages.put({"kind": "arm-done", "result": result, "queue": physical_device})
        messages.put({"kind": "queue-done", "queue": physical_device})
    except Exception as error:
        messages.put({"kind": "worker-failed", "queue": physical_device, **safe_error(error)})


def run(
    root, live, end, store, *, cpu=False, devices=(0, 1), worker_target=None, final_readback=True
):
    """Queues Q→QS and QC. Only the host Publisher can ACK durable progress."""
    from .runner import verify

    root, live = Path(root), Path(live)
    context = mp.get_context("spawn")
    messages, stop = context.Queue(), context.Event()
    queues = [["Q-R3b", "QS-R3b"], ["QC-R3b"]]
    acks = {i: context.Queue() for i in range(2)}
    target = worker_target or worker_entry
    workers = [
        context.Process(
            target=target,
            args=(live, arms, end, i, messages, acks[i], stop),
            kwargs={"cpu": cpu},
        )
        for i, arms in enumerate(queues)
    ]
    # Physical mapping is kept outside frozen scientific config; children see one
    # GPU. Devices passed here must match the preflight T4 inventory.
    if not cpu:
        workers = [
            context.Process(
                target=target, args=(live, arms, end, devices[i], messages, acks[i], stop)
            )
            for i, arms in enumerate(queues)
        ]
        acks = dict(zip(devices, acks.values()))
    publisher = Publisher(root, store)
    # The assigned CUDA visibility must already exist when the child imports
    # torch/bootstrap, not only when its target function eventually starts.
    parent_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for i, worker in enumerate(workers):
            os.environ["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(devices[i])
            worker.start()
    finally:
        if parent_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = parent_visibility
    finished, errors, results = set(), [], []
    try:
        while len(finished) < len(workers):
            try:
                message = messages.get(timeout=1)
            except queue.Empty:
                for key, process in zip(range(2) if cpu else devices, workers):
                    if process.exitcode is not None and key not in finished:
                        errors.append(
                            {"failure_code": "WORKER_EXIT_WITHOUT_COMPLETION", "queue": key}
                        )
                        stop.set()
                        finished.add(key)
                if all(not p.is_alive() for p in workers):
                    if len(finished) != len(workers):
                        errors.append({"failure_code": "WORKER_EXIT_WITHOUT_COMPLETION"})
                    break
                continue
            key = message["queue"]
            if message["kind"] == "checkpoint":
                ok = False
                if not errors:
                    try:
                        publisher.publish(message["arm"], message["path"])
                        ok = True
                    except Exception as error:
                        from .common import safe_error

                        errors.append(safe_error(error))
                        stop.set()
                acks[key].put({"token": message["token"], "ok": ok})
            elif message["kind"] == "arm-done":
                results.append(message["result"])
            elif message["kind"] == "worker-failed":
                errors.append(message)
                stop.set()
                finished.add(key)
            elif message["kind"] == "queue-done":
                finished.add(key)
        if errors:
            atomic(
                publisher.evidence / "failure-index.json",
                {
                    "status": "PARTIAL",
                    "failures": errors,
                    "arms": results,
                    "scientific_decision": False,
                },
            )
            raise ContinuationError("STORAGE_BLOCKED_OR_WORKER_FAILED")
        verify(root)
        return publisher.publish(readback=final_readback)
    finally:
        stop.set()
        # Never terminate or kill a worker. Each accepted queue ACK releases its
        # blocked safe boundary; active updates finish locally before exit.
        for ack in acks.values():
            ack.put({"token": "shutdown", "ok": False})
        for worker in workers:
            worker.join()
