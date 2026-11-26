"""Single-T4 SDK 0.1.0 lifecycle gate with a fresh-process import."""
import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from minicells_native import MiniCells
from minicells_native.errors import CellAlreadyExistsError
from common import environment, publishable, write

FORMAT = "minicells.m0.developer-preview.v1"
QUESTION = "What is the registry marker?"
LONG = "The archives describe ordinary procedures for processing incoming materials. " * 14 + QUESTION


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def hashes(mc):
    return {cell.id: digest(mc.cells._store.path(cell.id)) for cell in mc.cells.list()}


def query(mc, prompt, cell=None):
    options = {} if cell is None else {"cells": [cell]}
    with patch.object(mc._backbone, "step", wraps=mc._backbone.step) as step:
        result = mc.query(prompt, max_new_tokens=8, **options)
    sizes = [len(call.args[0]) for call in step.call_args_list]
    assert sizes and max(sizes) <= 64
    assert result.selected_cell_ids == ([] if cell is None else [cell])
    return dict(response=asdict(result), calls=sizes)


def writer(mc, work):
    a, b = mc.cells.create("A"), mc.cells.create("B")
    empty_b = hashes(mc)[b.id]
    a = mc.cells.write(a.id, "The registry marker for the northern archive is LIME-173.", resume=False)
    assert a.revision == 1 and hashes(mc)[b.id] == empty_b
    b = mc.cells.write(b.id, "The registry marker for the southern archive is PLUM-296.", resume=False)
    before = hashes(mc)
    baseline = query(mc, QUESTION)
    short = query(mc, QUESTION, a.id)
    repeat = query(mc, QUESTION, a.id)
    long = query(mc, LONG, a.id)
    assert short["response"]["text"] == repeat["response"]["text"]
    assert len(mc._backbone.tokenizer.encode(f"Question: {LONG}\nAnswer:", add_special_tokens=False)) > 64
    assert long["calls"][0] == 64 and len(long["calls"]) > 1
    assert hashes(mc) == before
    fork = mc.cells.fork(a.id, "A fork")
    assert fork.id not in before
    updated = mc.cells.append(fork.id, "The corrected registry marker is MAPLE-407.", resume=False)
    assert updated.revision == fork.revision + 1 and updated.tokens_written > fork.tokens_written
    assert all(hashes(mc)[key] == value for key, value in before.items())
    mc.cells.delete(fork.id)
    assert hashes(mc) == before
    bundle = work / "a.cell"
    mc.cells.export(a.id, bundle)
    assert digest(bundle) == before[a.id]
    return dict(pid=os.getpid(), baseline=baseline, short=short, long=long,
                repeated_read_stable=True, registry_hashes=before, imported_cell_id=a.id,
                bundle_sha256=digest(bundle), fork_id=fork.id, identity=mc.model_info)


def reader(mc, work):
    original = json.loads((work / "writer.json").read_text(encoding="utf-8"))
    bundle = work / "a.cell"
    assert original["pid"] != os.getpid() and mc.model_info == original["identity"]
    assert digest(bundle) == original["bundle_sha256"]
    imported = mc.cells.import_cell(bundle)
    assert imported.id == original["imported_cell_id"] and hashes(mc)[imported.id] == original["bundle_sha256"]
    try:
        mc.cells.import_cell(bundle)
    except CellAlreadyExistsError:
        pass
    else:
        raise AssertionError("Import overwrote an existing Cell")
    before = hashes(mc)
    response = query(mc, QUESTION, imported.id)
    assert hashes(mc) == before
    appended = mc.cells.append(imported.id, "The registry was checked after import.", resume=False)
    assert appended.revision == imported.revision + 1 and hashes(mc)[imported.id] != before[imported.id]
    mc.cells.delete(imported.id)
    assert not mc.cells.list()
    return dict(pid=os.getpid(), import_hash=before[imported.id], response=response,
                before_revision=imported.revision, after_revision=appended.revision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", choices=("writer", "reader"))
    args = parser.parse_args()
    out = args.output
    if args.worker:
        with MiniCells.open_preview(out / args.worker) as mc:
            result = (writer if args.worker == "writer" else reader)(mc, out)
        write(out / f"{args.worker}.json", result)
        return

    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir()
    started = time.monotonic()
    result = dict(experiment="M0-006 SDK 0.1.0 Developer Preview", status="INVALID",
                  engineering_status="FAIL")
    try:
        result["environment"] = environment()
        gates = {}
        result["gates"] = gates
        suite_path = out / "test-receipt.json"
        gates["release_tests"] = (json.loads(suite_path.read_text())["status"]
                                  if suite_path.exists() else "NOT_RUN")
        for name in ("writer", "reader"):
            print(f"M0_START {name}", flush=True)
            run = subprocess.run([sys.executable, __file__, str(work), "--worker", name], timeout=1200)
            gates[name] = "PASS" if run.returncode == 0 else "FAIL"
            if run.returncode:
                break
            payload = json.loads((work / f"{name}.json").read_text(encoding="utf-8"))
            write(out / f"{name}.json", payload)
            if name == "writer":
                result["identity"] = payload["identity"]
        passed = all(gates.get(key) == "PASS" for key in ("release_tests", "writer", "reader"))
        result.update(status="PASS" if passed else "FAIL", engineering_status="PASS",
                      diagnosis="The SDK lifecycle was assessed independently from M1 knowledge quality.",
                      next_action="Review the exact-commit evidence before publishing a release.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
