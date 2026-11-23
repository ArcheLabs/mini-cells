"""Real-GPU SDK smoke with process-isolated export/restore."""
import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from minicells import MiniCells
from minicells_native.backend import sha

from compositional_recall import read, write
from run_multicell_fusion import candidate_score
from multicell_fusion import FusionConfig, FusionReader

PROBE = "The Green Belt Movement was founded by"
CANDIDATES = (" Wangari Maathai.", " Jane Goodall.")


def teacher_forced(runtime, cell_id):
    reader = FusionReader(runtime._backbone)
    state = runtime.cells._state(cell_id)
    return [candidate_score(reader, [state], FusionConfig("single_cell"), None, PROBE, answer)
            for answer in CANDIDATES]


def cell_hashes(runtime):
    return {cell.id: sha(runtime.cells._store.path(cell.id)) for cell in runtime.cells.list()}


def response(runtime, prompt, **options):
    result = runtime.query(prompt, max_new_tokens=12, **options)
    if result.generation_tokens <= 0 or not result.text.strip():
        raise RuntimeError(f"No text generated for {prompt!r}")
    if not all(c.isprintable() or c.isspace() for c in result.text) or "\ufffd" in result.text:
        raise RuntimeError(f"Invalid decoded text: {result.text!r}")
    return asdict(result)


def phase(runtime, args):
    if args.stage == "write":
        probes = {}
        probes["base"] = response(runtime, "What is the capital of France?", cells=[])
        empty = response(runtime, "What is the capital of France?")
        assert empty["fallback_reason"] == "no_registered_cells"
        a, b = runtime.cells.create("founder"), runtime.cells.create("prize")
        runtime.cells.write(a.id, "The Green Belt Movement was founded by Wangari Maathai.")
        runtime.cells.write(b.id, "Wangari Maathai received the Nobel Peace Prize in 2004.")
        before = cell_hashes(runtime)
        probes["single"] = response(runtime, "Who founded the Green Belt Movement?", cells=[a.id])
        probes["two"] = response(runtime, "Who founded the Green Belt Movement?", cells=[a.id, b.id])
        probes["auto"] = response(runtime, "Who founded the Green Belt Movement?")
        assert set(probes["auto"]["candidate_cell_ids"]) == {a.id, b.id}
        assert len(probes["auto"]["selected_cell_ids"]) == 2
        assert probes["single"]["selected_cell_ids"] == [a.id]
        scores = teacher_forced(runtime, a.id)
        assert before == cell_hashes(runtime)
        runtime.cells.export(a.id, args.scratch / "one.zip")
        runtime.cells.export_collection(args.scratch / "all.zip")
        write(args.output / "write.json", dict(cells=[a.id, b.id], hashes=before,
                                               teacher_forced=scores, probes=probes,
                                               model=runtime.model_info))
    elif args.stage == "restore":
        source = read(args.output / "write.json")
        a, b = source["cells"]
        single = runtime.cells.import_cell(args.scratch / "one.zip")
        assert single.id == a
        runtime.cells.delete(a)
        assert {cell.id for cell in runtime.cells.import_collection(args.scratch / "all.zip")} == {a, b}
        assert cell_hashes(runtime) == source["hashes"]
        first = response(runtime, "Who founded the Green Belt Movement?", cells=[a])
        scores = teacher_forced(runtime, a)
        maximum = 0.0
        for before, after in zip(source["teacher_forced"], scores, strict=True):
            torch.testing.assert_close(torch.tensor(before), torch.tensor(after), atol=0.15, rtol=0.03)
            maximum = max(maximum, max(abs(x-y) for x, y in zip(before, after, strict=True)))
        assert cell_hashes(runtime) == source["hashes"]
        assert runtime.cells.append(a, "Maathai was born in Nyeri, Kenya.").revision == 2
        second = response(runtime, "Where was Wangari Maathai born?", cells=[a])
        assert cell_hashes(runtime)[b] == source["hashes"][b]
        write(args.output / "restore.json", dict(max_teacher_forced_delta=maximum,
                                                tolerance=dict(atol=0.15, rtol=0.03),
                                                before_write=first, after_write=second))
    else:
        source = read(args.output / "write.json")
        runtime.cells.import_cell(args.scratch / "one.zip")
        a = source["cells"][0]
        records = []
        for count in (1, 2, 4, 8):
            while len(runtime.cells.list()) < count:
                runtime.cells.fork(a, f"copy-{len(runtime.cells.list())}")
            query = "Who founded the Green Belt Movement?"
            response(runtime, query)
            records.append({"count": count, **response(runtime, query)})
        write(args.output / "profile.json", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("write", "restore", "profile"))
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.stage:
        torch.set_num_threads(2)
        with MiniCells.open(storage=args.scratch / args.stage, dtype="fp16") as runtime:
            phase(runtime, args)
        return
    args.output.mkdir()
    write(args.output / "environment.json", dict(gpu=torch.cuda.get_device_name(),
                                                torch=str(torch.__version__), cuda=torch.version.cuda))
    with tempfile.TemporaryDirectory(prefix="sdk-acceptance-") as scratch:
        for stage in ("write", "restore", "profile"):
            print(f"Stage {stage}", flush=True)
            subprocess.run([sys.executable, __file__, str(args.output), "--stage", stage,
                            "--scratch", scratch], check=True)
    write(args.output / "decision.json", dict(status="GPU_SMOKE_COMPLETE",
        diagnosis="Runtime and restore checks passed. Generated text is retained for readability review.",
        generation_quality="REVIEW_REQUIRED"))


if __name__ == "__main__":
    main()
