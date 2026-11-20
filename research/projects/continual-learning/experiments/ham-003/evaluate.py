"""Paired CPU standalone screening for HAM-003."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import subprocess
import platform

import numpy as np

from tasks import generate

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / n) for n in ("ham-002", "ham-001")]
from benchmark import Attention, Config as Geometry, score
from memory import Config, DeltaMemory

TASKS = ("basic", "correlated", "composition", "parity", "conditional",
         "multi_hop", "open_set", "continual")


def evaluate(cell, name, data, threshold):
    keys, values, queries, targets, meta = data
    retention = []
    for i, (k, v) in enumerate(zip(keys, values), 1):
        cell.write(k, v)
        if name == "continual" and i % 8 == 0:
            retention.append(score(cell.read(queries[:8]), targets[:8])["top1"])
            restored = cell.fresh()
            restored.import_state(cell.export_state())
            np.testing.assert_array_equal(restored.read(queries), cell.read(queries))
            cell = restored
    prediction = cell.read(queries)
    if name == "multi_hop":
        visited = []
        for hop in range(3):
            if hop:
                prediction = cell.read(values[visited[-1], np.arange(keys.shape[1])])
            distances = np.square(prediction[:, None] - values).mean(-1)
            visited.append(distances.argmin(1))
        expected = np.arange(len(queries))[:, None] + np.arange(3)[None, :]
        return {"primary": float((visited[-1] == expected[:, 2:3]).mean()),
                "first_hop": float((visited[0] == expected[:, :1]).mean())}
    if name in ("parity", "conditional"):
        return {"primary": float((np.sign(prediction[..., 0]) == targets[..., 0]).mean()),
                "mse": float(np.square(prediction - targets).mean())}
    if name == "open_set":
        confidence = np.linalg.norm(prediction, axis=-1)
        n = meta["known"]
        accepted = confidence >= threshold
        return {"primary": float((accepted[:n].mean() + (~accepted[n:]).mean()) / 2),
                "known_accept": float(accepted[:n].mean()),
                "unknown_false_accept": float(accepted[n:].mean()),
                "known_top1": score(prediction[:n], targets[:n])["top1"]}
    result = score(prediction, targets)
    return dict(primary=result["top1"], **result, a_retention=retention)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--titans-upstream", type=Path)
    args = parser.parse_args()
    arms = {"delta": lambda _: DeltaMemory(Config(beta=.5)),
            "full_attention_reference": lambda _: Attention(Geometry())}
    unavailable = {}
    if args.titans_upstream:
        import torch
        from titans_adapter import Titans, load_upstream, REVISION as upstream_revision
        torch.set_num_threads(2)
        upstream = load_upstream(args.titans_upstream)
        arms["titans"] = lambda seed: Titans(Geometry(), upstream, seed)
    else:
        unavailable["titans"] = "Pinned upstream and CPU dependencies required"
    rows = []
    for seed in (101, 102, 103):
        for task in TASKS:
            data = generate(seed, task)
            digest = hashlib.sha256(b"".join(x.tobytes() for x in data[:4])).hexdigest()
            for arm, create in arms.items():
                cell = create(seed)
                metrics = evaluate(cell, task, data, threshold=.25)
                rows.append(dict(seed=seed, task=task, arm=arm, dataset_sha256=digest,
                                 metrics=metrics, **cell.get_stats()))
    summary = []
    for task in TASKS:
        for arm in arms:
            v = [r["metrics"]["primary"] for r in rows if r["task"] == task and r["arm"] == arm]
            half = 4.302653 * stdev(v) / np.sqrt(len(v))
            summary.append(dict(task=task, arm=arm, mean=mean(v), sd=stdev(v),
                                ci95=[mean(v)-half, mean(v)+half]))
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(
        dict(rows=rows, summary=summary, unavailable=unavailable, device="cpu",
             open_set_norm_threshold=.25, stage="STANDALONE_SCREENING",
             source_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
             platform=platform.platform(), python=sys.version, numpy=np.__version__,
             titans_revision=None if not args.titans_upstream else upstream_revision,
             evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        indent=2, allow_nan=False))

    (args.output / "decision.json").write_text(json.dumps(dict(
        format="minicells.ham-003.v1", status="STANDALONE_SCREENING",
        diagnostic_status="COMPLETE", scientific_status="PENDING",
        unavailable=unavailable, hybrid_started=False), indent=2))


if __name__ == "__main__":
    main()
