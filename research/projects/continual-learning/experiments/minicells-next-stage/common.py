"""Report and hash utilities for the independent M0 and M1 jobs."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[5]


def canonical_hash(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def environment():
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=sys.version.split()[0],
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        cuda=torch.version.cuda,
        packages={name: version(name) for name in
                  ("torch", "triton", "flash-linear-attention", "transformers", "minicells-native-sdk")},
    )


def publishable(out, fmt, result, started):
    result["elapsed_seconds"] = time.monotonic() - started
    write(out / "result.json", result)
    keys = ("status", "engineering_status", "diagnosis", "next_action")
    if "state_equivalence" in result:
        keys += ("state_equivalence", "bridge_status", "capacity_status")
    write(out / "decision.json", dict(format=fmt, **{k: result[k] for k in keys if k in result}))
    lines = [f"# {result['experiment']}", "", f"**{result['status']}**", "", result["diagnosis"]]
    if result.get("gates"):
        lines += ["", *(f"- {key} — {value}" for key, value in result["gates"].items())]
    stats = result.get("statistics")
    if stats:
        total = stats["facts"]
        low, high = stats["paired_ci95"]
        lines += ["", f"- Full-context flips — {stats['full_context_flip_facts']}/{total}",
                  f"- Recurrent-only Cell flips — {stats['flip_facts']}/{total}",
                  f"- Mean paired margin difference — {stats['paired_mean']:.5g}",
                  f"- Fact-bootstrap 95% interval — [{low:.5g}, {high:.5g}]",
                  f"- No-memory side-preference p — {stats['no_memory_side_bias_p']:.5g}"]
    bridge = result.get("bridge_summary")
    if bridge:
        lines += ["", "| Bridge arm | Full-context swaps | Cell swaps |",
                  "| --- | ---: | ---: |"]
        for name, arm in bridge["arms"].items():
            lines.append(f"| {name} | {arm['full_flips']}/{arm['n']} | {arm['cell_flips']}/{arm['n']} |")
        lines += ["", f"Capacity stage — {result['capacity_status']}"]
    capacity = result.get("capacity_summary")
    if capacity:
        single = capacity["baseline"]["target_only"]
        lines += ["", "| Facts stored | Targets recalled in all conditions | Total targets |",
                  "| --- | ---: | ---: |",
                  f"| 1 | {single['recalled']} | {single['n']} |"]
        for n, level in capacity["levels"].items():
            lines.append(f"| {n} | {level['robust_recalled']} | {single['n']} |")
        lines += ["", "Capacity uses both wordings and, for 3 or more facts, both target orders.",
                  "Per-arm results and paired differences are in `statistics.json`."]
    reference = result.get("reference_capacity")
    if reference:
        lines += ["", "| Facts stored | Full groups | Cell groups | Full facts | Cell facts |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for n, stage in reference.items():
            full, cell = stage["full"], stage["cell"]
            groups, facts = stage["groups"], stage["target_facts"]
            lines.append(f"| {n} | {full['groups_passed']}/{groups} | {cell['groups_passed']}/{groups} "
                         f"| {full['facts_passed']}/{facts} | {cell['facts_passed']}/{facts} |")
    if "state_equivalence" in result:
        lines += ["", f"- State equivalence — {result['state_equivalence']}",
                  f"- Bridge — {result['bridge_status']}", f"- Capacity — {result['capacity_status']}"]
    r1 = result.get("r1_bridge")
    if r1:
        lines += ["", "| Arm | A Live | B Legacy | C Conv | D Cursor | E Full |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm, summary in r1["arms"].items():
            rates = [f"{data['flips']}/{data['n']}" for data in summary["paths"].values()]
            lines.append("| " + " | ".join([arm, *rates]) + " |")
    if result.get("next_action"):
        lines += ["", result["next_action"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(out / "manifest.json", {"artifacts_sha256": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in out.iterdir() if f.is_file() and f.name != "manifest.json"
    }})
