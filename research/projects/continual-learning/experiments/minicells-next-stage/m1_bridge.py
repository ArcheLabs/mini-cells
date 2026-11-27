"""M1-002A: isolate single-fact protocol changes before running capacity."""
import argparse
import json
from pathlib import Path
import random
from statistics import mean
import time

import torch

from minicells_native import MiniCells

from common import canonical_hash, environment, publishable, write
import m1_capacity as capacity
from m1_value_swap import DATASET_SHA256 as ANCHOR_SHA256
from m1_value_swap import make_dataset, score, step_prefix

FORMAT = "minicells.m1.capacity-bridge.v1"
DATASET_SHA256 = "3c3f307fef01bafc4222d28b6f15d2a0876e0cb9dd369654c954027bcba8a0d7"
MIN_RATE = 0.8
SEED = 66110
ARMS = ("reference", "anchor", "encoding", "candidates", "wording", "combined")


def build_dataset(tokenizer):
    anchor = make_dataset(tokenizer)
    cases = capacity.build_dataset(tokenizer)
    if canonical_hash(anchor) != ANCHOR_SHA256 or canonical_hash(cases) != capacity.DATASET_SHA256:
        raise ValueError("Source dataset hashes do not match the frozen fixtures")

    def ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def fixture(project, labels, values, new_wording):
        texts = {value: (
            f"Project {project} has archive code {value}" if new_wording else
            f"Project directory. The internal code for Project {project} is {value}."
        ) for value in labels}
        queries = (
            (f"\nQuestion: What is the archive code for Project {project}?\nAnswer:",
             f"\nThe code used to identify Project {project} in the archive is")
            if new_wording else
            (f"\nQuestion: What is the internal code for Project {project}?\nAnswer:",
             f"\nProject {project} can be identified by the following internal code:")
        )
        wordings = []
        for text in queries:
            prefix = ids(text)
            suffixes = {}
            for value in values:
                full = ids(f"{text} {value}")
                if full[:len(prefix)] != prefix or value in text:
                    raise ValueError(f"Invalid answer boundary for {project}")
                suffixes[value] = full[len(prefix):]
            lengths = {len(tokens) for tokens in suffixes.values()}
            vectors = {tuple(tokens) for tokens in suffixes.values()}
            if (len(lengths) != 1 or 0 in lengths or len(vectors) != len(values)
                    or len(prefix) + max(lengths) - 1 > 64):
                raise ValueError(f"Invalid candidate lengths or identity for {project}")
            wordings.append(dict(text=text, tokens=prefix, candidates=suffixes))
        return dict(labels=list(labels), texts=texts, wordings=wordings)

    rng = random.Random(SEED)
    dataset = []
    for ref, case in zip(anchor[:len(cases)], cases, strict=True):
        coded = tuple(ref["values"].values())
        gold = case["gold"]
        alt = rng.choice([word for word in capacity.WORDS if word != gold])
        colors = (gold, alt)
        project = case["project"]
        arms = dict(
            reference=fixture(ref["project"], coded, coded, False),
            anchor=fixture(project, coded, coded, False),
            encoding=fixture(project, colors, colors, False),
            candidates=fixture(project, colors, capacity.WORDS, False),
            wording=fixture(project, colors, colors, True),
            combined=fixture(project, colors, capacity.WORDS, True))
        final = arms["combined"]
        if (final["texts"][gold] != case["target"]
                or any(a["text"] != b["text"] or a["tokens"] != b["tokens"]
                       or {k: t[0] for k, t in a["candidates"].items()} != b["candidates"]
                       for a, b in zip(final["wordings"], case["queries"], strict=True))):
            raise ValueError("Combined arm differs from the capacity task")
        dataset.append(dict(index=case["index"], arms=arms))
    return dataset


def scores(backbone, factory, query):
    prefix, candidates = query["tokens"], query["candidates"]
    if len(next(iter(candidates.values()))) == 1:
        hidden = backbone.step(prefix, [factory()], [1.0])
        logits = backbone.model.lm_head(hidden[0, -1]).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite bridge logits")
        values = logits.log_softmax(-1)[[tokens[0] for tokens in candidates.values()]].tolist()
        return dict(zip(candidates, values, strict=True))
    return {name: score(backbone, factory(), prefix, tokens)["logp"]
            for name, tokens in candidates.items()}


def measure(backbone, factory, arm):
    rows = []
    for query in arm["wordings"]:
        logp = scores(backbone, factory, query)
        margins = {label: logp[label] - max(v for k, v in logp.items() if k != label)
                   for label in arm["labels"]}
        rows.append(dict(query=query["text"], logp=logp, margins=margins))
    return rows


def evaluate(mc, item):
    backbone = mc._backbone
    arms = {}
    shared = {}
    for name, arm in item["arms"].items():
        key = tuple(arm["texts"].values())
        shared.setdefault(key, []).append((name, arm))
    # Candidate subsets share the exact same Cells and query input.
    for group in shared.values():
        labels = group[0][1]["labels"]
        texts = group[0][1]["texts"]
        for name, arm in group:
            arms[name] = dict(no_memory=measure(backbone, lambda: backbone.stream({}), arm),
                              full={}, cell={}, state_sha256={})
        for label in labels:
            text = texts[label]
            tokens = backbone.tokenizer.encode(text, add_special_tokens=False)

            def live():
                return step_prefix(backbone, backbone.stream({}), tokens)

            with capacity.saved_cell(mc, [text], 64, f"{item['index']}-{group[0][0]}") as (factory, digest):
                for name, arm in group:
                    arms[name]["full"][label] = measure(backbone, live, arm)
                    arms[name]["cell"][label] = measure(backbone, factory, arm)
                    arms[name]["state_sha256"][label] = digest
    return dict(index=item["index"], arms=arms)


def summarize(rows):
    arms, hits_by_arm = {}, {}
    for name in ARMS:
        selected = [r["arms"][name] for r in rows]
        lane_stats = {}
        hits_by_arm[name] = {}
        for lane in ("full", "cell"):
            margins = [[row["margins"][label] for label, wordings in arm[lane].items() for row in wordings]
                       for arm in selected]
            hits = [all(margin > 0 for margin in episode) for episode in margins]
            hits_by_arm[name][lane] = hits
            lane_stats.update({f"{lane}_flips": sum(hits),
                               f"{lane}_margin": mean(mean(episode) for episode in margins)})
        prior_hits = sum(
            all(wording["margins"][next(iter(arm["full"]))] > 0 for wording in arm["no_memory"])
            for arm in selected)
        arms[name] = dict(n=len(selected), no_memory_original_hits=prior_hits, **lane_stats)
    contrasts = {}
    for observed, reference in (("anchor", "reference"), ("encoding", "anchor"),
                                ("candidates", "encoding"), ("wording", "encoding"),
                                ("combined", "wording"), ("combined", "candidates")):
        comparison = {}
        for lane in ("full", "cell"):
            paired = list(zip(hits_by_arm[observed][lane], hits_by_arm[reference][lane], strict=True))
            comparison[lane] = dict(gains=sum(a and not b for a, b in paired),
                                    losses=sum(b and not a for a, b in paired))
        contrasts[f"{observed}_vs_{reference}"] = comparison
    gates = {}
    for name in ("reference", "anchor", "combined"):
        for lane in ("full", "cell"):
            key = f"{name}_{lane}"
            gates[key] = "PASS" if arms[name][f"{lane}_flips"] / len(rows) >= MIN_RATE else "FAIL"
    gates["combined_no_memory"] = ("PASS" if arms["combined"]["no_memory_original_hits"] / len(rows)
                                   <= capacity.CONFIG["no_memory_gate"] else "FAIL")
    return dict(arms=arms, contrasts=contrasts, gates=gates)


@torch.inference_mode()
def run(mc, out, result):
    dataset = build_dataset(mc._backbone.tokenizer)
    digest = canonical_hash(dataset)
    result.update(bridge_dataset_sha256=digest, minimum_flip_rate=MIN_RATE,
                  capacity_status="SKIPPED")
    write(out / "bridge_dataset.json", dataset)
    if digest != DATASET_SHA256:
        raise RuntimeError("The pinned tokenizer did not reproduce the frozen bridge dataset")

    rows = []
    with (out / "bridge_cases.jsonl").open("w", encoding="utf-8") as stream:
        for item in dataset:
            row = evaluate(mc, item)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(f"M1_002A_BRIDGE_COMPLETED {len(rows)}/{len(dataset)}", flush=True)

    summary = summarize(rows)
    write(out / "bridge_statistics.json", summary)
    result.update(bridge_summary=summary, gates=summary["gates"])
    if "FAIL" in summary["gates"].values():
        result.update(status="BRIDGE_GATE_FAILED",
                      diagnosis="One or more single-fact bridge controls failed.",
                      next_action="Use the isolated protocol comparisons to identify which change lost recall.")
        print("M1_002A_CAPACITY_SKIPPED bridge gate failed", flush=True)
        return

    print("M1_002A_CAPACITY_STARTED bridge gate passed", flush=True)
    result["capacity_status"] = "STARTED"
    capacity_result = {}
    capacity.run(mc, out, capacity_result)
    result.update(status=capacity_result["status"], capacity_status=capacity_result["status"],
                  diagnosis=capacity_result["diagnosis"], next_action=capacity_result["next_action"],
                  capacity=capacity_result, capacity_summary=capacity_result["capacity_summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002A Single-Fact Protocol Bridge", status="INVALID", engineering_status="FAIL")
    try:
        result["environment"] = environment()
        with MiniCells.open_preview(out / "work" / "registry") as mc:
            result["identity"] = mc.model_info
            run(mc, out, result)
        result["engineering_status"] = "PASS"
    except Exception as exc:
        result.update(status="INVALID", diagnosis=f"{type(exc).__name__}: {exc}")
        if result.get("capacity_status") == "STARTED":
            result["capacity_status"] = "ERROR"
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
