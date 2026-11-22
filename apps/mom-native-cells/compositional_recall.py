"""Development-only two-hop recall on isolated, portable MoM Cells."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import torch

from backend import Backbone
from cells import Cells

SEEDS = (42, 43, 44, 45)
GROUPS_PER_SEED = 4
PAIRS = {
    "A0+B0": ("A0", "B0"),
    "A0+B1": ("A0", "B1"),
    "A1+B0": ("A1", "B0"),
    "A1+B1": ("A1", "B1"),
}
ARMS = {"no_cell": (), **{k: (k,) for k in ("A0", "A1", "B0", "B1")}, **PAIRS}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def dataset(tokenizer):
    # Establish one tokenizer-length bucket before generating any mappings.
    buckets = {}
    for number in range(1000, 2000):
        code = f" RC-{number}"
        _, tokens = candidate_tokens(tokenizer, prompt_for("PJ-2000"), code)
        tokens = tuple(tokens)
        if tokens:
            buckets.setdefault(len(tokens), {})[tokens] = code
    _, pool = max(buckets.items(), key=lambda item: (len(item[1]), -item[0]))
    remaining_codes = sorted(pool.values())
    projects = [f"PJ-{i}" for i in range(2000, 3000)]
    keys = [f"RK-{i}" for i in range(3000, 4000)]

    def draw(rng, pool):
        pair = rng.sample(pool, 2)
        for item in pair:
            pool.remove(item)
        return pair

    groups = []
    for seed in SEEDS:
        rng = random.Random(seed)
        for index in range(GROUPS_PER_SEED):
            project, key, code = (draw(rng, p) for p in (projects, keys, remaining_codes))
            if rng.randrange(2):
                project.reverse()
            # Keep row order fixed across interventions; only mapped values change.
            facts = {}
            for swap in range(2):
                facts[f"A{swap}"] = "\n".join(
                    f"Project {project[i]} is assigned relay key {key[i ^ swap]}." for i in range(2))
                facts[f"B{swap}"] = "\n".join(
                    f"Relay key {key[i]} resolves to recovery code {code[i ^ swap].strip()}."
                    for i in range(2))
            groups.append(dict(group=f"{seed}-{index}", seed=seed, projects=project,
                               keys=key, codes=code, candidates=sorted(code), facts=facts))
    return groups


def prompt_for(project):
    return f"Question: What recovery code is assigned to Project {project}?\nAnswer:"


def candidate_tokens(tokenizer, prompt, answer):
    context = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt + answer)
    if not context or full[:len(context)] != context or len(full) == len(context):
        raise ValueError("Prompt does not end on a candidate token boundary")
    return context, full[len(context):]


@torch.inference_mode()
def score_candidates(backbone, prompt, answers, fresh):
    scores = []
    for answer in answers:
        context, target = candidate_tokens(backbone.tokenizer, prompt, answer)
        caches = fresh()
        hidden = backbone.step(context + target[:-1], caches, [1.0 / len(caches)] * len(caches))
        logits = backbone.model.lm_head(hidden[:, len(context) - 1:]).float()
        logp = logits.log_softmax(-1)[0, range(len(target)), target].tolist()
        if not all(math.isfinite(value) for value in logp):
            raise FloatingPointError("Nonfinite answer scores")
        total = sum(logp)
        scores.append(dict(answer=answer, token_ids=target, token_logp=logp,
                           total_logp=total, mean_logp=total / len(target)))
    return scores


def preflight(tokenizer, groups):
    for group in groups:
        for project in group["projects"]:
            prompt = prompt_for(project)
            tokenizations = [candidate_tokens(tokenizer, prompt, answer)
                             for answer in group["candidates"]]
            (context, left), (_, right) = tokenizations
            if left == right or len(left) != len(right):
                raise ValueError(f"Unequal-length or identical candidate tokens in {group['group']}")
            if len(context) + len(left) - 1 > 64:
                raise ValueError("Query exceeds the T4 recurrent kernel limit")
        for fact in group["facts"].values():
            if not encode(tokenizer, fact):
                raise ValueError("Empty write")


def gold_for(group, project_index, pair):
    a, b = PAIRS[pair]
    return group["codes"][project_index ^ int(a[1]) ^ int(b[1])]


def grade(candidates, gold):
    correct = next(row for row in candidates if row["answer"] == gold)
    distractor = next(row for row in candidates if row["answer"] != gold)
    margin = correct["total_logp"] - distractor["total_logp"]
    return dict(gold=gold, top1=margin > 0, tie=margin == 0, margin=margin,
                gold_total_logp=correct["total_logp"], gold_mean_logp=correct["mean_logp"])


def run_group(backbone, group, states, restoring=False):
    rows = []
    for index, project in enumerate(group["projects"]):
        prompt = prompt_for(project)
        predictions = {}
        for arm, labels in (PAIRS if restoring else ARMS).items():
            def fresh():
                return [backbone.stream(states[label]) for label in labels] or [backbone.stream({})]

            predictions[arm] = score_candidates(backbone, prompt, group["candidates"], fresh)
        rows.append(dict(group=group["group"], seed=group["seed"], project=project, prompt=prompt,
                         project_index=index, scores=predictions))
    return rows


def write_cells(backbone, store, group):
    metadata, tensors = {}, {}
    for label, text in group["facts"].items():
        tokens = encode(backbone.tokenizer, text)
        cache = backbone.stream({})
        for start in range(0, len(tokens), 32):
            backbone.step(tokens[start:start + 32], [cache], [1.0])
        meta = store.create(label)
        state = backbone.persistent(cache)
        if not state:
            raise RuntimeError(f"No recurrent state after writing {label}")
        metadata[label] = store.store({**meta, "revision": 1, "parent_revision": 0,
                                       "tokens_written": len(tokens)}, state)
        tensors[label] = state
    return metadata, tensors


def restore_cells(source, store, metadata):
    tensors = {}
    for label, meta in metadata.items():
        bundle = source.path(meta["cell_id"])
        restored = store.import_bundle(bundle.read_bytes())
        if restored != meta:
            raise RuntimeError(f"Cell provenance changed for {label}")
        tensors[label] = store.read(restored["cell_id"])[1]
    return tensors


def worker(args):
    torch.set_num_threads(2)
    backbone = Backbone(args.source, args.snapshot, args.tokenizer, dtype="float16")
    output = args.output
    restoring = args.worker == "restore"
    groups = read(output / "dataset.json") if restoring else dataset(backbone.tokenizer)
    preflight(backbone.tokenizer, groups)
    name = args.worker
    if not restoring:
        write(output / "dataset.json", groups)
    environment = dict(pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                       torch=str(torch.__version__), cuda=torch.version.cuda, backbone=backbone.identity)
    write(output / f"environment-{name}.json", environment)
    rows, manifests = [], {}
    if restoring:
        manifests = read(output / "cell-manifests.json")
    for group in groups:
        key = group["group"]
        source = Cells(args.bundles / key, backbone)
        print(f"{name}: {key}", flush=True)
        if restoring:
            with tempfile.TemporaryDirectory(dir=args.bundles) as temp:
                states = restore_cells(source, Cells(temp, backbone), manifests[key])
                rows.extend(run_group(backbone, group, states, restoring=True))
        else:
            manifests[key], states = write_cells(backbone, source, group)
            write(output / "cell-manifests.json", manifests)
            rows.extend(run_group(backbone, group, states))
        write(output / f"{name}.json", rows)


def summarize(groups, local, restored):
    group_map = {group["group"]: group for group in groups}
    total = len(groups) * 2
    if len(local) != total or len(restored) != total:
        raise ValueError("Incomplete paired query results")
    condition_counts = {arm: 0 for arm in (*ARMS, "fresh_process")}
    complete, restored_complete, query_groups = set(), set(), set()
    comparisons = []
    restored_by_pair = dict.fromkeys(PAIRS, 0)
    per_seed = {str(group["seed"]): dict(queries=0, all_four=0, fresh_all_four=0, groups=0,
                                        complete_groups=0, fresh_complete_groups=0)
                for group in groups}
    for row, fresh in zip(local, restored, strict=True):
        group = group_map[row["group"]]
        index = row["project_index"]
        key = (row["group"], index)
        if (index not in (0, 1) or key in query_groups
                or any(row[field] != fresh[field] for field in ("project_index", "group", "prompt"))
                or row["project"] != group["projects"][index]):
            raise ValueError("Local and restored results do not pair")
        query_groups.add(key)
        grades, fresh_grades = {}, {}
        for pair, (a, b) in PAIRS.items():
            gold = gold_for(group, index, pair)
            grades[pair] = {arm: grade(row["scores"][arm], gold) for arm in ("no_cell", a, b, pair)}
            fresh_grades[pair] = grade(fresh["scores"][pair], gold)
            direct = grades[pair][pair]
            after = fresh_grades[pair]
            condition_counts[pair] += direct["top1"]
            condition_counts["fresh_process"] += after["top1"]
            restored_by_pair[pair] += after["top1"]
            for control in ("no_cell", a, b):
                condition_counts[control] += grades[pair][control]["top1"]
            comparisons.append(dict(group=key[0], project_index=index, pair=pair,
                                    local=grades[pair], restored=after,
                                    margin_delta=after["margin"] - direct["margin"],
                                    gold_logp_delta=after["gold_total_logp"] - direct["gold_total_logp"]))
        seed = per_seed[str(group["seed"])]
        seed["queries"] += 1
        if all(grades[pair][pair]["top1"] for pair in PAIRS):
            complete.add(key)
            seed["all_four"] += 1
        if all(fresh_grades[pair]["top1"] for pair in PAIRS):
            restored_complete.add(key)
            seed["fresh_all_four"] += 1
    for group in groups:
        seed = per_seed[str(group["seed"])]
        seed["groups"] += 1
        seed["complete_groups"] += all((group["group"], i) in complete for i in range(2))
        seed["fresh_complete_groups"] += all((group["group"], i) in restored_complete for i in range(2))
    denominators = dict(no_cell=total * 4, A0=total * 2, A1=total * 2, B0=total * 2,
                        B1=total * 2, fresh_process=total * 4)
    denominators.update({pair: total for pair in PAIRS})
    return dict(groups=len(groups), queries=total, primary_metric="total_logp",
                conditions={arm: dict(top1=n, total=denominators[arm])
                            for arm, n in condition_counts.items()},
                local_all_four=len(complete), fresh_all_four=len(restored_complete),
                both_all_four=len(complete & restored_complete),
                fresh_by_pair={arm: dict(top1=n, total=total) for arm, n in restored_by_pair.items()},
                restore_rank_changes=sum((row["local"][row["pair"]]["top1"],
                                          row["local"][row["pair"]]["tie"])
                                         != (row["restored"]["top1"], row["restored"]["tie"])
                                         for row in comparisons),
                local_complete_groups=sum(seed["complete_groups"] for seed in per_seed.values()),
                fresh_complete_groups=sum(seed["fresh_complete_groups"] for seed in per_seed.values()),
                max_abs_restore_margin_delta=max(abs(row["margin_delta"]) for row in comparisons),
                max_abs_restore_gold_logp_delta=max(abs(row["gold_logp_delta"]) for row in comparisons),
                per_seed=per_seed), comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "snapshot", "tokenizer", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--worker", choices=("local", "restore"), help=argparse.SUPPRESS)
    parser.add_argument("--bundles", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    total_started = time.monotonic()
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="mom-v2-cells-", dir=args.output.parent) as temp:
        for mode in ("local", "restore"):
            subprocess.run([sys.executable, "-u", __file__, str(args.source), str(args.snapshot),
                            str(args.tokenizer), str(args.output), "--worker", mode, "--bundles",
                            temp], check=True)
    local_env = read(args.output / "environment-local.json")
    fresh_env = read(args.output / "environment-restore.json")
    if (local_env["pid"] == fresh_env["pid"]
            or any(local_env[key] != fresh_env[key] for key in ("backbone", "gpu", "torch", "cuda"))):
        raise RuntimeError("Expected independent processes with matching backbone identity")
    data_path = args.output / "dataset.json"
    summary, comparisons = summarize(read(data_path), read(args.output / "local.json"),
                                     read(args.output / "restore.json"))
    write(args.output / "comparison.json", comparisons)
    write(args.output / "summary.json", summary)
    write(args.output / "decision.json", dict(
        status="DIAGNOSTIC_COMPLETE", scope="DEVELOPMENT_ONLY",
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic() - total_started,
        recommendation="EVALUATE_FACTORIAL_AND_SINGLE_CELL_CONTROLS",
    ))


if __name__ == "__main__":
    main()
