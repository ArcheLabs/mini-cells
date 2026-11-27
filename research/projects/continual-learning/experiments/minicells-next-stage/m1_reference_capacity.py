"""M1-002B: all-fact capacity under the M1-001 value-swap protocol."""
import argparse
from copy import deepcopy
from hashlib import file_digest
import json
from pathlib import Path
import random
from statistics import mean
import time

import torch

from minicells_native import MiniCells

from common import canonical_hash, environment, publishable, write
from m1_value_swap import DATASET_SHA256 as REFERENCE_SHA256
from m1_value_swap import MAX_CHUNK, make_dataset, paired_scores, step_prefix

FORMAT = "minicells.m1.reference-capacity.v1"
DATASET_SHA256 = "6b2560de959dbf71fffaf955c96e582b23d3beda87fd4b6979bfe43fe58d05e7"
LEVELS = (1, 3, 8, 16)
GROUPS = 12
MIN_GROUPS = 10
FACT_SEEDS = {3: 66203, 8: 66208, 16: 66216}
ORDER_SEED = 66200
BRIDGE_COMMIT = "d9fca132573de69da91b164dcdc44f66e7b41ca3"


def assemble(facts, tokenizer):
    """Keep every distractor unchanged and match write tokens across the swap."""
    original = " ".join(fact["texts"]["A"] for fact in facts)
    tokens = [token for fact in facts for token in fact["write_tokens"]["A"]]
    if tokenizer.encode(original, add_special_tokens=False) != tokens:
        raise ValueError("Original write changed tokenization at a record boundary")
    swaps = []
    offset = 0
    for index, fact in enumerate(facts):
        replacement = fact["write_tokens"]["B"]
        count = len(fact["write_tokens"]["A"])
        swapped = " ".join(row["texts"]["B" if i == index else "A"]
                           for i, row in enumerate(facts))
        expected = tokens[:offset] + replacement + tokens[offset + count:]
        if (len(replacement) != count or expected == tokens
                or tokenizer.encode(swapped, add_special_tokens=False) != expected):
            raise ValueError("Swap altered the length or tokens outside the target record")
        swaps.append(dict(text=swapped, tokens=expected, offset=offset))
        offset += count
    return dict(order=[fact["project"] for fact in facts], facts=facts,
                original=dict(text=original, tokens=tokens), swaps=swaps)


def build_dataset(tokenizer):
    reference = make_dataset(tokenizer)
    if canonical_hash(reference) != REFERENCE_SHA256:
        raise ValueError("Original M1-001 fixture hash does not match")
    result = {"1": [assemble([fact], tokenizer) for fact in reference[:GROUPS]]}
    used = {fact["project"] for fact in reference}
    rng = random.Random(ORDER_SEED)
    for n in LEVELS[1:]:
        facts = make_dataset(tokenizer, seed=FACT_SEEDS[n], facts_count=GROUPS * n)
        if used.intersection(fact["project"] for fact in facts):
            raise ValueError("Project identifiers must be distinct across capacity levels")
        used.update(fact["project"] for fact in facts)
        result[str(n)] = []
        for i in range(GROUPS):
            group = facts[i*n:(i+1)*n]
            rng.shuffle(group)
            result[str(n)].append(assemble(group, tokenizer))
    return result


def live_factory(backbone, tokens):
    full = step_prefix(backbone, backbone.stream({}), tokens)
    layers, cursor = full.to_legacy_cache(), full.get_seq_length()

    def fresh():
        return backbone.cache_type.from_legacy_cache(deepcopy(layers), seen_tokens=cursor)

    return fresh


def read_condition(backbone, factory, fact, written):
    return {
        query_text: paired_scores(backbone, factory, query, written)
        for query_text, query in fact["queries"].items()
    }


def digest_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def cell_condition(mc, name, record, fact, written):
    cell = mc.cells.create(name)
    try:
        mc.cells.write(cell.id, record["text"], chunk_size=MAX_CHUNK, resume=False)
        bundle = mc.cells._store.path(cell.id)
        before = digest_file(bundle)
        meta, tensors = mc.cells._store.read(cell.id)
        if meta["tokens_written"] != len(record["tokens"]):
            raise ValueError("Cell wrote a different token count")
        rows = read_condition(mc._backbone, lambda: mc._backbone.stream(tensors), fact, written)
        if digest_file(bundle) != before:
            raise RuntimeError("Read changed the Cell bundle")
        return rows, dict(cell_id=cell.id, state_sha256=meta["tensor_sha256"], bundle_sha256=before)
    finally:
        mc.cells.delete(cell.id)


def evaluate_target(mc, index, level, group, position, full_original):
    backbone = mc._backbone
    fact = group["facts"][position]
    original, swapped = group["original"], group["swaps"][position]
    row = dict(group=index, facts_per_cell=level, position=position+1, project=fact["project"],
               values=fact["values"], write_order=group["order"], total_tokens=len(original["tokens"]),
               target_token_offset=swapped["offset"], state={}, wordings={})
    label = f"{level}-{index}-{position}"
    conditions = {
        "full_original": read_condition(backbone, full_original, fact, "A"),
        "full_swapped": read_condition(backbone, live_factory(backbone, swapped["tokens"]), fact, "B"),
    }
    for name, record, written in (("cell_original", original, "A"), ("cell_swapped", swapped, "B")):
        conditions[name], row["state"][name] = cell_condition(mc, f"{label}-{written}", record, fact, written)
    if level == 1:
        conditions["no_memory"] = read_condition(backbone, lambda: backbone.stream({}), fact, None)
    row["wordings"] = {text: {name: results[text] for name, results in conditions.items()}
                       for text in fact["queries"]}
    if not all(arm["order_equal"] for word in row["wordings"].values() for arm in word.values()):
        raise RuntimeError("Candidate evaluation order changed a score")
    for lane in ("full", "cell"):
        row[f"{lane}_flip"] = all(word[f"{lane}_original"]["margin"] > 0
                                  and word[f"{lane}_swapped"]["margin"] < 0
                                  for word in row["wordings"].values())
    return row


def summarize(groups):
    flat = [row for group in groups for row in group]
    return {
        "groups": len(groups),
        "target_facts": len(flat),
        **{
            lane: dict(groups_passed=sum(all(row[f"{lane}_flip"] for row in group) for group in groups),
                       facts_passed=sum(row[f"{lane}_flip"] for row in flat),
                       position_passed={str(position): sum(row[f"{lane}_flip"] for row in flat
                                                           if row["position"] == position)
                                        for position in range(1, len(groups[0])+1)},
                       mean_gold_nll=mean(
                           word[f"{lane}_{condition}"]["gold_nll"]
                           for row in flat for word in row["wordings"].values()
                           for condition in ("original", "swapped")))
            for lane in ("full", "cell")
        },
    }


def decide(stats):
    first_fail = next((str(n) for n in LEVELS
                       if min(stats[str(n)][lane]["groups_passed"] for lane in ("full", "cell")) < MIN_GROUPS), None)
    if first_fail is None:
        return "PASS", "All four reference-protocol capacity gates passed.", "Freeze a separate 20–30-fact confirmation."
    if first_fail == "1":
        return "SINGLE_FACT_GATE_FAILED", "The mandatory single-fact gate failed.", "Recheck the single-fact bridge."
    stage = stats[first_fail]
    if stage["full"]["groups_passed"] < MIN_GROUPS:
        return ("FULL_CONTEXT_GATE_FAILED", f"The full-context control failed at {first_fail} facts.",
                "Diagnose the multi-fact full-context control before attributing the result to Cell interference.")
    actions = {"3": "Investigate multi-fact interference at the first capacity increase.",
               "8": "Investigate the decline between 3 and 8 facts.",
               "16": "Test an intermediate capacity between 8 and 16 facts."}
    return "CAPACITY_GATE_FAILED", f"The Cell gate first failed at {first_fail} facts.", actions[first_fail]


@torch.inference_mode()
def run(mc, out, result):
    dataset = build_dataset(mc._backbone.tokenizer)
    digest = canonical_hash(dataset)
    write(out / "dataset.json", dataset)
    result.update(dataset_sha256=digest, bridge_commit=BRIDGE_COMMIT, minimum_groups=MIN_GROUPS,
                  fact_seeds=FACT_SEEDS, order_seed=ORDER_SEED, reference_sha256=REFERENCE_SHA256)
    if digest != DATASET_SHA256:
        raise ValueError("Tokenizer did not reproduce the frozen M1-002B dataset")
    stats = {}
    result.update(reference_capacity=stats, levels_completed=[])
    with (out / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for level in LEVELS:
            groups = []
            print(f"M1_002B_STAGE_STARTED {level}", flush=True)
            for index, group in enumerate(dataset[str(level)], 1):
                original = live_factory(mc._backbone, group["original"]["tokens"])
                evaluated = []
                for position in range(level):
                    row = evaluate_target(mc, index, level, group, position, original)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    evaluated.append(row)
                    print(f"M1_002B_PROGRESS facts={level} group={index}/{GROUPS} "
                          f"target={position+1}/{level}", flush=True)
                groups.append(evaluated)
            stats[str(level)] = summarize(groups)
            result["levels_completed"].append(level)
            write(out / "statistics.json", stats)
            summary = stats[str(level)]
            print(f"M1_002B_STAGE_COMPLETED {level} full={summary['full']['groups_passed']}/{GROUPS} "
                  f"cell={summary['cell']['groups_passed']}/{GROUPS}", flush=True)
            if level == 1 and min(summary[lane]["groups_passed"] for lane in ("full", "cell")) < MIN_GROUPS:
                print("M1_002B_CAPACITY_SKIPPED single-fact gate failed", flush=True)
                break
    status, diagnosis, next_action = decide(stats)
    result.update(status=status, diagnosis=diagnosis, next_action=next_action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002B Reference-Protocol Capacity", status="INVALID", engineering_status="FAIL")
    try:
        result["environment"] = environment()
        with MiniCells.open_preview(out / "work" / "registry") as mc:
            result["identity"] = mc.model_info
            if (result["environment"]["gpu"] != "Tesla T4"
                    or result["environment"]["capability"] != [7, 5]
                    or result["identity"]["dtype"] != "float16"):
                raise RuntimeError("M1-002B requires Tesla T4 with the FP16 developer preview")
            run(mc, out, result)
        result["engineering_status"] = "PASS"
    except Exception as exc:
        result.update(status="INVALID", diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
