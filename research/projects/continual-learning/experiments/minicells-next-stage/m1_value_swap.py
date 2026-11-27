"""M1-001 fixed-seed, fact-level value-swap development gate."""
import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
import string
import time

import torch

from minicells_native import MiniCells
from common import canonical_hash, environment, publishable, write

FORMAT = "minicells.m1.value-swap.v1"
FACTS = 30
SEED = 47031
BOOTSTRAP_SEED = 87293
BOOTSTRAP_RESAMPLES = 10_000
MIN_FLIPS = 27
MAX_CHUNK = 64
DATASET_SHA256 = "413dd77f2ca9e40d0d4ab5b9d0b83dc1478ffe771b66f65c4f790232bb078913"
ORDER = ("A", "B")
ARMS = ("no_memory", "full_original", "full_swapped", "cell_original", "cell_swapped")
CONFIG = dict(facts=FACTS, seed=SEED, bootstrap_seed=BOOTSTRAP_SEED,
              bootstrap_resamples=BOOTSTRAP_RESAMPLES, minimum_paired_flips=MIN_FLIPS,
              bias_alpha=.05, chunk=MAX_CHUNK, minimum_wordings=2,
              score="teacher-forced summed log probability", bos=False, dataset_sha256=DATASET_SHA256)


def make_dataset(tokenizer, *, seed=SEED, facts_count=FACTS):
    rng = random.Random(seed)
    used = set()
    facts = []

    def ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def code():
        return "".join(rng.choices(string.ascii_uppercase, k=3)) + "-" + str(rng.randrange(1000, 10_000))

    for _ in range(10_000):
        name = f"Novera-{rng.randrange(10_000_000, 99_999_999)}"
        a, b = code(), code()
        if a == b or {name, a, b} & used:
            continue
        values = dict(A=a, B=b)
        queries = [
            f"\nQuestion: What is the internal code for Project {name}?\nAnswer:",
            f"\nProject {name} can be identified by the following internal code:",
        ]
        targets = {}
        for query in queries:
            prefix = ids(query)
            continuations = {key: ids(f"{query} {value}") for key, value in values.items()}
            if any(tokens[:len(prefix)] != prefix for tokens in continuations.values()):
                break
            suffixes = {key: tokens[len(prefix):] for key, tokens in continuations.items()}
            if (not suffixes["A"] or suffixes["A"] == suffixes["B"]
                    or len(suffixes["A"]) != len(suffixes["B"])
                    or len(prefix) + len(suffixes["A"]) - 1 > MAX_CHUNK):
                break
            if any(value in query for value in values.values()):
                raise RuntimeError("Candidate value leaked into its query")
            targets[query] = dict(query_ids=prefix, candidates=suffixes)
        if len(targets) != len(queries):
            continue
        texts = {key: f"Project directory. The internal code for Project {name} is {value}."
                 for key, value in values.items()}
        if len(ids(texts["A"])) != len(ids(texts["B"])):
            continue
        used.update((name, a, b))
        facts.append(dict(index=len(facts), project=name, values=values, texts=texts, queries=targets,
                          write_tokens={key: ids(text) for key, text in texts.items()}))
        if len(facts) == facts_count:
            return facts
    raise RuntimeError("Unable to construct the frozen equal-token fixture")


def step_prefix(backbone, cache, tokens):
    for start in range(0, len(tokens), MAX_CHUNK):
        backbone.step(tokens[start:start + MAX_CHUNK], [cache], [1.0])
    return cache


def score(backbone, cache, prefix, candidate):
    hidden = backbone.step(prefix + candidate[:-1], [cache], [1.0])
    logits = backbone.model.lm_head(hidden[0, len(prefix) - 1:]).float()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite candidate logits")
    logp = logits.log_softmax(-1)[list(range(len(candidate))), candidate].tolist()
    return dict(token_logp=logp, logp=sum(logp))


def paired_scores(backbone, factory, query, correct):
    rounds = []
    for order in (ORDER, ORDER[::-1]):
        scores = {key: score(backbone, factory(), query["query_ids"], query["candidates"][key])
                  for key in order}
        rounds.append(scores)
    a, b = rounds[0]["A"], rounds[0]["B"]
    return dict(candidates=rounds[0], reversed_candidates=rounds[1], order_equal=rounds[0] == rounds[1],
                margin=a["logp"] - b["logp"], written_candidate=correct,
                gold_nll=None if correct is None else -rounds[0][correct]["logp"] /
                len(query["candidates"][correct]))


def bootstrap(values):
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES))
    return [estimates[249], estimates[9749]]


def all_wordings(row, original, swapped):
    return all(q["arms"][original]["margin"] > 0 and q["arms"][swapped]["margin"] < 0
               for q in row["wordings"])


def summarize(rows):
    cell = sum(all_wordings(r, "cell_original", "cell_swapped") for r in rows)
    full = sum(all_wordings(r, "full_original", "full_swapped") for r in rows)
    delta = [mean(q["arms"]["cell_original"]["margin"] - q["arms"]["cell_swapped"]["margin"]
                  for q in r["wordings"]) for r in rows]
    prior = [mean(q["arms"]["no_memory"]["margin"] for q in r["wordings"]) for r in rows]
    pos, neg = sum(m > 0 for m in prior), sum(m < 0 for m in prior)
    n = pos + neg
    bias_p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2**n)
    ci = bootstrap(delta)
    order = all(arm["order_equal"] for r in rows for q in r["wordings"] for arm in q["arms"].values())
    gates = dict(
        full_context="PASS" if full >= MIN_FLIPS else "FAIL",
        paired_flip="PASS" if cell >= MIN_FLIPS else "FAIL",
        paired_margin_ci="PASS" if ci[0] > 0 else "FAIL",
        no_memory_side_bias="PASS" if bias_p >= CONFIG["bias_alpha"] else "FAIL",
        candidate_order="PASS" if order else "FAIL",
    )
    return dict(facts=len(rows), flip_facts=cell, full_context_flip_facts=full, paired_mean=mean(delta),
                paired_ci95=ci, no_memory_A_preferred=pos, no_memory_B_preferred=neg,
                no_memory_ties=len(rows)-n, no_memory_side_bias_p=bias_p, gates=gates)


@torch.inference_mode()
def run(mc, out, result):
    backbone = mc._backbone
    dataset = make_dataset(backbone.tokenizer)
    actual_hash = canonical_hash(dataset)
    result.update(dataset_sha256=actual_hash, configuration_sha256=canonical_hash(CONFIG), configuration=CONFIG)
    write(out / "dataset.json", dataset)
    if actual_hash != DATASET_SHA256:
        raise RuntimeError("Tokenizer did not reproduce the frozen Value Swap dataset")
    rows = []
    with (out / "cases.jsonl").open("w", encoding="utf-8") as output:
        for item in dataset:
            cells = {k: mc.cells.write(mc.cells.create(f"{item['index']}-{k}").id, text, resume=False)
                     for k, text in item["texts"].items()}
            states = {k: mc.cells._state(c.id) for k, c in cells.items()}
            stored = {k: mc.cells._store.read(c.id)[0]["tensor_sha256"] for k, c in cells.items()}
            factories = {
                "no_memory": lambda: backbone.stream({}),
                "full_original": lambda: step_prefix(backbone, backbone.stream({}), item["write_tokens"]["A"]),
                "full_swapped": lambda: step_prefix(backbone, backbone.stream({}), item["write_tokens"]["B"]),
                "cell_original": lambda: backbone.stream(states["A"]),
                "cell_swapped": lambda: backbone.stream(states["B"]),
            }
            written = (None, "A", "B", "A", "B")
            wordings = [dict(query=text, arms={name: paired_scores(backbone, factories[name], query, key)
                        for name, key in zip(ARMS, written, strict=True)})
                        for text, query in item["queries"].items()]
            if stored != {k: mc.cells._store.read(c.id)[0]["tensor_sha256"] for k, c in cells.items()}:
                raise RuntimeError("Query changed a saved Cell")
            row = dict(index=item["index"], cell_ids={k: c.id for k, c in cells.items()},
                       state_sha256=stored, wordings=wordings)
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            for cell in cells.values():
                mc.cells.delete(cell.id)
            print(f"M1_FACT_COMPLETED {len(rows)}/{FACTS}", flush=True)
    summary = summarize(rows)
    write(out / "statistics.json", summary)
    result.update(statistics=summary, gates=summary["gates"])
    gates = summary["gates"]
    if any(gates[key] == "FAIL" for key in ("full_context", "candidate_order")):
        result.update(status="INVALID", diagnosis="The full-context or candidate-order control failed.",
                      next_action="Diagnose the measurement before drawing a Cell conclusion.")
    elif all(value == "PASS" for value in gates.values()):
        result.update(status="PASS", diagnosis="The frozen fact-level Value Swap gate passed.",
                      next_action="Review evidence before opening M1-002 capacity.")
    else:
        result.update(status="FAIL", diagnosis="The frozen Value Swap or baseline-bias gate failed.",
                      next_action="Pause capacity and diagnose the single-Cell path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-001 Value Swap", status="INVALID", engineering_status="FAIL")
    try:
        result["environment"] = environment()
        with MiniCells.open_preview(out / "work" / "registry") as mc:
            result["identity"] = mc.model_info
            run(mc, out, result)
        result["engineering_status"] = "PASS"
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
