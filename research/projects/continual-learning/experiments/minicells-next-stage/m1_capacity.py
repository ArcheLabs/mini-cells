"""M1-002: paired single-Cell capacity and interference on the bounded runtime."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import random
from statistics import mean
import time

import torch

from minicells_native import MiniCells

from common import canonical_hash, environment, publishable, write
from m1_value_swap import step_prefix

FORMAT = "minicells.m1.capacity.v1"
VALUE_SWAP_RESULTS = "7744e8c63dcb2b7085ec58b9a9a948bb694eec14"
DATASET_SHA256 = "a02e726996e1f257e73a80cb5b923f5a789fa1016ed93a1287570c106e7937fb"
SEED = 66109
EPISODES = 12
LEVELS = (1, 3, 8, 16)
WORDS = ("bronze", "dawn", "gold", "green", "mint", "pine", "silver", "stone", "white", "red",
         "blue", "black", "yellow", "purple", "orange", "pink", "brown", "gray", "rose", "lemon")
NEUTRAL = ("The page holds plain text and a small mark in blue ink and has a line of print at the "
           "side of the sheet near the top where the light shines through the glass").split()
CONFIG = dict(seed=SEED, episodes=EPISODES, facts_per_cell=list(LEVELS), candidates=len(WORDS),
              wordings=2, chunk_sizes=[64, 32], exact_recall_gate=.80, no_memory_gate=.10,
              first_stage_max_facts=16, same_program_as_m1_001=VALUE_SWAP_RESULTS)


def chunk_lengths(size, limit):
    return [min(limit, size - start) for start in range(0, size, limit)]


def build_dataset(tokenizer):
    rng = random.Random(SEED)

    def ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    neutral = {len(ids(" ".join(NEUTRAL[:n]))): " ".join(NEUTRAL[:n])
               for n in range(1, len(NEUTRAL) + 1)}
    projects = rng.sample(range(10_000_000, 99_999_999), EPISODES * max(LEVELS))
    rows = []
    for i in range(EPISODES):
        numbers = projects[i * 16:(i + 1) * 16]
        values = rng.sample(WORDS, k=16)
        names = [f"Ravena-{n}" for n in numbers]
        lines = [f"Project {name} has archive code {word}" for name, word in zip(names, values, strict=True)]
        target, *distractors = lines
        query_rows = []
        for text in (f"\nQuestion: What is the archive code for Project {names[0]}?\nAnswer:",
                     f"\nThe code used to identify Project {names[0]} in the archive is"):
            prefix = ids(text)
            candidates = {}
            for word in WORDS:
                tokens = ids(f"{text} {word}")
                if tokens[:len(prefix)] != prefix or len(tokens) != len(prefix) + 1:
                    raise ValueError(f"Expected one stable candidate token for {word!r}")
                candidates[word] = tokens[-1]
            if len(set(candidates.values())) != len(WORDS) or not 1 <= len(prefix) <= 64:
                raise ValueError("Invalid candidate or query boundaries")
            query_rows.append(dict(text=text, tokens=prefix, candidates=candidates))

        orders = {}
        for n in LEVELS[1:]:
            choices = distractors[:n-1]
            permutations = {}
            for label in ("target_first", "target_last"):
                others = rng.sample(choices, k=len(choices))
                pieces = [target, *others] if label == "target_first" else [*others, target]
                neutral_pieces = [piece if piece == target else neutral[len(ids(piece))] for piece in pieces]
                factual_ids = [ids(piece) for piece in pieces]
                neutral_ids = [ids(piece) for piece in neutral_pieces]
                if list(map(len, factual_ids)) != list(map(len, neutral_ids)):
                    raise ValueError("Neutral token count differs from factual input")
                joined = ids(" ".join(pieces))
                if (joined != [token for part in factual_ids for token in part]
                        or ids(" ".join(neutral_pieces)) != [token for part in neutral_ids for token in part]):
                    raise ValueError("Unstable record boundary for the append control")
                split = n // 2
                batches = [" ".join(pieces[:split]), " ".join(pieces[split:])]
                chunks = [ids(piece) for piece in batches]
                if chunks[0] + chunks[1] != joined:
                    raise ValueError("Append changed the token sequence")
                schedules = dict(
                    factual=chunk_lengths(len(joined), 64),
                    split32=chunk_lengths(len(joined), 32),
                    append=[chunk_lengths(len(batch), 64) for batch in chunks],
                )
                if schedules["factual"] == schedules["split32"]:
                    raise ValueError("Segmentation control must exercise distinct schedules")
                permutations[label] = dict(factual=" ".join(pieces), neutral=" ".join(neutral_pieces),
                                           batches=batches, batch_ids=chunks, tokens=joined, schedules=schedules,
                                           neutral_tokens=ids(" ".join(neutral_pieces)))
            orders[str(n)] = permutations
        rows.append(dict(index=i, project=names[0], gold=values[0], queries=query_rows,
                         target=target, target_ids=ids(target), orders=orders))
    return rows


def measure(backbone, factory, item):
    wordings = []
    for row in item["queries"]:
        cache = factory()
        start = cache.get_seq_length()
        hidden = backbone.step(row["tokens"], [cache], [1.0])
        logits = backbone.model.lm_head(hidden[0, -1]).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite capacity logits")
        probs = logits.log_softmax(-1)
        scores = dict(zip(row["candidates"], probs[list(row["candidates"].values())].tolist(), strict=True))
        gold = scores[item["gold"]]
        margin = gold - max(v for word, v in scores.items() if word != item["gold"])
        wordings.append(dict(query=row["text"], logp=scores, margin=margin, gold_nll=-gold,
                             correct=margin > 0, start_cursor=start, end_cursor=cache.get_seq_length()))
    return dict(wordings=wordings, recalled=all(row["correct"] for row in wordings),
                mean_margin=mean(row["margin"] for row in wordings),
                mean_gold_nll=mean(row["gold_nll"] for row in wordings))


@contextmanager
def saved_cell(mc, texts, chunk, name):
    cell = mc.cells.create(name)
    try:
        mc.cells.write(cell.id, texts[0], chunk_size=chunk, resume=False)
        for text in texts[1:]:
            mc.cells.append(cell.id, text, chunk_size=chunk, resume=False)
        bundle = mc.cells._store.path(cell.id)
        with bundle.open("rb") as stream:
            before = file_digest(stream, "sha256").hexdigest()
        state = mc.cells._state(cell.id)
        yield lambda: mc._backbone.stream(state), before
        with bundle.open("rb") as stream:
            after = file_digest(stream, "sha256").hexdigest()
        if before != after:
            raise RuntimeError("Read modified the stored Cell bundle")
    finally:
        mc.cells.delete(cell.id)


def evaluate_item(mc, item):
    backbone = mc._backbone

    def score_cell(texts, chunk, label):
        with saved_cell(mc, texts, chunk, f"{item['index']}-{label}") as (factory, bundle_hash):
            return dict(measure(backbone, factory, item), bundle_sha256=bundle_hash)

    def live():
        return step_prefix(backbone, backbone.stream({}), item["target_ids"])

    baseline = dict(no_memory=measure(backbone, lambda: backbone.stream({}), item),
                    full_context=measure(backbone, live, item),
                    target_only=score_cell([item["target"]], 64, "target"))
    levels = {}
    for n, options in item["orders"].items():
        outcomes = {}
        for order, text in options.items():
            prefix = f"{n}-{order}"
            runs = dict(
                factual=score_cell([text["factual"]], 64, f"{prefix}-facts"),
                neutral=score_cell([text["neutral"]], 64, f"{prefix}-neutral"),
                split32=score_cell([text["factual"]], 32, f"{prefix}-split"),
                append=score_cell(text["batches"], 64, f"{prefix}-append"))
            # Same tokens and per-forward boundaries as Append, with one continuous stream.
            full = backbone.stream({})
            for batch in text["batch_ids"]:
                step_prefix(backbone, full, batch)
            state = backbone.persistent(full)
            runs["continuous_append_schedule"] = measure(backbone, lambda: backbone.stream(state), item)
            outcomes[order] = runs
        levels[n] = outcomes
    return dict(index=item["index"], project=item["project"], gold=item["gold"], baseline=baseline, levels=levels)


def summarize(rows):
    def aggregate(items):
        return dict(recalled=sum(r["recalled"] for r in items), n=len(items),
                    accuracy=mean(r["recalled"] for r in items),
                    margin=mean(r["mean_margin"] for r in items),
                    gold_nll=mean(r["mean_gold_nll"] for r in items))

    def paired(observed, reference):
        return dict(
            mean_delta_margin=mean(a["mean_margin"]-b["mean_margin"] for a,b in zip(observed, reference, strict=True)),
            mean_delta_gold_nll=mean(a["mean_gold_nll"]-b["mean_gold_nll"] for a,b in zip(observed, reference, strict=True)),
            recall_losses=sum(b["recalled"] and not a["recalled"] for a,b in zip(observed, reference, strict=True)),
            recall_gains=sum(a["recalled"] and not b["recalled"] for a,b in zip(observed, reference, strict=True)))

    baseline = {arm: aggregate([row["baseline"][arm] for row in rows]) for arm in rows[0]["baseline"]}
    levels = {}
    for level in map(str, LEVELS[1:]):
        arms = defaultdict(list)
        for row in rows:
            for outcomes in row["levels"][level].values():
                for arm, value in outcomes.items():
                    arms[arm].append(value)
        target = [row["baseline"]["target_only"] for row in rows for _ in range(2)]
        first = [row["levels"][level]["target_first"]["factual"] for row in rows]
        last = [row["levels"][level]["target_last"]["factual"] for row in rows]
        stable = sum(a["recalled"] and b["recalled"] for a, b in zip(first,last,strict=True))
        levels[level] = dict(
            arms={arm: aggregate(values) for arm, values in arms.items()},
            robust_recalled=stable, robust_accuracy=stable / len(rows),
            interference_rate=mean(b["recalled"] and not a["recalled"]
                                   for a,b in zip(arms["factual"],arms["neutral"],strict=True)),
            contrasts=dict(
                extra_tokens=paired(arms["neutral"], target),
                extra_facts_at_equal_length=paired(arms["factual"], arms["neutral"]),
                prefill_segmentation=paired(arms["split32"], arms["factual"]),
                append_session_reset=paired(arms["append"], arms["continuous_append_schedule"]),
                order_target_last_vs_first=paired(last, first)))
    gates = dict(full_context="PASS" if baseline["full_context"]["accuracy"] >= CONFIG["exact_recall_gate"] else "FAIL",
                 no_memory="PASS" if baseline["no_memory"]["accuracy"] <= CONFIG["no_memory_gate"] else "FAIL",
                 single_fact="PASS" if baseline["target_only"]["accuracy"] >= CONFIG["exact_recall_gate"] else "FAIL",
                 robust_capacity="PASS" if all(r["robust_accuracy"] >= CONFIG["exact_recall_gate"] for r in levels.values()) else "FAIL")
    return dict(baseline=baseline, levels=levels, gates=gates)


@torch.inference_mode()
def run(mc, out, result):
    dataset = build_dataset(mc._backbone.tokenizer)
    actual = canonical_hash(dataset)
    write(out / "dataset.json", dataset)
    result.update(configuration=CONFIG, configuration_sha256=canonical_hash(CONFIG), dataset_sha256=actual)
    if actual != DATASET_SHA256:
        raise RuntimeError("The pinned tokenizer did not reproduce the frozen capacity dataset")
    rows = []
    with (out / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for item in dataset:
            row = evaluate_item(mc, item)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(f"M1_002_TARGET_COMPLETED {len(rows)}/{len(dataset)}", flush=True)
    stats = summarize(rows)
    write(out / "statistics.json", stats)
    result["gates"] = gates = stats["gates"]
    result["capacity_summary"] = stats
    if gates["full_context"] == "FAIL" or gates["no_memory"] == "FAIL":
        result.update(status="INVALID", diagnosis="A task control failed.",
                      next_action="Review controls before drawing a capacity conclusion.")
    elif all(value == "PASS" for value in gates.values()):
        result.update(status="PASS", diagnosis="Initial capacity gate passed through 16 stored facts.",
                      next_action="Freeze a separate 20–30-fact confirmation before M1-003.")
    else:
        result.update(status="FAIL", diagnosis="The initial recall gate did not pass through all tested sizes.",
                      next_action="Use the paired contrasts to diagnose the lowest failing capacity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(experiment="M1-002 Capacity and Interference", status="INVALID", engineering_status="FAIL")
    try:
        result["environment"] = environment()
        with MiniCells.open_preview(out / "work" / "registry") as mc:
            result["identity"] = mc.model_info
            run(mc, out, result)
        result["engineering_status"] = "PASS"
    except Exception as exc:
        result.update(status="INVALID", diagnosis=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        publishable(out, FORMAT, result, started)


if __name__ == "__main__":
    main()
