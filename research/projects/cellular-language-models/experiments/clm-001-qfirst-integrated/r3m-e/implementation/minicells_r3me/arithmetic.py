"""Frozen two-digit addition facts with fact-level SHA-256 Train/Dev splitting."""

from collections import Counter
import hashlib
from pathlib import Path

from minicells_r3e.common import canonical

TEMPLATES = (
    "What is {a} plus {b}? Answer: ",
    "Add {a} and {b}. Answer: ",
    "Calculate {a} + {b}. Answer: ",
    "Find the sum of {a} and {b}. Answer: ",
)
DOMAIN = "MINICELLS-R3ME-ARITH-v1"


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fact_group(a, b):
    return "%d+%d" % (min(int(a), int(b)), max(int(a), int(b)))


def role_for(a, b, seed):
    group = fact_group(a, b)
    bucket = int(_digest("%s:%d:%s" % (DOMAIN, seed, group))[:8], 16) % 5
    return "dev" if bucket == 0 else "train"


def _offsets(encoded, prompt_chars):
    indices = [i for i, (lo, hi) in enumerate(encoded.offsets) if hi > prompt_chars]
    if not indices or indices[0] == 0:
        raise ValueError("ANSWER_TOKEN_MASK_INVALID")
    return indices


def generate(tokenizer, seed=153301, max_length=64, eos_token_id=1):
    if int(eos_token_id) <= 0:
        raise ValueError("ARITHMETIC_EOS_TOKEN_INVALID")
    rows = {"train": [], "dev": []}
    facts = {"train": set(), "dev": set()}
    token_lengths = {"train": Counter(), "dev": Counter()}
    carry_counts = {"train": Counter(), "dev": Counter()}
    for a in range(10, 100):
        for b in range(a, 100):
            role = role_for(a, b, seed)
            group = fact_group(a, b)
            if group in facts["train"] | facts["dev"]:
                raise ValueError("DUPLICATE_CANONICAL_FACT")
            facts[role].add(group)
            template_hash = _digest("%s:template:%d:%s" % (DOMAIN, seed, group))
            template_index = int(template_hash[:8], 16) % len(TEMPLATES)
            prompt = TEMPLATES[template_index].format(a=a, b=b)
            answer = str(a + b)
            text = prompt + answer
            encoded = tokenizer.encode(text)
            answer_positions = _offsets(encoded, len(prompt))
            ids = list(encoded.ids)
            if len(ids) > max_length:
                raise ValueError("ARITHMETIC_SEQUENCE_TOO_LONG")
            row = {
                "a": a,
                "b": b,
                "answer": a + b,
                "carry": (a % 10 + b % 10) >= 10,
                "group": group,
                "template_id": template_index,
                "text": text,
                "ids": ids,
                "prompt_ids": list(encoded.ids[: answer_positions[0]]),
                "answer_positions": answer_positions,
                "answer_tokens": [int(encoded.ids[i]) for i in answer_positions],
                "sample_sha256": _digest(text),
            }
            rows[role].append(row)
            token_lengths[role][len(ids)] += 1
            carry_counts[role]["carry" if row["carry"] else "no_carry"] += 1
    if facts["train"] & facts["dev"]:
        raise ValueError("ARITHMETIC_FACT_LEAKAGE")
    if set(carry_counts["train"]) != {"carry", "no_carry"} or set(carry_counts["dev"]) != {
        "carry",
        "no_carry",
    }:
        raise ValueError("ARITHMETIC_CARRY_STRATA_MISSING")
    for role in rows:
        rows[role].sort(key=lambda r: (r["a"], r["b"]))
    evidence = {
        "schema": "minicells.r3me.arithmetic-data-audit.v1",
        "generator": DOMAIN,
        "generator_seed": int(seed),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "eos_token_id": int(eos_token_id),
        "eos_delimiter_added_between_training_facts": True,
        "tokenizer_sha256": None,
        "fact_split": "SHA256_DOMAIN_SEED_CANONICAL_UNORDERED_PAIR_MOD5_ZERO_IS_DEV",
        "templates": list(TEMPLATES),
        "template_variants_are_not_independent_facts": True,
        "fact_counts": {k: len(v) for k, v in facts.items()},
        "sample_counts": {k: len(v) for k, v in rows.items()},
        "fact_sets_sha256": {k: canonical(sorted(v)) for k, v in facts.items()},
        "sample_hashes_sha256": {k: canonical([r["sample_sha256"] for r in rows[k]]) for k in rows},
        "carry_strata": {k: dict(v) for k, v in carry_counts.items()},
        "token_length_histograms": {k: dict(v) for k, v in token_lengths.items()},
        "operand_range_inclusive": [10, 99],
        "answer_range_inclusive": [20, 198],
        "train_dev_fact_overlap": 0,
        "confirmation": "SEALED",
    }
    return rows, evidence

