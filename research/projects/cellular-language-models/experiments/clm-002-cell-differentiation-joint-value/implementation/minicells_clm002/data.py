"""Metadata-only task pairing and explicit effective-target budgets; no task IDs in model inputs."""

import copy
import re
import unicodedata
import numpy as np
import torch
from .common import sha


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def allocation_audit(rows, history_coverage_complete=False, window_size=4):
    """Small-record auditor. UNKNOWN historical coverage cannot become global PASS."""
    errors = []
    seen = {}
    grouped = {}
    windows = {}
    shingles = []
    for row in rows:
        role = row["role"]
        if role not in ("train", "dev", "confirm"):
            raise ValueError("CLM002_DATA_ROLE_INVALID")
        text = normalize(row["text"])
        key = sha(text)
        if key in seen and seen[key] != role:
            errors.append("CROSS_ROLE_NORMALIZED_TEXT")
        seen[key] = role
        words = re.findall(r"\w+", text)
        grams = {tuple(words[i : i + 5]) for i in range(max(1, len(words) - 4))}
        for old_role, old_grams in shingles:
            if (
                old_role != role
                and grams
                and old_grams
                and len(grams & old_grams) / len(grams | old_grams) >= 0.8
            ):
                errors.append("CROSS_ROLE_NEAR_DUPLICATE")
        shingles.append((role, grams))
        if row.get("task") == "Arithmetic":
            required = ("template_family", "operator_combination", "numeric_range")
            if any(k not in row for k in required):
                errors.append("ARITHMETIC_GROUP_IDENTITY_UNKNOWN")
                continue
            for field in required[:2]:
                group = (field, str(row[field]))
                if group in grouped and grouped[group] != role:
                    errors.append("ARITHMETIC_" + field.upper() + "_OVERLAP")
                grouped[group] = role
            low, high = row["numeric_range"]
            for old in rows:
                if (
                    old is row
                    or old.get("task") != "Arithmetic"
                    or old["role"] == role
                    or "numeric_range" not in old
                ):
                    continue
                a, b = old["numeric_range"]
                if max(a, low) < min(b, high):
                    errors.append("ARITHMETIC_NUMERIC_RANGE_OVERLAP")
        if "tokens" in row and "tokenizer_sha256" in row:
            for i in range(max(0, len(row["tokens"]) - window_size + 1)):
                window = (row["tokenizer_sha256"], tuple(row["tokens"][i : i + window_size]))
                if window in windows and windows[window] != role:
                    errors.append("EXACT_TOKEN_WINDOW_OVERLAP")
                windows[window] = role
        else:
            errors.append("TOKEN_WINDOW_IDENTITY_UNKNOWN")
    return dict(
        status="CONFLICT"
        if errors
        else "PASS_LOCAL_ROWS_ONLY"
        if history_coverage_complete
        else "UNKNOWN_HISTORICAL_COVERAGE",
        errors=sorted(set(errors)),
        rows=len(rows),
        raw_or_normalized_text_uploaded=False,
        content_identity_sha256=sha(
            [dict(role=r["role"], text_sha256=sha(normalize(r["text"]))) for r in rows]
        ),
        full_historical_coverage=history_coverage_complete,
        global_scientific_isolation_pass=False,
        limitations=[
            "Window check requires explicit same-Tokenizer IDs; cross-story and cross-Tokenizer historical coverage need original streams.",
            "Near-duplicate threshold is an engineering candidate, not a registered scientific leakage threshold.",
        ],
    )


class PairedStream:
    def __init__(self, pools, seed, condition, target_tokens):
        if condition not in ("H", "S") or target_tokens % 2:
            raise ValueError("CLM002_PAIRING_CONFIG_INVALID")
        self.pools = pools
        self.condition = condition
        self.target_tokens = target_tokens
        self.step = 0
        self.order = np.random.default_rng(seed)
        self.streams = [np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)]
        self.pool_sha = sha(
            {
                name: [
                    dict(x=r["x"].tolist(), y=r["y"].tolist(), valid=r["valid"].tolist())
                    for r in rows
                ]
                for name, rows in pools.items()
            }
        )

    def next(self):
        selected = []
        budgets = [0, 0]
        attempts = 0
        while min(budgets) < self.target_tokens // 2:
            side = 0 if budgets[0] < self.target_tokens // 2 else 1
            task = "Story" if self.condition == "S" or side == 0 else "Arithmetic"
            rows = self.pools[task]
            row = copy.deepcopy(rows[int(self.streams[side].integers(len(rows)))])
            valid = int(row["valid"].sum())
            if not valid or budgets[side] + valid > self.target_tokens // 2:
                raise ValueError("CLM002_BATCH_SHAPE_CANNOT_MATCH_EFFECTIVE_TOKEN_BUDGET")
            row.update(task=task, stream=side)
            selected.append(row)
            budgets[side] += valid
            attempts += 1
            if attempts > self.target_tokens:
                raise ValueError("CLM002_SAMPLER_NONPROGRESS")
        order = self.order.permutation(len(selected))
        self.step += 1
        return [selected[int(i)] for i in order]

    def state(self):
        return copy.deepcopy(
            dict(
                condition=self.condition,
                target_tokens=self.target_tokens,
                pool_sha=self.pool_sha,
                step=self.step,
                order=self.order.bit_generator.state,
                streams=[r.bit_generator.state for r in self.streams],
            )
        )

    def restore(self, state):
        if any(state[k] != self.state()[k] for k in ("condition", "target_tokens", "pool_sha")):
            raise RuntimeError("CLM002_PAIRED_STREAM_IDENTITY_CHANGED")
        self.step = state["step"]
        self.order.bit_generator.state = copy.deepcopy(state["order"])
        for rng, value in zip(self.streams, state["streams"]):
            rng.bit_generator.state = copy.deepcopy(value)


def unit_pools(c):
    u = c["engineering"]
    rng = torch.Generator().manual_seed(u["seed"])
    result = {}
    for task in ("Story", "Arithmetic"):
        rows = []
        for _ in range(3):
            tokens = torch.randint(0, u["vocab_size"], (1, u["sequence_length"] + 1), generator=rng)
            rows.append(
                dict(x=tokens[:, :-1], y=tokens[:, 1:], valid=torch.ones_like(tokens[:, 1:]))
            )
        result[task] = rows
    return result
