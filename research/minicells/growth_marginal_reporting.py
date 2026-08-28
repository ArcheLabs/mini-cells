"""Formal aggregation and decisions for CLM-0.3b marginal growth utility."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .growth_reporting import write_ppl_history


FORMAL_ARMS = ("fixed4", "marginal_growth", "random_growth")
FORMAL_REPLICATES = (0, 1, 2)
FORMAL_MIN_SATURATION_TOKENS = 1_500_000
FORMAL_MAX_PREBIRTH_TOKENS = 3_000_000
FORMAL_POST_BIRTH_TOKENS = 1_000_000
FORMAL_UTILITY_THRESHOLD = 0.995


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_ppl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["replicate"] = int(raw["replicate"])
            row["tokens"] = int(raw["tokens"])
            for key in (
                "ppl",
                "nll",
                "fixed4_ppl",
                "clm01_start_ppl",
                "textnca_frozen_ppl",
                "ppl_vs_fixed4",
                "ppl_vs_clm01",
                "ppl_vs_textnca",
            ):
                value = raw.get(key, "")
                row[key] = None if value in (None, "", "None", "nan", "NaN") else float(value)
            rows.append(row)
    return rows


def _latest_age_diagnostic(rows: list[dict[str, Any]], age: int) -> dict[str, Any] | None:
    candidates = [row for row in rows if int(row.get("offset_tokens", -1)) == age]
    return candidates[-1] if candidates else None


def _paired_prebirth_ok(
    replicate: int,
    worker_rows: dict[tuple[int, str], list[dict[str, Any]]],
    *,
    cutoff: int,
    rtol: float = 1e-8,
) -> bool:
    maps = {
        arm: {
            int(row["tokens"]): float(row["ppl"])
            for row in worker_rows[(replicate, arm)]
            if int(row["tokens"]) <= cutoff
        }
        for arm in FORMAL_ARMS
    }
    token_sets = [set(rows) for rows in maps.values()]
    if not token_sets or any(tokens != token_sets[0] for tokens in token_sets[1:]):
        return False
    for token in sorted(token_sets[0]):
        reference = maps["fixed4"][token]
        for arm in ("marginal_growth", "random_growth"):
            observed = maps[arm][token]
            if not math.isclose(reference, observed, rel_tol=rtol, abs_tol=1e-10):
                return False
    return True


def marginal_growth_decision(
    summaries: list[dict[str, Any]],
    *,
    saturation_replicates: int,
    paired_prebirth_replicates: int,
    equivalent_growth_births: int,
    viable_marginal_births: int,
    causal_positive_ci_replicates: int,
    formal_gpu_experiment_run: bool,
    training_code_commit: str | None = None,
    training_code_tree_sha: str | None = None,
) -> dict[str, Any]:
    by_arm = {
        arm: sorted((row for row in summaries if row["arm"] == arm), key=lambda row: row["replicate"])
        for arm in FORMAL_ARMS
    }
    marginal = by_arm["marginal_growth"]
    fixed = by_arm["fixed4"]
    random_rows = by_arm["random_growth"]
    utility_passes = sum(
        growth["ppl"] / control["ppl"] <= FORMAL_UTILITY_THRESHOLD
        for growth, control in zip(marginal, fixed)
        if growth["replicate"] == control["replicate"]
    )
    selector_wins = sum(
        growth["ppl"] < control["ppl"]
        for growth, control in zip(marginal, random_rows)
        if growth["replicate"] == control["replicate"]
    )
    paired_ok = paired_prebirth_replicates == len(FORMAL_REPLICATES)
    saturation_ok = saturation_replicates == len(FORMAL_REPLICATES)
    equivalence_ok = equivalent_growth_births == 2 * len(FORMAL_REPLICATES)
    viability_ok = viable_marginal_births == len(FORMAL_REPLICATES)
    utility_ok = paired_ok and saturation_ok and viability_ok and utility_passes >= 2
    selection_ok = paired_ok and saturation_ok and viability_ok and selector_wins >= 2
    causal_ok = saturation_ok and causal_positive_ci_replicates >= 2
    return {
        "format": "minicells.clm-0.3b-marginal-growth-utility.decision.v1",
        "formal_gpu_experiment_run": bool(formal_gpu_experiment_run),
        "training_code_commit": training_code_commit,
        "training_code_tree_sha": training_code_tree_sha,
        "paired_prebirth": {
            "status": "CLM_PAIRED_PREBIRTH_EQUIVALENCE" if paired_ok else "CLM_PAIRED_PREBIRTH_MISMATCH",
            "replicates_passed": paired_prebirth_replicates,
        },
        "saturation_regime": {
            "status": "CLM_SATURATION_REGIME_ESTABLISHED" if saturation_ok else "NO_SATURATION_REGIME",
            "replicates_passed": saturation_replicates,
        },
        "growth_equivalence": {
            "status": "CLM_GROWTH_EQUIVALENCE" if equivalence_ok else "CLM_GROWTH_EQUIVALENCE_FAILURE",
            "births_checked": equivalent_growth_births,
        },
        "growth_viability": {
            "status": "CLM_MARGINAL_GROWTH_VIABILITY" if viability_ok else "NO_MARGINAL_GROWTH_VIABILITY",
            "replicates_passed": viable_marginal_births,
        },
        "marginal_growth_utility": {
            "status": "CLM_MARGINAL_GROWTH_UTILITY_SIGNAL" if utility_ok else "NO_MARGINAL_GROWTH_UTILITY_SIGNAL",
            "threshold": FORMAL_UTILITY_THRESHOLD,
            "replicates_passed": utility_passes,
        },
        "marginal_selection": {
            "status": "CLM_MARGINAL_SELECTION_SIGNAL" if selection_ok else "NO_MARGINAL_SELECTION_SIGNAL",
            "replicates_passed": selector_wins,
        },
        "causal_utility": {
            "status": "CLM_NEWBORN_CAUSAL_UTILITY_SIGNAL" if causal_ok else "NO_NEWBORN_CAUSAL_UTILITY_SIGNAL",
            "replicates_with_positive_ci95": causal_positive_ci_replicates,
        },
    }


def aggregate_marginal_results(
    output_root: str | Path,
    *,
    formal_gpu_experiment_run: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    worker_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    saturation: dict[tuple[int, str], dict[str, Any]] = {}
    worker_complete: dict[tuple[int, str], dict[str, Any]] = {}
    growth_evidence: dict[tuple[int, str], dict[str, Any]] = {}
    code_commits: set[str] = set()
    code_trees: set[str] = set()

    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            directory = root / f"r{replicate}-{arm}"
            required = [directory / "events.jsonl", directory / "ppl-history.csv", directory / "saturation.json"]
            if arm != "fixed4":
                required.extend([directory / "growth-history.json", directory / "newborn-diagnostics.json"])
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError(f"CLM-0.3b worker artifacts missing for r{replicate}/{arm}: {missing}")
            events = _read_events(directory / "events.jsonl")
            completes = [
                event for event in events
                if event.get("type") == "worker_complete" and event.get("mode") != "preflight_only"
            ]
            if not completes:
                raise RuntimeError(f"r{replicate}/{arm} has no completed worker event")
            complete = completes[-1]
            if int(complete.get("consumed_tokens", -1)) < int(complete.get("target_tokens", 0)):
                raise RuntimeError(f"r{replicate}/{arm} stopped before its declared target")
            worker_complete[(replicate, arm)] = complete
            if complete.get("code_commit"):
                code_commits.add(str(complete["code_commit"]))
            if complete.get("code_tree_sha"):
                code_trees.add(str(complete["code_tree_sha"]))
            worker_rows[(replicate, arm)] = _read_ppl(directory / "ppl-history.csv")
            saturation[(replicate, arm)] = _read_json(directory / "saturation.json")
            if arm != "fixed4":
                history = _read_json(directory / "growth-history.json")
                diagnostics = _read_json(directory / "newborn-diagnostics.json")
                final_diag = _latest_age_diagnostic(diagnostics, FORMAL_POST_BIRTH_TOKENS)
                growth_evidence[(replicate, arm)] = {
                    "history": history,
                    "diagnostics": diagnostics,
                    "final_diagnostic": final_diag,
                }

    if formal_gpu_experiment_run and (len(code_commits) != 1 or len(code_trees) != 1):
        raise RuntimeError(
            f"formal CLM-0.3b matrix mixed code provenance: commits={sorted(code_commits)}, trees={sorted(code_trees)}"
        )
    training_commit = next(iter(code_commits)) if len(code_commits) == 1 else None
    training_tree = next(iter(code_trees)) if len(code_trees) == 1 else None

    saturation_replicates = 0
    saturation_tokens: dict[int, int] = {}
    paired_prebirth_replicates = 0
    for replicate in FORMAL_REPLICATES:
        entries = [saturation[(replicate, arm)] for arm in FORMAL_ARMS]
        detected = all(bool(item.get("detected", False)) for item in entries)
        tokens = {int(item.get("token", -1)) for item in entries if item.get("token") is not None}
        if detected and len(tokens) == 1:
            saturation_replicates += 1
            saturation_tokens[replicate] = next(iter(tokens))
        cutoff = saturation_tokens.get(
            replicate,
            min(max(int(row["tokens"]) for row in worker_rows[(replicate, arm)]) for arm in FORMAL_ARMS),
        )
        if _paired_prebirth_ok(replicate, worker_rows, cutoff=cutoff):
            paired_prebirth_replicates += 1
        elif formal_gpu_experiment_run:
            raise RuntimeError(f"paired pre-birth trajectory mismatch in replicate {replicate}")

    fixed_by_key: dict[tuple[int, int], float] = {}
    for replicate in FORMAL_REPLICATES:
        for row in worker_rows[(replicate, "fixed4")]:
            fixed_by_key[(replicate, int(row["tokens"]))] = float(row["ppl"])

    formal_rows: list[dict[str, Any]] = []
    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            for source in worker_rows[(replicate, arm)]:
                key = (replicate, int(source["tokens"]))
                fixed = fixed_by_key.get(key)
                if fixed is None:
                    continue
                row = dict(source)
                ppl = float(row["ppl"])
                row["fixed4_ppl"] = fixed
                row["ppl_vs_fixed4"] = ppl / fixed
                if row.get("clm01_start_ppl") is not None:
                    row["ppl_vs_clm01"] = ppl / float(row["clm01_start_ppl"])
                if row.get("textnca_frozen_ppl") is not None:
                    row["ppl_vs_textnca"] = ppl / float(row["textnca_frozen_ppl"])
                formal_rows.append(row)
    formal_rows.sort(key=lambda row: (int(row["replicate"]), str(row["arm"]), int(row["tokens"])))
    write_ppl_history(root / "formal-ppl-history.csv", formal_rows)

    summaries: list[dict[str, Any]] = []
    equivalent_births = 0
    viable_marginal = 0
    causal_positive_ci = 0
    for replicate in FORMAL_REPLICATES:
        expected_final = saturation_tokens.get(replicate, -FORMAL_POST_BIRTH_TOKENS) + FORMAL_POST_BIRTH_TOKENS
        for arm in FORMAL_ARMS:
            rows = [row for row in formal_rows if row["replicate"] == replicate and row["arm"] == arm]
            final_candidates = [row for row in rows if int(row["tokens"]) == expected_final]
            final = final_candidates[-1] if final_candidates else (max(rows, key=lambda row: int(row["tokens"])) if rows else None)
            if final is None:
                raise RuntimeError(f"r{replicate}/{arm} has no formal PPL row")
            summary: dict[str, Any] = {
                "replicate": replicate,
                "arm": arm,
                "tokens": int(final["tokens"]),
                "ppl": float(final["ppl"]),
                "fixed4_ppl": float(final["fixed4_ppl"]),
                "ppl_vs_fixed4": float(final["ppl_vs_fixed4"]),
                "saturation_token": saturation_tokens.get(replicate),
            }
            if arm != "fixed4":
                evidence = growth_evidence[(replicate, arm)]
                history = evidence["history"]
                final_diag = evidence["final_diagnostic"]
                equivalent = len(history) == 1 and history[0].get("parity", {}).get("status") == "CLM_GROWTH_EQUIVALENCE"
                equivalent_births += int(equivalent)
                viable = bool(
                    final_diag is not None
                    and float(final_diag.get("child_usage", 0.0)) > 0.0
                    and float(final_diag.get("relative_l2", 0.0)) > 0.0
                    and float(final_diag.get("split_entropy", 0.0)) > 0.0
                )
                if arm == "marginal_growth":
                    viable_marginal += int(viable)
                    if final_diag is not None and float(final_diag.get("causal_merge_back_ci95_low", float("-inf"))) > 0.0:
                        causal_positive_ci += 1
                summary.update({
                    "equivalent_birth": equivalent,
                    "viable": viable,
                    "birth_evidence": history,
                    "final_diagnostic": final_diag,
                })
            summaries.append(summary)

    decision = marginal_growth_decision(
        summaries,
        saturation_replicates=saturation_replicates,
        paired_prebirth_replicates=paired_prebirth_replicates,
        equivalent_growth_births=equivalent_births,
        viable_marginal_births=viable_marginal,
        causal_positive_ci_replicates=causal_positive_ci,
        formal_gpu_experiment_run=formal_gpu_experiment_run,
        training_code_commit=training_commit,
        training_code_tree_sha=training_tree,
    )
    (root / "replicate-summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "formal_rows": formal_rows,
        "summaries": summaries,
        "decision": decision,
        "saturation_tokens": saturation_tokens,
        "training_code_commit": training_commit,
        "training_code_tree_sha": training_tree,
    }
