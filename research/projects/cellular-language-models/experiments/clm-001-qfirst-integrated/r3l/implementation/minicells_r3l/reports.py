"""Separate realizable frozen gains, historical search comparison, learning and target-aware diagnostic."""

from pathlib import Path
import numpy as np
from minicells_r3e.common import read, atomic, digest
from minicells_r3j.reports import interval

FILES = (
    "INPUT_VERIFICATION.json",
    "DATA_BOUNDARY_PROOF.json",
    "ANCHOR_REGISTRY.json",
    "CANDIDATE_REGISTRY.json",
    "TRUST_REGION_RESULTS.json",
    "SEARCH_RESULTS.json",
    "SELECT_RESULTS.json",
    "FROZEN_FINALISTS.json",
    "FROZEN_CELL_OPPORTUNITY.json",
    "LONG_HORIZON_RESULTS.json",
    "CELL_CONNECTION_ATTRIBUTION.json",
    "PATH_ORACLE_DIAGNOSTIC.json",
    "COMPUTE_COST.json",
    "SCIENTIFIC-FINDINGS.md",
    "RUN_MANIFEST.json",
)


def report(c, output, failures, reference):
    from .runner import root_for, identity

    output = Path(output)
    folder = output / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    documents = {k: [] for k in FILES if k.endswith(".json")}
    outcomes = []
    cost = []
    oracles = []
    for seed in c["seeds"]:
        for arm in c["arms"]:
            root = root_for(output, seed, arm)
            common = dict(seed=seed, arm=arm)
            for name, target in [
                ("CANDIDATES.json", "CANDIDATE_REGISTRY.json"),
                ("SEARCH.json", "SEARCH_RESULTS.json"),
                ("SELECT.json", "SELECT_RESULTS.json"),
                ("FINALISTS.json", "FROZEN_FINALISTS.json"),
            ]:
                if (root / name).exists():
                    documents[target].append(dict(common, content=read(root / name)))
            if (root / "CANDIDATES.json").exists():
                rows = read(root / "CANDIDATES.json")
                documents["TRUST_REGION_RESULTS.json"].append(
                    dict(
                        common,
                        candidates=[
                            dict(
                                candidate_id=r["candidate_id"],
                                family=r["family"],
                                status=r["status"],
                                trust=r["trust"],
                            )
                            for r in rows
                        ],
                    )
                )
                search = (
                    read(root / "SEARCH.json")
                    if (root / "SEARCH.json").exists()
                    else {"candidates": {}}
                )
                documents["TRUST_REGION_RESULTS.json"][-1]["search_efficiency"] = dict(
                    nonbaseline=sum(r["family"] != "BASELINE" for r in rows),
                    admitted=sum(r["valid"] and r["family"] != "BASELINE" for r in rows),
                    unique_parameter_states=len({r["parameter_sha256"] for r in rows}),
                    effective_search_paths=sum(
                        r.get("effective_reorganization", False)
                        for r in search["candidates"].values()
                    ),
                    unique_search_paths=len(
                        {r["path_sha256"] for r in search["candidates"].values()}
                    ),
                )
                documents["ANCHOR_REGISTRY.json"].append(
                    dict(common, step=300, receipt=rows[0]["anchor_receipt"])
                )
            if (root / "TRAINED.json").exists():
                cost.append(dict(common, **read(root / "TRAINED.json")))
            if (root / "ORACLE.json").exists():
                oracles.append(dict(common, **read(root / "ORACLE.json")))
            if not (root / "EVALUATION.json").exists():
                continue
            for raw in read(root / "EVALUATION.json"):
                r = {k: v for k, v in raw.items() if k != "stories"}
                r.update(common)
                r["bootstrap"] = interval(raw, c["bootstrap_seed"], c["bootstrap_replicates"])
                r["error_bound"] = 2 * r["numerical_resolution"]
                r["practical_gain"] = r["V"] >= c["practical_gain"]
                outcomes.append(r)
    documents["LONG_HORIZON_RESULTS.json"] = outcomes
    documents["CELL_CONNECTION_ATTRIBUTION.json"] = [
        {k: r[k] for k in ("seed", "arm", "role", "horizon", "V", "R", "C", "I", "error_bound")}
        for r in outcomes
    ]
    frozen = [r for r in outcomes if r["horizon"] == 0]
    comparisons = []
    for architecture in "AB":
        for name, horizon, left, right in [
            ("F2", 0, "SEARCH_BEST", "BASELINE"),
            ("F3", 0, "SEARCH_BEST", "HISTORICAL"),
            ("L1", 100, "SEARCH_BEST", "BASELINE"),
        ]:
            rows = [r for r in outcomes if r["arm"][0] == architecture and r["horizon"] == horizon]
            deltas = []
            byseed = {}
            for seed in c["seeds"]:
                values = []
                for mode in ("H", "T"):
                    a = next(
                        (
                            r
                            for r in rows
                            if r["seed"] == seed
                            and r["arm"] == architecture + "-C" + mode
                            and r["role"] == left
                        ),
                        None,
                    )
                    b = next(
                        (
                            r
                            for r in rows
                            if r["seed"] == seed
                            and r["arm"] == architecture + "-C" + mode
                            and r["role"] == right
                        ),
                        None,
                    )
                    if a and b:
                        values.append(a["V"] - b["V"])
                if len(values) == 2:
                    byseed[str(seed)] = float(np.mean(values))
                    deltas += values
            mean = float(np.mean(deltas)) if deltas else None
            # Exact Seed-level sign flips: three Seeds cannot yield strong confirmation.
            seed_values = list(byseed.values())
            null = (
                [
                    abs(
                        np.mean(
                            [v * (1 if mask & (1 << i) else -1) for i, v in enumerate(seed_values)]
                        )
                    )
                    for mask in range(2 ** len(seed_values))
                ]
                if seed_values
                else []
            )
            p = (
                sum(x >= abs(np.mean(seed_values)) - 1e-15 for x in null) / len(null)
                if null
                else None
            )
            comparisons.append(
                dict(
                    architecture=architecture,
                    comparison=name,
                    horizon=horizon,
                    mean_gain=mean,
                    by_seed=byseed,
                    exploratory_p=p,
                    assessable=len(byseed) == 3,
                    signal=bool(
                        len(byseed) == 3
                        and mean >= 0.005
                        and sum(v > 0 for v in byseed.values()) >= 2
                        and all(
                            np.mean([v for j, v in enumerate(deltas) if j != i]) > 0
                            for i in range(len(deltas))
                        )
                    ),
                )
            )
    for architecture in "AB":
        group = sorted(
            [
                r
                for r in comparisons
                if r["architecture"] == architecture and r["exploratory_p"] is not None
            ],
            key=lambda r: r["exploratory_p"],
        )
        prior = 0.0
        for i, row in enumerate(group):
            prior = max(prior, min(1, (len(group) - i) * row["exploratory_p"]))
            row["holm_adjusted_exploratory_p"] = prior
    documents["FROZEN_CELL_OPPORTUNITY.json"] = dict(
        cases=frozen,
        comparisons=comparisons,
        independence="Three Seeds; candidates/Arms are paired dependent comparisons, not independent replications",
    )
    documents["PATH_ORACLE_DIAGNOSTIC.json"] = oracles or [{"status": "DIAGNOSTIC_NOT_COMPLETED"}]
    documents["COMPUTE_COST.json"] = dict(
        training=cost,
        new_updates=sum(r["new_updates"] for r in cost),
        new_tokens=sum(r["new_tokens"] for r in cost),
        oracle_seconds=sum(r["seconds"] for r in oracles),
        scope="Measured time and executed operator rows; not observed GPU FLOPs",
    )
    if documents["COMPUTE_COST.json"]["new_updates"] > 4800:
        raise RuntimeError("TRAINING_BUDGET_EXCEEDED")
    complete = (
        len(cost) == 12
        and all(
            (root_for(output, s, a) / "EVALUATION.json").exists()
            for s in c["seeds"]
            for a in c["arms"]
        )
        and len(documents["ANCHOR_REGISTRY.json"]) == 12
        and len(documents["FROZEN_FINALISTS.json"]) == 12
        and (output / "DATA_BOUNDARY_PROOF.json").exists()
        and not failures
    )

    def outcome(key, yes, no):
        rows = [r for r in comparisons if r["comparison"] == key]
        if not complete:
            return "INCONCLUSIVE"
        if any(r["signal"] for r in rows):
            return yes
        return "INCONCLUSIVE" if any(not r["assessable"] for r in rows) else no

    summary = dict(
        identity=identity(c),
        status="COMPLETED_EXPLORATORY" if complete else "PARTIAL",
        failures=failures,
        realizable=outcome(
            "F2",
            "REALIZABLE_CONNECTION_OPPORTUNITY_OBSERVED",
            "NO_DETECTABLE_REALIZABLE_CONNECTION_OPPORTUNITY",
        ),
        historical_search=outcome(
            "F3", "SEARCH_OUTPERFORMS_HISTORICAL_PROPOSAL", "NO_SEARCH_ADVANTAGE"
        ),
        long_horizon=outcome("L1", "LONG_HORIZON_REORGANIZATION_GAIN", "NO_LONG_HORIZON_GAIN"),
        oracle="DIAGNOSTIC_NOT_COMPLETED"
        if len(oracles) != 4
        or any(len(x.get("records", [])) != c["oracle_tokens"] for x in oracles)
        else "TARGET_AWARE_PATH_OPPORTUNITY_OBSERVED"
        if any(
            any(r["baseline_nll"] - r["beam_nll"] >= 0.005 for r in x["records"]) for x in oracles
        )
        else "NO_DETECTABLE_BEAM_PATH_OPPORTUNITY",
        confirmation="SEALED",
        hf_reference=reference,
        scientific_confirmation=False,
    )
    documents["INPUT_VERIFICATION.json"] = summary
    documents["DATA_BOUNDARY_PROOF.json"] = (
        read(output / "DATA_BOUNDARY_PROOF.json")
        if (output / "DATA_BOUNDARY_PROOF.json").exists()
        else {"status": "NOT_VERIFIED"}
    )
    for name, row in documents.items():
        if name != "RUN_MANIFEST.json":
            atomic(folder / name, row)
    (folder / "SCIENTIFIC-FINDINGS.md").write_text(
        "# R3l — Reorganization Potential\n\n"
        + summary["status"]
        + "\n\n"
        + "\n".join(
            k + ": " + str(summary[k])
            for k in ("realizable", "historical_search", "long_horizon", "oracle")
        )
        + "\n\nFinite parameter search is not a global optimum. The target-aware Beam diagnostic is non-deployable and not a global quality bound. Only three independent Seeds; Story intervals do not replace Seed replication. No automatic next experiment.\n"
    )
    atomic(
        folder / "RUN_MANIFEST.json",
        dict(
            summary,
            artifacts=[
                dict(path=x.name, bytes=x.stat().st_size, sha256=digest(x))
                for x in sorted(folder.iterdir())
                if x.is_file() and x.name != "RUN_MANIFEST.json"
            ],
        ),
    )
    return summary
