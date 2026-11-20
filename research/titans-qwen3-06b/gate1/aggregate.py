"""Aggregate the three paired seed runs for one Gate 1 arm."""
import argparse
import json
from pathlib import Path
from statistics import mean
from protocol import SEEDS, summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs=3,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    rows, web, configs = {}, {}, []
    for run in a.runs:
        decision=json.loads((run/"decision.json").read_text())
        if decision["status"]!="ARM_SEED_COMPLETE":
            raise ValueError("Incomplete seed")
        configs.append(decision)
        seed=decision["seed"]
        if seed in rows:
            raise ValueError("Duplicate seed")
        results=json.loads((run/"validation-500.json").read_text())
        rows[seed],web[seed]=results["synthetic"],results["fineweb"]
    if set(rows)!=set(SEEDS) or any(len({c[k] for c in configs})!=1 for k in
        ("arm","source_commit","model_revision","titans_revision")):
        raise ValueError("Incompatible seed runs")
    cd,ce=(summarize(rows,c) for c in ("donor","empty"))
    nll={k:mean(mean(r[k] for r in web[s]) for s in SEEDS) for k in ("native","memory")}
    regression=nll["memory"]/nll["native"]-1
    numeric=(ce["mean"]>0 and cd["ci95"][0]>0 and sum(x>0 for x in cd["per_seed"].values())>=2 and regression<=.01)
    a.output.write_text(json.dumps(dict(arm=configs[0]["arm"],correct_vs_donor=cd,
        correct_vs_empty=ce,exact={c:summarize(rows,c,"exact") for c in ("empty","donor")},
        fineweb=nll,fineweb_relative_regression=regression,numeric_criteria_met=numeric,
        scientific_status="REVIEW_REQUIRED" if numeric else "PASS_CRITERIA_NOT_MET",
        outstanding_review="Gate-only explanation and retrieval diagnostics"),indent=2,allow_nan=False))

if __name__=="__main__":
    main()
