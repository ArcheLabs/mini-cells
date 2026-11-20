"""QTR-1 document-cluster inference from complete Step 500 holdout rows."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean
from profiles import ARMS, SEEDS


def decide(records):
    runs = {(r["arm"], r["seed"]): r for r in records}
    if len(records) != 12 or set(runs) != {(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("All twelve arm/seed results are required")
    if any(r["step"] != 500 or r["engineering_validity"] != "PASS" or
           r["backbone_freeze"] != "PASS" for r in records):
        return dict(decision="INVALID", winning_arm="NONE", root_cause="ENGINEERING_INVALID")
    reference = [r["document_sha256"] for r in records[0]["documents"]]
    if len(reference) != 64 or len(set(reference)) != 64 or any(
        [d["document_sha256"] for d in r["documents"]] != reference for r in records):
        raise ValueError("Unpaired holdout documents")
    donors = [d["donor_indices"] for d in records[0]["documents"]]
    if any(len(d) != 3 or len(set(d)) != 3 or i in d or any(not 0 <= j < 64 for j in d)
           for i,d in enumerate(donors)) or any(
        [d["donor_indices"] for d in r["documents"]] != donors for r in records):
        raise ValueError("Unpaired or invalid donor assignments")
    rng = random.Random(20261119)
    draws = [rng.choices(range(64), k=64) for _ in range(10000)]

    def stats(values):
        samples = sorted(mean(values[i] for i in draw) for draw in draws)
        return dict(mean=mean(values), ci95=[samples[250],samples[9749]],
                    positive_documents=mean(x>0 for x in values))

    results, specificity = {}, {}
    for arm in ARMS:
        values, seed_values, nll = {}, {}, {}
        rows = [runs[arm,s]["documents"] for s in SEEDS]
        for condition in ("correct","donor","empty","zero_read","shuffled"):
            per_seed = [[mean(d["donor_nll"]) if condition=="donor" else d[condition+"_nll"]
                         for d in rr] for rr in rows]
            nll[condition] = mean(map(mean,per_seed))
            if condition != "correct":
                diffs = [[x-d["correct_nll"] for x,d in zip(v,rr)] for v,rr in zip(per_seed,rows)]
                seed_values[condition] = dict(zip(SEEDS,map(mean,diffs)))
                values[condition] = list(map(mean,zip(*diffs)))
        specificity[arm] = values["donor"]
        comparisons = {k:stats(v) for k,v in values.items()}
        cd = comparisons["donor"]
        passed = (cd["mean"] >= .01 and cd["ci95"][0]>0 and cd["positive_documents"]>=.6
                  and sum(x>0 for x in seed_values["donor"].values())>=2
                  and all(comparisons[k]["ci95"][0]>0 for k in ("empty","zero_read")))
        results[arm] = dict(pass_criteria=passed,nll=nll,comparisons=comparisons,per_seed=seed_values)

    effects = dict(inner_lr=stats([(b-a+d-c)/2 for a,b,c,d in zip(*specificity.values())]),
                   optimizer=stats([(c-a+d-b)/2 for a,b,c,d in zip(*specificity.values())]),
                   interaction=stats([(d-c)-(b-a) for a,b,c,d in zip(*specificity.values())]))
    winners = [arm for arm,r in results.items() if r["pass_criteria"]]
    best = max(winners,key=lambda a:results[a]["comparisons"]["donor"]["mean"]) if winners else "NONE"
    positive = [label for key,label in (("inner_lr","INNER_UPDATE"),("optimizer","OUTER_OPTIMIZATION"),
                ("interaction","INTERACTION")) if effects[key]["ci95"][0]>0]
    return dict(decision="GO" if winners else "NO_GO",winning_arm=best,
                primary_metric="content_specificity",
                root_cause=positive[0] if winners and len(positive)==1 else "NOT_DEMONSTRATED",
                strong_negative=not winners and all(
                    r["comparisons"]["donor"]["ci95"][1]<.01 for r in results.values()),
                effects=effects, arms=results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(decide(json.loads(a.input.read_text())),indent=2,allow_nan=False))
