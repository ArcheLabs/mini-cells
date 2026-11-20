"""Paired validation/holdout CPU screening and independent-state transfer."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
import json
import hashlib
import platform
import subprocess
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ham-001"))
import benchmark as legacy
from memory import Config, DeltaMemory, Parallel

COEFFICIENTS = (0, .01, .05, .1, .25, .5, 1)
VALIDATION = (39, 40, 41)
HOLDOUT = (42, 43, 44, 45, 46)


def transfer(m, k, v):
    end = len(k)//3
    for key, value in zip(k[:end], v[:end]):
        m.write(key, value)
    state = m.export_state()
    expected = m.read(k[:end])
    b, fork = m.fresh(), m.fresh()
    b.import_state(state)
    fork.import_state(state)
    np.testing.assert_allclose(b.read(k[0]), expected[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(b.read(k[:end]), expected)
    for key, value in zip(k[end:2*end], v[end:2*end]):
        b.write(key, value)
    c = m.fresh()
    c.import_state(b.export_state())
    np.testing.assert_array_equal(c.read(k[:end]), b.read(k[:end]))
    for key, value in zip(k[2*end:], v[2*end:]):
        c.write(key, value)
    np.testing.assert_array_equal(fork.read(k[:end]), expected)
    np.testing.assert_array_equal(m.read(k[:end]), expected)
    return [legacy.score(x, v[:end]) for x in (expected, b.read(k[:end]), c.read(k[:end]))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--titans-upstream", type=Path)
    a = p.parse_args()
    config, geometry = Config(), legacy.Config()
    upstream = None
    if a.titans_upstream:
        from titans_adapter import Titans, load_upstream, REVISION
        import torch
        torch.set_num_threads(2)
        upstream = load_upstream(a.titans_upstream)
        pilot = Parallel(DeltaMemory(config), Titans(geometry, upstream, 42), 0)
        k, v, _, _ = legacy.dataset(42, 8, "random", geometry)
        np.testing.assert_array_equal(pilot.step(k[0], k[0], v[0]), 0)
        np.testing.assert_array_equal(pilot.read(k), pilot.delta.read(k))
        transfer(pilot.fresh(), k, v)

    rows, transfers = [], []
    started = time.time()
    for split, seeds in (("validation", VALIDATION), ("holdout", HOLDOUT)):
        for seed in seeds:
            for task, count in ([("random", n) for n in (8,16,32,64,128,256,512)] +
                                [(t,16) for t in ("correlated","hard_negative","overwrite","interference","nonlinear")]):
                data = legacy.dataset(seed,count,task,geometry)
                k,v,q,target = data
                digest = hashlib.sha256(b"".join(x.tobytes() for x in data)).hexdigest()
                arms = {"D0":legacy.FullKV(geometry), "D1":DeltaMemory(config),
                        "D2":DeltaMemory(replace(config, alpha=.99)), "D4":legacy.Attention(geometry),
                        "beta_0.5":DeltaMemory(replace(config, beta=.5))}
                if upstream is not None:
                    arms["D3"] = Titans(geometry, upstream, seed)
                    arms.update({f"F:{c:g}":Parallel(DeltaMemory(config), Titans(geometry,upstream,seed),c)
                                 for c in COEFFICIENTS})
                for arm,m in arms.items():
                    result = legacy.evaluate(m,data,task)
                    if isinstance(m, Parallel):
                        d,t = m.branches(q)
                        result["branch_norms"] = dict(delta=float(np.linalg.norm(d)), titans=float(np.linalg.norm(t)),
                            scaled_titans=float(m.coefficient*np.linalg.norm(t)), fused=float(np.linalg.norm(m.read(q))),
                            cosine=float((legacy.unit(d)*legacy.unit(t)).sum(-1).mean()))
                    rows.append(dict(split=split, seed=seed, task=task,count=count,arm=arm,
                                     dataset_hash=digest, **result))
                    if split == "holdout" and task == "random" and count == 32:
                        transfers.append(dict(seed=seed,arm=arm,retention=transfer(m.fresh(),k,v)))
            print(split, seed, flush=True)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["split"],r["task"],r["count"],r["arm"]].append(r["metrics"]["top1"])
    summary = [dict(split=s,task=t,count=n,arm=a,mean=mean(v),sd=stdev(v),
                    ci95=[mean(v)-({3:4.303,5:2.776}[len(v)]*stdev(v)/len(v)**.5),
                          mean(v)+({3:4.303,5:2.776}[len(v)]*stdev(v)/len(v)**.5)])
               for (s,t,n,a),v in buckets.items()]
    # Freeze selection using validation random recall only; every sweep point is a diagnostic.
    selected = None if upstream is None else max(
        COEFFICIENTS, key=lambda c: mean(r["mean"] for r in summary if
                 r["split"]=="validation" and r["task"]=="random" and r["arm"]==f"F:{c:g}"))
    a.output.mkdir(parents=True)
    result = dict(rows=rows, transfer=transfers, summary=summary, selected_coefficient=selected,
                  unavailable={"Mamba-2":"No verified adapter", **({"Titans":"Upstream not supplied"} if upstream is None else {})},
                  config=vars(config), seeds=dict(validation=VALIDATION, holdout=HOLDOUT),
                  device="cpu", platform=platform.platform(), python=sys.version, numpy=np.__version__,
                  torch=None if upstream is None else torch.__version__,
                  titans_revision=None if upstream is None else REVISION,
                  elapsed_seconds=time.time()-started,
                  source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  evaluator_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/"results.json").write_text(json.dumps(result,allow_nan=False))
    lines = ["# HAM-002 — CPU screening", "", "| KV count | Delta | Gated Delta |",
             "|---:|---:|---:|"]
    for count in (8,16,32,64,128,256,512):
        scores=[next(r["mean"] for r in summary if r["split"]=="holdout" and r["task"]=="random"
                     and r["count"]==count and r["arm"]==arm) for arm in ("D1","D2")]
        lines.append(f"| {count} | {scores[0]:.2%} | {scores[1]:.2%} |")
    decision = ("Run pinned Titans parity and parallel CPU screening before any complementarity claim." if selected is None
                else f"Review held-out task and transfer results for validation-selected lambda={selected:g} before considering residual or model integration.")
    lines += ["", decision, "", "Resource data are descriptive, not matched budgets."]
    (a.output/"HAM_002_FINAL_REPORT.md").write_text("\n".join(lines)+"\n")
    (a.output/"decision.json").write_text(json.dumps(dict(format="minicells.ham-002.v1",
         status="STANDALONE_SCREENING" if selected is None else "PARALLEL_SCREENING",
         next_step=decision, residual_started=False, integration_started=False)))


if __name__ == "__main__":
    main()
