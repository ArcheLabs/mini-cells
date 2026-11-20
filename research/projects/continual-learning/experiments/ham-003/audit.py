"""Recompute the HAM-002 headline from published per-seed rows."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ORIGIN = "2fa67ec4e6d10b08c9c5ac81ab20edc6391df4d3"

def audit(path):
    raw = path.read_bytes()
    d = json.loads(raw)
    rows = d["rows"]
    assert len(rows) == len({(r["split"],r["seed"],r["task"],r["count"],r["arm"]) for r in rows}) == 1248
    def recall(split, arm):
        v = [r["metrics"]["top1"] for r in rows if r["split"] == split and r["arm"] == arm and r["task"] == "random"]
        assert len(v) == 7 * (3 if split == "validation" else 5)
        return mean(v)
    coeffs = (0,.01,.05,.1,.25,.5,1)
    selected = max(coeffs, key=lambda c: recall("validation",f"F:{c:g}"))
    assert selected == d["selected_coefficient"]
    return dict(artifact_commit=ORIGIN,input_sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
                selected_coefficient=selected, holdout={a:recall("holdout",a) for a in
                    ("D1","D2","D3","beta_0.5",*(f"F:{c:g}" for c in coeffs))},
                transfer={a:[mean(r["retention"][i]["top1"] for r in d["transfer"] if r["arm"]==a)
                             for i in range(3)] for a in ("D1","D2","beta_0.5","F:0","F:1")})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(audit(a.input),indent=2)+"\n")
