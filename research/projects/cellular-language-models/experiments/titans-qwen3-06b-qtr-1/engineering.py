"""Verify actual optimizer continuation for each arm without opening holdout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from profiles import ARMS


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    elif isinstance(a, dict):
        if a.keys()!=b.keys():
            raise RuntimeError("Checkpoint field mismatch")
        for k in a:
            equal(a[k],b[k])
    elif isinstance(a, (tuple,list)):
        for x,y in zip(a,b,strict=True):
            equal(x,y)
    elif a!=b:
        raise RuntimeError("Checkpoint metadata mismatch")


def main(upstream, output):
    output.mkdir(parents=True)
    evidence = {}
    for arm in ARMS:
        base=output/arm
        def run(name, stop, *args):
            dest=base/name
            subprocess.run([sys.executable,"train.py",str(upstream),str(dest),
                "--arm",arm,"--seed","42","--engineering","--stop-after",str(stop),*args],check=True)
            return dest/"resume.pt"
        continuous=run("continuous",2)
        checkpoint=run("pause",1)
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        resumed=run("resumed",2,"--resume-from",str(checkpoint),"--resume-sha256",sha)
        a,b=(torch.load(p,weights_only=True,map_location="cpu") for p in (continuous,resumed))
        equal(a,b)
        evidence[arm]=dict(status="PASS", initial=a["config"]["initial_memory_sha256"],
                           data=a["config"]["data_hash"])
        for path in (continuous,checkpoint,resumed):
            path.unlink()
    for key in ("initial","data"):
        if len({row[key] for row in evidence.values()})!=1:
            raise RuntimeError(f"Unpaired {key}")
    (output/"decision.json").write_text(json.dumps(dict(format="minicells.qtr-1.v1",
        status="ENGINEERING_ONLY", holdout="SEALED", arms=evidence),indent=2))


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
