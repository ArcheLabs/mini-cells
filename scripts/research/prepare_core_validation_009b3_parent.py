#!/usr/bin/env python3
"""Commit a parent lock only after the exact remote Core 009B-2 branch is formally positive."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
from publish_core_validation_007 import _authenticated_git_env,_check_branch
from publish_experiment_results import EXPECTED_ORIGIN,run_git
ROOT=Path(__file__).resolve().parents[2]
VAL=ROOT/"research/validations/core-009b3-deployable-effect-addressability"; PRO=VAL/"protocol.json"; LOCK=VAL/"parent-lock.json"
PDEC="artifacts/experiments/core-validation-009b2-persistent-effect-geometry/confirmation/decision.json"; PBAS="artifacts/experiments/core-validation-009b2-persistent-effect-geometry/basis-lock.json"; PPRO="research/validations/core-009b2-persistent-effect-geometry/protocol.json"
def sha(b): return hashlib.sha256(b).hexdigest()
def gout(*a): return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
def show(ref,path): return subprocess.check_output(["git","show",f"{ref}:{path}"],cwd=ROOT)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--branch",default="codex/core-validation-009b3-deployable-effect-addressability"); ap.add_argument("--secret-name",default="GITHUB_TOKEN"); ap.add_argument("--push-results",action="store_true"); x=ap.parse_args()
    p=json.loads(PRO.read_text()); parent=p["parent_evidence"]["required_branch"]; rr=f"refs/remotes/origin/{parent}"; subprocess.run(["git","fetch",EXPECTED_ORIGIN+".git",f"{parent}:{rr}"],cwd=ROOT,check=True); commit=gout("rev-parse",rr); ref=f"origin/{parent}"
    dr,br,pr=show(ref,PDEC),show(ref,PBAS),show(ref,PPRO); d,b,pp=json.loads(dr),json.loads(br),json.loads(pr)
    if d.get("status")!=p["parent_evidence"]["required_status"] or d.get("scientific_decision") is not True or d.get("supported") is not True: raise RuntimeError("Core 009B-2 is not formally positive")
    dim=int(d.get("locked_dimension",-1))
    if dim!=int(b.get("locked_dimension",-2)) or not 1<=dim<=int(p["parent_evidence"]["maximum_locked_dimension"]): raise RuntimeError("invalid Core 009B-2 dimension")
    ps=sha(pr)
    if d.get("protocol_sha256")!=ps or b.get("protocol_sha256")!=ps: raise RuntimeError("Core 009B-2 protocol hash mismatch")
    rho=float(pp["parent_evidence"]["core009b1_locked_rho"])
    if rho!=float(p["parent_evidence"]["expected_causal_rho"]): raise RuntimeError("unexpected inherited causal rho")
    lock={"format":"minicells.core-validation.deployable-effect-addressability-parent-lock.v1","experiment_id":p["experiment_id"],"protocol_sha256":hashlib.sha256(PRO.read_bytes()).hexdigest(),"parent_branch":parent,"parent_result_commit":commit,"parent_decision_sha256":sha(dr),"parent_basis_lock_sha256":sha(br),"parent_protocol_sha256":ps,"locked_dimension":dim,"causal_rho":rho,"parent_status":d["status"],"parent_scientific_decision":True,"parent_supported":True,"discovery_allowed":True,"scientific_decision":False}
    LOCK.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); run_git(ROOT,"config","user.name","MiniCells Kaggle"); run_git(ROOT,"config","user.email","kaggle@minicells.local"); run_git(ROOT,"add","--",LOCK.relative_to(ROOT).as_posix())
    if run_git(ROOT,"diff","--cached","--quiet",check=False).returncode!=0: run_git(ROOT,"commit","-m","research: lock Core Validation 009B-3 parent geometry")
    print(json.dumps(lock,indent=2,sort_keys=True))
    if x.push_results:
        _check_branch(x.branch)
        with _authenticated_git_env(x.secret_name) as env: r=run_git(ROOT,"push",EXPECTED_ORIGIN+".git",f"HEAD:refs/heads/{x.branch}",env=env,check=False)
        if r.returncode: raise RuntimeError((r.stderr or r.stdout or "").strip())
        print(f"pushed Core 009B-3 parent lock to {x.branch}")
    return 0
if __name__=="__main__": raise SystemExit(main())
