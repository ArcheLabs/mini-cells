#!/usr/bin/env python3
"""Core 009B-3 discovery/confirmation runner, reporter, checkpoint publisher."""
from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path
import torch
from minicells.real_representation_006_experiment import prepare_seed
from minicells.real_representation_006_io import extract_frozen_sequences,load_foundation,load_frozen_cache,save_frozen_cache,select_real_sequences,write_data_manifest
from minicells.real_representation_007_config import CoreValidation007Config
from minicells.real_representation_009b1_experiment import analysis_sequences,extract_causal_sequences
from minicells.real_representation_009b3_experiment import run_discovery,run_confirmation,select_router,confirmation_gate
from publish_core_validation_007 import _authenticated_git_env,_check_branch
from publish_experiment_results import EXPECTED_ORIGIN,run_git

ROOT=Path(__file__).resolve().parents[2]
VAL=ROOT/"research/validations/core-009b3-deployable-effect-addressability"; PRO=VAL/"protocol.json"; PLOCK=VAL/"parent-lock.json"; RLOCK=VAL/"router-lock.json"
RES=ROOT/"results/core-validation-009b3-deployable-effect-addressability"; ART=ROOT/"artifacts/experiments/core-validation-009b3-deployable-effect-addressability"
C7=ROOT/"research/validations/core-007-functional-boundary-discovery/protocol.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def protocol(): return json.loads(PRO.read_text())
def heartbeat(seed,msg): print(f"[core009b3 seed={seed}] {msg}",flush=True)
def load_parent():
    if not PLOCK.is_file(): raise RuntimeError("PREPARED_WAITING_FOR_009B2_LOCK: committed parent-lock.json is missing")
    x=json.loads(PLOCK.read_text()); p=protocol()
    if x.get("protocol_sha256")!=sha(PRO) or x.get("parent_status")!=p["parent_evidence"]["required_status"] or x.get("parent_supported") is not True: raise RuntimeError("invalid 009B-3 parent lock")
    if not 1<=int(x["locked_dimension"])<=int(p["parent_evidence"]["maximum_locked_dimension"]): raise RuntimeError("invalid parent dimension")
    return x
def load_router(parent):
    if not RLOCK.is_file(): raise RuntimeError("confirmation requires committed router-lock.json")
    x=json.loads(RLOCK.read_text())
    if x.get("protocol_sha256")!=sha(PRO) or x.get("parent_lock_sha256")!=sha(PLOCK) or int(x["locked_dimension"])!=int(parent["locked_dimension"]): raise RuntimeError("invalid router lock")
    if x.get("confirmation_allowed") is not True: raise RuntimeError("router lock forbids confirmation")
    return x
def seed_format(phase): return f"minicells.core-validation.deployable-effect-addressability-{phase}-seed.v1"
def valid(path,phase,seed):
    if not path.is_file() or not PLOCK.is_file(): return False
    try:x=json.loads(path.read_text())
    except: return False
    return x.get("format")==seed_format(phase) and x.get("phase")==phase and int(x.get("seed",-1))==seed and x.get("protocol_sha256")==sha(PRO) and x.get("parent_lock_sha256")==sha(PLOCK)
def hydrate(phase,seed):
    a=ART/phase/"seeds"/f"seed-{seed}.json"; b=RES/phase/"seeds"/f"seed-{seed}.json"
    if not valid(a,phase,seed): return False
    b.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a,b); print(f"[core009b3] hydrated {phase} seed {seed}",flush=True); return True

def run_seed(phase,seed,device,parent,router):
    p=protocol(); cfg=CoreValidation007Config.from_protocol(C7); out=RES/phase/"seeds"; out.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True); started=time.time()
    heartbeat(seed,f"loading foundation on {device}"); tok,model=load_foundation(cfg.base,device=device); heartbeat(seed,"foundation loaded; selecting pinned data")
    records,manifest=select_real_sequences(cfg.base,tok); got=str(manifest["manifest_sha256"])
    if got!=p["data"]["expected_manifest_sha256"]: raise RuntimeError("data manifest mismatch")
    write_data_manifest(manifest,RES/"data-manifest.json"); cache=RES/"frozen-hidden.pt"; frozen=load_frozen_cache(manifest,cache)
    if frozen is None:
        heartbeat(seed,"hidden cache miss; extracting"); frozen=extract_frozen_sequences(records,model,device=device); save_frozen_cache(frozen,manifest,cache)
    else: heartbeat(seed,"reused frozen hidden cache")
    W=model.embed_out.weight.detach().clone(); del model
    if device.type=="cuda": torch.cuda.empty_cache()
    heartbeat(seed,"preparing seeded projection"); u,_,_,projected=prepare_seed(frozen,cfg.base,seed=seed)
    heartbeat(seed,"extracting frozen write signatures"); causal=analysis_sequences(extract_causal_sequences(projected,u,W,device=device))
    d=int(parent["locked_dimension"]); heartbeat(seed,f"address pack ready: train={sum(x.partition=='train' for x in causal)} eval={sum(x.partition=='eval' for x in causal)} d={d}")
    if phase=="discovery":
        heartbeat(seed,"CPU router discovery; causal NLL forbidden"); q=run_discovery(causal,p,seed,d)
    else:
        heartbeat(seed,f"causal confirmation router={router['router_name']}"); q=run_confirmation(causal,p,seed,d,router["router_name"],u,W,device)
    q.update({"phase":phase,"protocol_version":p["protocol_version"],"protocol_sha256":sha(PRO),"parent_lock_sha256":sha(PLOCK),"data_manifest_sha256":got,"device":str(device),"elapsed_seconds":time.time()-started})
    path=out/f"seed-{seed}.json"; path.write_text(json.dumps(q,indent=2,sort_keys=True)+"\n"); heartbeat(seed,f"complete -> {path}")

def load_seed(phase,seed):
    p=RES/phase/"seeds"/f"seed-{seed}.json"; return json.loads(p.read_text()) if valid(p,phase,seed) else None
def report(phase,parent):
    p=protocol(); out=RES/phase; out.mkdir(parents=True,exist_ok=True); seeds=list(map(int,p[phase]["seeds"])); qs=[q for s in seeds if (q:=load_seed(phase,s))]; done=[q["seed"] for q in qs]; missing=[s for s in seeds if s not in done]
    if phase=="discovery":
        name,summary=select_router(qs,p); allowed=not missing and name is not None; status="DISCOVERY_INCOMPLETE" if missing else p["discovery"]["positive_status"] if allowed else p["discovery"]["failure_status"]
        d={"format":"minicells.core-validation.deployable-effect-addressability-discovery-decision.v1","experiment_id":p["experiment_id"],"protocol_sha256":sha(PRO),"parent_lock_sha256":sha(PLOCK),"completed_seeds":done,"missing_seeds":missing,"locked_dimension":parent["locked_dimension"],"router_selection":summary,"locked_router_name":name,"confirmation_allowed":allowed,"scientific_decision":False,"status":status}
        if allowed:
            spec=next(x for x in p["router_candidates"] if x["name"]==name)
            lock={"format":"minicells.core-validation.deployable-effect-addressability-router-lock.v1","experiment_id":p["experiment_id"],"protocol_sha256":sha(PRO),"parent_lock_sha256":sha(PLOCK),"locked_dimension":parent["locked_dimension"],"router_name":name,"router_family":spec["family"],"discovery_seeds":seeds,"selection_uses_effect_space_only":True,"causal_metrics_used_for_selection":False,"confirmation_allowed":True,"scientific_decision":False}
            (out/"router-lock.json").write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n")
    else:
        gates=[confirmation_gate(q,p) for q in qs]; complete=not missing
        if complete:
            iid=all(g["iid_pass"] for g in gates); ood=all(g["ood_pass"] for g in gates); supported=iid and ood
            status=p["confirmation"]["positive_status"] if supported else p["confirmation"]["iid_only_status"] if iid else p["confirmation"]["negative_status"]; scientific=True
        else: supported=None; scientific=False; status="CONFIRMATION_INCOMPLETE"
        names=sorted({q["router_name"] for q in qs})
        d={"format":"minicells.core-validation.deployable-effect-addressability-confirmation-decision.v1","experiment_id":p["experiment_id"],"protocol_sha256":sha(PRO),"parent_lock_sha256":sha(PLOCK),"completed_seeds":done,"missing_seeds":missing,"locked_dimension":parent["locked_dimension"],"router_name":names[0] if len(names)==1 else None,"gate_rows":gates,"passed_seeds":sum(g["pass"] for g in gates),"scientific_decision":scientific,"supported":supported,"status":status}
    (out/"decision.json").write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); (out/"RESULTS.md").write_text(f"# Core Validation 009B-3\n\n- Phase: `{phase}`\n- Status: `{d['status']}`\n- Completed: `{done}`\n- Missing: `{missing}`\n")
    print(json.dumps(d,indent=2,sort_keys=True)); return d

def publish(phase,decision,branch,secret):
    dest=ART/phase
    if dest.exists(): shutil.rmtree(dest)
    dest.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(RES/phase,dest,ignore=lambda d,n:[x for x in n if x in {"frozen-hidden.pt","address-pack.pt"}])
    ART.mkdir(parents=True,exist_ok=True); shutil.copy2(PRO,ART/"protocol.json"); shutil.copy2(PLOCK,ART/"parent-lock.json")
    if phase=="discovery" and (RES/phase/"router-lock.json").is_file(): shutil.copy2(RES/phase/"router-lock.json",RLOCK); shutil.copy2(RLOCK,ART/"router-lock.json")
    elif RLOCK.is_file(): shutil.copy2(RLOCK,ART/"router-lock.json")
    if (RES/"data-manifest.json").is_file(): shutil.copy2(RES/"data-manifest.json",ART/"data-manifest.json")
    run_git(ROOT,"config","user.name","MiniCells Kaggle"); run_git(ROOT,"config","user.email","kaggle@minicells.local"); paths=[ART.relative_to(ROOT).as_posix(),PLOCK.relative_to(ROOT).as_posix()]
    if RLOCK.is_file(): paths.append(RLOCK.relative_to(ROOT).as_posix())
    run_git(ROOT,"add","--",*paths)
    if run_git(ROOT,"diff","--cached","--quiet",check=False).returncode!=0:
        msg=("research: lock Core Validation 009B-3 deployable router" if phase=="discovery" and decision.get("confirmation_allowed") else "research: record Core Validation 009B-3 deployable effect addressability" if phase=="confirmation" and decision.get("scientific_decision") else f"research: checkpoint Core Validation 009B-3 {phase}")
        run_git(ROOT,"commit","-m",msg)
    _check_branch(branch)
    with _authenticated_git_env(secret) as env: r=run_git(ROOT,"push",EXPECTED_ORIGIN+".git",f"HEAD:refs/heads/{branch}",env=env,check=False)
    if r.returncode: raise RuntimeError((r.stderr or r.stdout or "").strip())
    print(f"pushed Core 009B-3 artifacts to {branch}")

def main():
    a=argparse.ArgumentParser(); a.add_argument("--phase",choices=("discovery","confirmation"),required=True); a.add_argument("--branch",default="codex/core-validation-009b3-deployable-effect-addressability"); a.add_argument("--secret-name",default="GITHUB_TOKEN"); a.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda"); a.add_argument("--push-results",action="store_true"); a.add_argument("--force",action="store_true"); x=a.parse_args()
    parent=load_parent(); router=load_router(parent) if x.phase=="confirmation" else None; p=protocol(); device=torch.device("cuda" if x.device=="auto" and torch.cuda.is_available() else "cpu" if x.device=="auto" else x.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    for seed in map(int,p[x.phase]["seeds"]):
        path=RES/x.phase/"seeds"/f"seed-{seed}.json"; complete=False if x.force else valid(path,x.phase,seed) or hydrate(x.phase,seed)
        if complete: print(f"[core009b3] {x.phase} seed={seed} complete; skipping",flush=True)
        else: run_seed(x.phase,seed,device,parent,router)
        d=report(x.phase,parent)
        if x.push_results: publish(x.phase,d,x.branch,x.secret_name)
    return 0
if __name__=="__main__": raise SystemExit(main())
