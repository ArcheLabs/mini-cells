#!/usr/bin/env python3
"""Run MiniCells Core Validation 004 — Growth-Restored Plasticity."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from typing import Any
import torch
from minicells.growth_plasticity_004_config import CoreValidation004Config
from minicells.growth_plasticity_004_ops import smoke_config
from minicells.growth_plasticity_004_experiment import run_primary_seed, summarize_experiment
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research" / "core-validation-004-protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-004-growth-restored-plasticity"

def _git(command:list[str])->str|None:
    try: return subprocess.check_output(["git",*command],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None

def _tracked_tree_dirty()->bool|None:
    try:
        a=subprocess.run(["git","diff","--quiet"],cwd=ROOT,check=False,stderr=subprocess.DEVNULL).returncode
        b=subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT,check=False,stderr=subprocess.DEVNULL).returncode
        return bool(a or b)
    except OSError: return None

def _sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _device(name:str)->torch.device:
    if name=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL); p.add_argument("--out",type=Path,default=DEFAULT_OUT); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--smoke",action="store_true"); p.add_argument("--seed",type=int,action="append",dest="seeds"); return p.parse_args()
def main()->int:
    args=parse_args(); protocol:dict[str,Any]=json.loads(args.protocol.read_text()); cfg=CoreValidation004Config.from_protocol(args.protocol)
    if args.smoke: cfg=smoke_config(cfg)
    device=_device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("formal Core Validation 004 requires CUDA")
    protocol_seeds=[int(x) for x in protocol["replication"]["seeds"]]; seeds=args.seeds or (protocol_seeds[:1] if args.smoke else protocol_seeds)
    if not args.smoke and seeds!=protocol_seeds: raise RuntimeError(f"formal Core Validation 004 must run exactly frozen seeds {protocol_seeds}")
    args.out.mkdir(parents=True,exist_ok=True); started=time.time(); runs=[]
    for seed in seeds:
        print(f"[core-004] seed={seed} device={device}",flush=True); run=run_primary_seed(cfg,seed=seed,device=device); runs.append(run); g=run["gate_summary"]; s=run["variants"]["local_tx_growth"]["summary"]
        print("[core-004] " f"seed={seed} base={g['pretraining']['base_normalized_mse']:.5f} " f"accept={s['effective_acceptance_rate']:.4f} rescue={s['growth_rescue_rate']:.4f} " f"reuse={s['private_cell_reuse_acceptance_rate']:.4f} FSR={s['false_safe_rate']:.4f} " f"damage_ratio={g['regression_damage_ratio_vs_local_always']:.4f} gain_ratio={g['committed_gain_ratio_vs_local_always']:.4f} " f"spawn_per_commit={s['spawned_cells_per_effective_commit']:.4f} pass={g['pass']}",flush=True)
    if args.smoke:
        decision={"status":"SMOKE_ONLY","pass":None,"scientific_decision":False,"passed_seeds":None,"total_seeds":len(runs),"reason":"Smoke mode validates execution only and cannot emit a scientific decision."}
    else:
        decision=summarize_experiment(runs,positive_status=str(protocol["gates"]["positive_status"]),negative_status=str(protocol["gates"]["negative_status"]))
    payload={"format":protocol["format"],"experiment_id":protocol["experiment_id"],"mode":"smoke" if args.smoke else "formal","protocol_sha256":_sha256(args.protocol),"parent_experiment":protocol["parent_experiment"],"provenance":{"code_commit":_git(["rev-parse","HEAD"]),"code_tree":_git(["rev-parse","HEAD^{tree}"]),"tracked_tree_dirty":_tracked_tree_dirty(),"torch":torch.__version__,"cuda":torch.version.cuda,"device":str(device),"gpu_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None},"runs":runs,"decision":decision,"elapsed_seconds":time.time()-started}
    raw=args.out/"raw.json"; raw.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(decision,indent=2,sort_keys=True)); print(f"wrote {raw}"); return 0
if __name__=="__main__": sys.exit(main())
