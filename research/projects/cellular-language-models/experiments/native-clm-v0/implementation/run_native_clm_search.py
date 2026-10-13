from __future__ import annotations

import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_clm_runtime as base
from native_clm_candidates import (
    CANDIDATE_NAMES, PARAMETER_TOLERANCE, VERIFIED_DATASET_REVISION,
    build_candidate, candidate_config, estimate_flops, parameter_summary,
    validation_log_token_auc,
)
base.DATASET_REVISION = VERIFIED_DATASET_REVISION
DEFAULT_ANCHOR = HERE / "results/dev/baseline-seed-91001"


def args_parser():
    p=argparse.ArgumentParser(description="Native CLM Phase-A architecture search"); s=p.add_subparsers(dest="command",required=True)
    q=s.add_parser("sweep"); q.add_argument("--models",nargs="+",choices=CANDIDATE_NAMES,default=list(CANDIDATE_NAMES)); q.add_argument("--profile",choices=tuple(base.PROFILES),default="baseline"); q.add_argument("--seed",type=int,default=91001); q.add_argument("--cache-root",type=Path,required=True); q.add_argument("--output-root",type=Path,required=True); q.add_argument("--anchor-dir",type=Path,default=DEFAULT_ANCHOR); q.add_argument("--allow-cpu",action="store_true"); q.add_argument("--no-resume",action="store_true")
    w=s.add_parser("worker"); w.add_argument("--model",choices=CANDIDATE_NAMES,required=True); w.add_argument("--profile",choices=tuple(base.PROFILES),required=True); w.add_argument("--seed",type=int,required=True); w.add_argument("--cache-root",type=Path,required=True); w.add_argument("--output-dir",type=Path,required=True); w.add_argument("--allow-cpu",action="store_true"); w.add_argument("--no-resume",action="store_true")
    return p.parse_args()


def device(allow_cpu):
    if torch.cuda.is_available(): return torch.device("cuda")
    if allow_cpu: return torch.device("cpu")
    raise RuntimeError("architecture search requires CUDA; --allow-cpu is smoke/debug only")


def read_rows(path: Path):
    numeric={"step","consumed_tokens","train_loss","learning_rate","grad_norm","parameters","elapsed_seconds","tokens_per_second","peak_vram_bytes","validation_nll","validation_ppl","validation_tokens"}
    with path.open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
    for r in rows:
        for k in numeric:
            if k in r and r[k] != "": r[k]=float(r[k])
    return rows


def train_one_candidate(name, corpus, profile, output_dir, seed, dev, resume):
    params=base.count_parameters(build_candidate(name)); t1=base.count_parameters(base.build_model(base.T1_NAME))
    if abs(params/t1-1)>=PARAMETER_TOLERANCE: raise RuntimeError(f"parameter mismatch: {name}={params}, T1={t1}")
    old=(base.C0_NAME,base.build_model,base.model_config,base.estimate_flops)
    base.C0_NAME=name
    base.build_model=lambda n,vocab_size=base.VOCAB_SIZE: build_candidate(n,vocab_size)
    base.model_config=lambda n,vocab_size=base.VOCAB_SIZE: {**candidate_config(n),"name":n,"vocab_size":vocab_size,"context_length":base.CONTEXT_LENGTH,"search_phase":"A-single-factor"}
    base.estimate_flops=lambda n,sequence_length,tokens: estimate_flops(n,tokens,sequence_length)
    try:
        summary=base.train_one(name,corpus=corpus,profile=profile,output_dir=output_dir,seed=seed,device=dev,resume=resume)
    finally:
        base.C0_NAME,base.build_model,base.model_config,base.estimate_flops=old
    rows=read_rows(output_dir/name/"checkpoints.csv")
    summary["checkpoints"]=rows; summary["validation_log_token_nll_auc"]=validation_log_token_auc(rows); summary["parameter_ratio_to_t1"]=params/t1
    (output_dir/name/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return summary


def run_worker(a):
    profile=base.PROFILES[a.profile]; corpus=base.prepare_corpus(a.cache_root,profile,hf_token=os.environ.get("HF_TOKEN") or None)
    s=train_one_candidate(a.model,corpus,profile,a.output_dir,a.seed,device(a.allow_cpu),not a.no_resume); f=s["final"] or {}
    print(json.dumps({"model":a.model,"parameters":s["parameters"],"validation_ppl":f.get("validation_ppl"),"auc":s["validation_log_token_nll_auc"],"tokens_per_second":f.get("tokens_per_second")},indent=2)); return 0


def worker_cmd(a,model,out):
    c=[sys.executable,str(Path(__file__).resolve()),"worker","--model",model,"--profile",a.profile,"--seed",str(a.seed),"--cache-root",str(a.cache_root),"--output-dir",str(out)]
    if a.no_resume:c.append("--no-resume")
    if a.allow_cpu:c.append("--allow-cpu")
    return c


def launch(a,models,out):
    ng=torch.cuda.device_count()
    if ng==0:
        if not a.allow_cpu: raise RuntimeError("No CUDA GPU visible")
        for m in models: subprocess.run(worker_cmd(a,m,out),check=True)
        return 0
    width=min(ng,len(models))
    for start in range(0,len(models),width):
        active=[]
        for gpu,m in enumerate(models[start:start+width]):
            env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); log=out/f"{m}.log"; h=log.open("w",encoding="utf-8")
            p=subprocess.Popen(worker_cmd(a,m,out),env=env,stdout=h,stderr=subprocess.STDOUT,text=True); active.append((m,p,h,log)); print(f"started {m} on physical GPU {gpu}",flush=True)
        failed=[]
        for m,p,h,log in active:
            code=p.wait(); h.close(); print(f"--- {m} ---\n{log.read_text(encoding='utf-8').rstrip()}",flush=True)
            if code: failed.append(f"{m} exited {code}; see {log}")
        if failed: raise RuntimeError("; ".join(failed))
    return width


def anchor_row(d,key,label):
    s=json.loads((d/f"{key}-summary.json").read_text(encoding="utf-8")); rows=read_rows(d/f"{key}-checkpoints.csv"); auc=validation_log_token_auc(rows); f=s["final"]
    return {"id":label,"model":s["model"],"kind":"frozen-anchor","parameters":int(s["parameters"]),"validation_ppl_10m":float(f["validation_ppl"]),"validation_nll_10m":float(f["validation_nll"]),"log_token_nll_auc":auc["auc"],"log_token_nll_auc_mean":auc["mean_nll"],"train_flops_estimate":float(s["flops"]["train_flops_estimate"]),"inference_flops_per_token_estimate":float(s["flops"]["inference_forward_flops_per_token"]),"tokens_per_second":float(f["tokens_per_second"]),"peak_vram_bytes":int(f["peak_vram_bytes"])}


def leaderboard(anchor,out,models):
    rows=[anchor_row(anchor,"T1","T1"),anchor_row(anchor,"C0","C0")]; t1=int(rows[0]["parameters"])
    for m in models:
        s=json.loads((out/m/"summary.json").read_text(encoding="utf-8")); f=s["final"]; auc=s["validation_log_token_nll_auc"]
        rows.append({"id":m.split("-")[0],"model":m,"kind":"phase-A-candidate","parameters":int(s["parameters"]),"validation_ppl_10m":float(f["validation_ppl"]),"validation_nll_10m":float(f["validation_nll"]),"log_token_nll_auc":float(auc["auc"]),"log_token_nll_auc_mean":float(auc["mean_nll"]),"train_flops_estimate":float(s["flops"]["train_flops_estimate"]),"inference_flops_per_token_estimate":float(s["flops"]["inference_forward_flops_per_token"]),"tokens_per_second":float(f["tokens_per_second"]),"peak_vram_bytes":int(f["peak_vram_bytes"])} )
    c0p=float(rows[1]["validation_ppl_10m"]); c0a=float(rows[1]["log_token_nll_auc_mean"])
    for r in rows: r["parameter_ratio_to_t1"]=int(r["parameters"])/t1; r["ppl_over_c0"]=float(r["validation_ppl_10m"])/c0p; r["auc_mean_over_c0"]=float(r["log_token_nll_auc_mean"])/c0a
    return rows


def run_sweep(a):
    if a.seed!=91001: raise RuntimeError("Phase-A is frozen to development seed 91001; do not consume other seeds")
    profile=base.PROFILES[a.profile]; ps=parameter_summary(); bad={n:r for n,r in ps.items() if n in CANDIDATE_NAMES and float(r["relative_error"])>=.01}
    if bad: raise RuntimeError(f"parameter mismatch: {bad}")
    corpus=base.prepare_corpus(a.cache_root,profile,hf_token=os.environ.get("HF_TOKEN") or None); out=a.output_root/f"architecture-search-{a.profile}-seed-{a.seed}"; out.mkdir(parents=True,exist_ok=True)
    protocol={"format":"minicells.native-clm-architecture-search.v1","phase":"A-single-factor","profile":profile.__dict__,"seed":a.seed,"models":a.models,"candidate_parameters":ps,"candidate_configs":{n:candidate_config(n) for n in a.models},"corpus_manifest":corpus.manifest,"environment_before_launch":base.environment_record(),"decision_rule":"rank 10M PPL and log-token NLL AUC; report compute separately; combine only causal single-factor wins"}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n",encoding="utf-8"); used=launch(a,list(a.models),out); rows=leaderboard(a.anchor_dir,out,list(a.models))
    (out/"architecture-leaderboard.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (out/"architecture-leaderboard.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"format":"minicells.native-clm-architecture-search-summary.v1","profile":a.profile,"seed":a.seed,"models":a.models,"gpus_used":used,"leaderboard":rows}; (out/"search-summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== Phase A leaderboard ===")
    for r in sorted(rows,key=lambda x:float(x["validation_ppl_10m"])): print(f"{r['id']:>3} ppl={float(r['validation_ppl_10m']):8.4f} auc_mean={float(r['log_token_nll_auc_mean']):7.4f} params={int(r['parameters']):,} tok/s={float(r['tokens_per_second']):,.0f}")
    print(f"output: {out}"); return 0


def main():
    a=args_parser(); return run_worker(a) if a.command=="worker" else run_sweep(a)
if __name__=="__main__": raise SystemExit(main())
