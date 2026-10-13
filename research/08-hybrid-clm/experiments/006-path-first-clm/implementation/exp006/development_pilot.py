"""Unregistered REAL DATA development pilot; confirmation is never accessed."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from .assets import prepare
from .baseline import make_model
from .checkpoint import identity,file_sha256
from .config import BASE,ROOT,digest,validate
from .data import assert_disjoint,select_windows
from .profiling import benchmark,parity
from .reuse import canonical_sha256,make_batch,write_json
from .training import _train

SHARD="data/train-00000-of-00004-2d5a1467fff1081b.parquet"
PILOT_PROTOCOL={"status":"UNREGISTERED_DEVELOPMENT_PILOT","scientific_decision":False}
ARMS=("R0","R1","R2","M0")

def source_commit():
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def proposed(tokens):
    c=json.loads((BASE/"config/proposed.json").read_text())
    if tokens not in (1000000,10000000):raise ValueError("pilot token budget must be 1M or 10M")
    t=c["training"];t["steps"]=tokens//(t["batch_size"]*t["sequence_length"]);t["tokens"]=tokens
    t["warmup_steps"]=50 if tokens==1000000 else 200
    t["evaluate_every"]=100 if tokens==1000000 else 500
    validate(c)
    return c

def verified(root,c):
    root=Path(root);folder=root/"assets";mp=folder/"manifest.json"
    if digest(mp)!=c["data"]["manifest_sha256"]:raise ValueError("pilot manifest digest mismatch")
    m=json.loads(mp.read_text())
    if m["experiment_id"]!="006-path-first-clm" or m["confirmation_materialized"]:raise ValueError("unexpected confirmation data")
    if m["source_parquet_sha256"]!=c["data"]["source_sha256"]:raise ValueError("raw data provenance mismatch")
    if m["sequence_length"]!=c["training"]["sequence_length"] or m["vocab_size"]!=c["model"]["vocab_size"]:raise ValueError("asset dimensions mismatch")
    allowed={"train.npy","development.npy","tokenizer.json","windows.json"}
    if set(m["files"])!=allowed:raise ValueError("unexpected pilot asset files")
    for name,sha in m["files"].items():
        if digest(folder/name)!=sha:raise ValueError("corrupt pilot asset "+name)
    w=json.loads((folder/"windows.json").read_text())
    assert_disjoint(w["train"]["story_hashes"],w["development"]["story_hashes"])
    starts,groups=select_windows(w["development"]["starts"],w["development"]["groups"],c["evaluation"]["windows"],c["evaluation"]["window_seed"])
    return {"train":np.load(folder/"train.npy",mmap_mode="r"),
            "development":np.load(folder/"development.npy",mmap_mode="r"),
            "train_starts":w["train"]["starts"],"eval_starts":starts,
            "eval_groups":groups,"manifest_sha256":digest(mp)}

def load(root,tokens):
    root=Path(root);c=json.loads((root/"pilot-config.json").read_text());expected=proposed(tokens)
    for field in ("source_sha256","manifest_sha256"):expected["data"][field]=c["data"][field]
    if canonical_sha256(expected)!=canonical_sha256(c):raise ValueError("frozen pilot config differs from requested run")
    if any(c["authorization"].values()) or c["seeds"]["confirmation"]:raise ValueError("pilot cannot grant confirmation access")
    validate(c)
    return c,verified(root,c)

def prepare_pilot(root,tokens):
    root=Path(root)
    if (root/"pilot-config.json").exists():
        c,_=load(root,tokens)
        print(json.dumps({"status":"DEV_PILOT_ASSETS_VERIFIED","manifest_sha256":c["data"]["manifest_sha256"]}),flush=True)
        return
    if (root/"assets").exists():raise FileExistsError("incomplete asset directory; inspect instead of silently overwriting")
    from huggingface_hub import hf_hub_download
    c=proposed(tokens)
    src=hf_hub_download(repo_id=c["data"]["dataset"],repo_type="dataset",revision=c["data"]["revision"],filename=SHARD)
    c["data"]["source_sha256"]=digest(src)
    root.mkdir(parents=True,exist_ok=True)
    prepare(c,root/"assets",src,c["data"]["source_sha256"])
    c["data"]["manifest_sha256"]=digest(root/"assets/manifest.json")
    verified(root,c)
    write_json(root/"pilot-config.json",c)
    write_json(root/"pilot-provenance.json",{
        "status":"UNREGISTERED_DEVELOPMENT_PILOT","scientific_decision":False,
        "source_commit":source_commit(),"dataset_revision":c["data"]["revision"],
        "source_shard":SHARD,"source_parquet_sha256":c["data"]["source_sha256"],
        "asset_manifest_sha256":c["data"]["manifest_sha256"],
        "tokenizer_sha256":digest(root/"assets/tokenizer.json"),
        "config_sha256":canonical_sha256(c),
        "confirmation_access":False,"formal_registration":False})
    print(json.dumps({"status":"DEV_PILOT_ASSETS_READY","root":str(root)}),flush=True)

def ident(c,arm,seed,source):
    if arm not in ARMS or seed<0 or source!=source_commit():raise ValueError("invalid pilot arm/seed/source")
    i=identity(source,c,PILOT_PROTOCOL,c["data"]["manifest_sha256"],arm,seed,"development")
    i["run_class"]="UNREGISTERED_DEV_PILOT"
    return i

def train_pilot(root,tokens,arm,seed,source,device):
    c,assets=load(root,tokens);i=ident(c,arm,seed,source)
    output=Path(root)/"runs"/("seed-"+str(seed))/arm;output.mkdir(parents=True,exist_ok=True)
    marker=output/"pilot-run.json"
    record={"identity":i,"scientific_decision":False,"scope":"unregistered real-data development pilot"}
    if marker.exists():
        if json.loads(marker.read_text())!=record:raise ValueError("pilot resume identity changed")
    else:write_json(marker,record)
    if (output/"metrics.json").exists():
        result=json.loads((output/"metrics.json").read_text())
        if result["identity"]!=i:raise ValueError("completed pilot identity differs")
        print(json.dumps({"status":"DEV_PILOT_ALREADY_COMPLETE","arm":arm}),flush=True);return
    _train(c,i,assets,output,device=device,resume=True)
    print(json.dumps({"status":"DEV_PILOT_TRAIN_COMPLETE","arm":arm,"folder":str(output)}),flush=True)

def profile_pilot(root,tokens,arm,seed,source,device):
    c,assets=load(root,tokens);i=ident(c,arm,seed,source)
    output=Path(root)/"runs"/("seed-"+str(seed))/arm
    m=json.loads((output/"metrics.json").read_text());ckpt=output/"latest.pt"
    if m["identity"]!=i or m["status"]!="completed" or m["checkpoint_sha"]!=file_sha256(ckpt):
        raise ValueError("pilot final checkpoint or identity mismatched")
    dest=output/"pilot-decode-profile.json"
    if dest.exists():
        saved=json.loads(dest.read_text())
        if saved["identity"]!=i or saved["checkpoint_sha"]!=m["checkpoint_sha"]:raise ValueError("profile identity mismatch")
        print(json.dumps({"status":"DEV_PILOT_PROFILE_VERIFIED","arm":arm}),flush=True);return
    model=make_model(c["model"],arm,seed).to(device)
    payload=torch.load(ckpt,map_location="cpu",weights_only=False)
    if payload["identity"]!=i:raise ValueError("pilot checkpoint payload identity mismatch")
    model.load_state_dict(payload["model"]);model.eval()
    x,_=make_batch(assets["development"],np.asarray(assets["eval_starts"][:1]),
                   c["training"]["sequence_length"],torch.device(device))
    check=parity(model,x,c["evaluation"]["parity_atol"],c["evaluation"]["parity_rtol"])
    if not check["passed"]:raise RuntimeError("pilot incremental decode parity failed")
    performance=benchmark(model,x,c)
    write_json(dest,{"identity":i,"checkpoint_sha":m["checkpoint_sha"],
                     "status":"UNREGISTERED_DEV_PILOT_PROFILE","scientific_decision":False,
                     "parity":check,"profile":performance})
    print(json.dumps({"status":"DEV_PILOT_PROFILE_COMPLETE","arm":arm,"p50_ms":performance["warm_p50_ms"]}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("prepare","train","profile"),required=True)
    p.add_argument("--workdir",required=True)
    p.add_argument("--tokens",type=int,choices=(1000000,10000000),default=1000000)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--seed",type=int,default=6060611)
    p.add_argument("--source-commit")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    if a.mode=="prepare":prepare_pilot(a.workdir,a.tokens);return
    if not a.arm or not a.source_commit:p.error("train/profile require arm and immutable source commit")
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real-data pilot requires CUDA, not synthetic CPU preflight")
    if a.mode=="train":train_pilot(a.workdir,a.tokens,a.arm,a.seed,a.source_commit,a.device)
    else:profile_pilot(a.workdir,a.tokens,a.arm,a.seed,a.source_commit,a.device)

if __name__=="__main__":main()
