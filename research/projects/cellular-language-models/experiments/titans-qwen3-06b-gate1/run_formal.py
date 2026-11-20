"""Gate 1 formal single-arm/seed runner, resumable at episode boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from adapter import ARMS, MODEL, REVISION, TITANS, install
from evaluation import evaluate, natural_regression
from final_smoke import build, fineweb, train_step, FINEWEB_REV
from protocol import mixed, SEEDS, STEPS, EVAL_STEPS


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("upstream", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--smoke-decision", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    smoke = json.loads(a.smoke_decision.read_text())
    if smoke.get("formal_ready") is not True or smoke.get("checkpoint_archive", {}).get("status") != "VERIFIED":
        raise ValueError("A successful published final smoke decision is required")
    identity = ARMS[a.arm][0]
    if identity and smoke.get("identity") != "COMPATIBLE_ON_PROBE":
        raise ValueError("Identity is not applicable per final smoke")
    def git(*args):
        return subprocess.check_output(["git", *args],text=True).strip()
    if git("-C",str(a.upstream),"rev-parse","HEAD") != TITANS:
        raise ValueError("Wrong Titans revision")
    sys.path.insert(0,str(a.upstream.resolve()))
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    train = mixed(build(tok,a.seed,10000,"ABCD",STEPS//2),
                  fineweb(tok,STEPS//2))
    valid = build(tok,20261119,30000,"EFGH",32)
    web_valid = fineweb(tok, 8, offset=3000)
    payload = json.dumps(dict(train=train,validation=valid,web_validation=web_valid,fineweb_revision=FINEWEB_REV))
    config = dict(arm=a.arm,seed=a.seed,steps=STEPS,data_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  model_revision=REVISION,titans_revision=TITANS,source_commit=git("rev-parse","HEAD"))
    if a.resume:
        if json.loads((a.output/"config.json").read_text()) != config:
            raise ValueError("Resume configuration mismatch")
    else:
        a.output.mkdir(parents=True)
        (a.output/"config.json").write_text(json.dumps(config,indent=2))
        (a.output/"data_manifest.json").write_text(payload)
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,dtype=torch.float16,
                    attn_implementation="eager").cuda().eval()
    cell = install(model,a.arm,identity_validated=identity)
    params = {n:p for n,p in cell.memory.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([dict(params=[p for n,p in params.items() if not n.startswith("history_mix")],lr=5e-4),
            dict(params=[p for n,p in params.items() if n.startswith("history_mix")],lr=5e-5)],
            betas=(.9,.95),weight_decay=.01)
    checkpoint = a.output/"resume.pt"
    step = 0
    if a.resume:
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        with torch.no_grad():
            for n,p in params.items():
                p.copy_(state["parameters"][n])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        step = state["step"]
    def save_eval(step):
        rows = evaluate(model,cell,valid,"cuda")
        (a.output/f"validation-{step:03}.json").write_text(json.dumps(dict(synthetic=rows,fineweb=natural_regression(model,cell,web_valid,"cuda")),allow_nan=False))
    if not a.resume:
        save_eval(0)
    with (a.output/"training.jsonl").open("a") as log:
        for i in range(step,STEPS):
            start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            row = train_step(model,cell,train[i],opt,"cuda")
            row.update(step=i+1,seconds=time.monotonic()-start,
                       peak_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row,allow_nan=False)+"\n")
            log.flush()
            print(row,flush=True)
            if row["peak_bytes"] > 13*2**30:
                raise RuntimeError("Allocation exceeded 13 GiB")
            if i+1 in EVAL_STEPS:
                save_eval(i+1)
                tmp = checkpoint.with_suffix(".tmp")
                torch.save(dict(config=config,step=i+1,parameters={n:p.detach().cpu() for n,p in params.items()},
                               optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state()),tmp)
                tmp.replace(checkpoint)
    (a.output/"decision.json").write_text(json.dumps(dict(status="ARM_SEED_COMPLETE",
        scientific_status="PENDING_AGGREGATE_AND_REGRESSION",test="SEALED",**config),indent=2))


if __name__ == "__main__":
    main()
