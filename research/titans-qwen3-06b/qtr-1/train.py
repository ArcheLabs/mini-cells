"""QTR-1 Gate 1 rescue: one arm/seed, resumable at episode boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from profiles import ARMS, PROFILES, SEEDS, STEPS, learning_rate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gate1"))
from adapter import MODEL, REVISION, TITANS, SEGMENT, LAYER, install
from final_smoke import FINEWEB_REV, build, fineweb, train_step
from evaluation import evaluate
from protocol import mixed


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parameter_hash(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from profiles import make_optimizer

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", choices=SEEDS, type=int, required=True)
    p.add_argument("--stop-after", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--engineering", action="store_true")
    a = p.parse_args()
    if not 1 <= a.stop_after <= STEPS:
        p.error("--stop-after must lie in 1..500")
    if a.engineering and (a.resume or a.stop_after not in (1, 2)):
        p.error("Engineering verification uses a fresh two-step run")
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    upstream = a.upstream.resolve()
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip() != TITANS:
        raise ValueError("Incorrect Titans revision")
    sys.path.insert(0,str(upstream))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    n = 1 if a.engineering else STEPS//2
    train = mixed(build(tok,a.seed,10000,"ABCD",n),fineweb(tok,n))
    dev = build(tok,20261119,30000,"EFGH",2 if a.engineering else 32)
    config = dict(experiment="QTR-1 Gate 1 rescue",source_commit=source,arm=a.arm,
                  seed=a.seed,model_revision=REVISION,titans_revision=TITANS,
                  fineweb_revision=FINEWEB_REV,data_hash=digest([train,dev]),
                  layer_index=LAYER,store_chunk=1,segment_tokens=SEGMENT,
                  episode_tokens=3*SEGMENT,engineering=a.engineering)
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,
        dtype=torch.float16,attn_implementation="eager").cuda().eval()
    backbone = dict(model.named_parameters())
    initial_backbone = parameter_hash(backbone)
    inner, profile = ARMS[a.arm]
    cell = install(model,"G1-A" if inner == .0025 else "G1-B")
    named = {k:v for k,v in cell.memory.named_parameters() if v.requires_grad}
    if {id(v) for v in model.parameters() if v.requires_grad} != {id(v) for v in named.values()}:
        raise RuntimeError("Trainable parameters outside memory")
    config["initial_memory_sha256"] = parameter_hash(named)
    config["outer_profile"] = profile
    config["inner_lr_max"] = inner
    config["backbone_sha256"] = initial_backbone
    opt = make_optimizer(profile,named,backbone.values(),upstream)
    path = a.output/"resume.pt"
    start = 0
    if a.resume or a.resume_from:
        incoming = a.resume_from or path
        if a.resume_from and hashlib.sha256(incoming.read_bytes()).hexdigest() != a.resume_sha256:
            raise ValueError("Remote checkpoint hash mismatch")
        if a.resume_from:
            a.output.mkdir(parents=True)
            (a.output/"config.json").write_text(json.dumps(config,indent=2))
        record = torch.load(incoming,map_location="cpu",weights_only=True)
        if record["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        with torch.no_grad():
            for key, param in named.items():
                param.copy_(record["parameters"][key])
        opt.load_state_dict(record["optimizer"])
        torch.set_rng_state(record["cpu_rng"])
        torch.cuda.set_rng_state(record["cuda_rng"])
        start = record["step"]
    else:
        a.output.mkdir(parents=True)
        (a.output/"config.json").write_text(json.dumps(config,indent=2))
    if start >= a.stop_after:
        p.error("Checkpoint is already at or beyond requested stop step")

    def assess(step):
        rows = evaluate(model,cell,dev,"cuda")
        (a.output/f"development-{step:03}.json").write_text(json.dumps(rows,allow_nan=False))

    if not (a.resume or a.resume_from):
        assess(0)
    for step in range(start+1,a.stop_after+1):
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        lr = learning_rate(profile,step)
        for group in opt.param_groups:
            group["lr"] = lr
        metrics = train_step(model,cell,train[step-1],opt,"cuda",clip=PROFILES[profile]["clip"])
        peak = torch.cuda.max_memory_allocated()
        metrics.update(step=step,lr=lr,seconds=time.monotonic()-begin,peak_bytes=peak)
        print(json.dumps(metrics,allow_nan=False),flush=True)
        (a.output/f"step-{step:03}.json").write_text(json.dumps(metrics,allow_nan=False))
        if peak > 13*2**30:
            raise RuntimeError("13 GiB allocation target exceeded")
        if step % 100 == 0 or step == a.stop_after:
            if any(p.requires_grad or p.grad is not None for p in backbone.values()) or parameter_hash(backbone) != initial_backbone:
                raise RuntimeError("Backbone freeze audit failed")
            tmp = path.with_suffix(".tmp")
            torch.save(dict(config=config,step=step,parameters={k:v.detach().cpu() for k,v in named.items()},
                            optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state()),tmp)
            tmp.replace(path)
            assess(step)
    (a.output/"decision.json").write_text(json.dumps(dict(
        format="minicells.qtr-1.v1", status="ENGINEERING_ONLY" if a.engineering else "ARM_SEGMENT_COMPLETE",
        holdout="SEALED",step=a.stop_after,backbone_freeze="PASS",
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=config),indent=2))


if __name__ == "__main__":
    main()
