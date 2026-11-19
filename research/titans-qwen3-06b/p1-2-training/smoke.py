"""Stage 0: pinned real-model synthetic engineering smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from adapter import install

REV = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
UP = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"
MODEL = "Qwen/Qwen3-0.6B-Base"
KIND = "minicells.titans-p1-2-training.v1"


def episodes(tok):
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    values = [encode(" "+c) for c in "ABCDEFGHIJKL"]
    if not all(len(v)==1 for v in values) or len({v[0] for v in values}) != 12:
        raise ValueError("Candidate tokenizer mismatch")
    rng = random.Random(42)
    out = []
    for _ in range(8):
        keys = rng.sample(range(10000,99999), 4)
        chosen = rng.sample(range(12),4)
        labels = [values[i][0] for i in chosen]
        write = "".join(f"Code {k}: {'ABCDEFGHIJKL'[i]}\n" for k,i in zip(keys,chosen))
        def pad(text, length):
            ids = encode(text)
            filler = encode(" An unrelated observation was recorded.\n") * 512
            if len(ids)>length:
                raise ValueError("Segment overflow")
            return filler[:length-len(ids)] + ids
        prompt = f"What value was assigned to code {keys[0]}?\nCode {keys[0]}:"
        if encode(prompt+" "+"ABCDEFGHIJKL"[chosen[0]]) != encode(prompt)+[labels[0]]:
            raise ValueError("Answer boundary mismatch")
        parts = [pad(write,512),pad("No codes are discussed.",512),
                 pad("The weather was clear.",512), pad(prompt,511)+[labels[0]]]
        if any(set(labels).intersection(part) for part in (parts[1],parts[2],parts[3][:-1])):
            raise ValueError("Answer leaked outside write segment")
        out.append(dict(keys=keys,labels=labels,segments=parts))
    return out


def update(model, wrapper, segments, optimizer, scheduler, chunk):
    wrapper.state = wrapper.pending = None
    counts = []
    early = None
    memory = wrapper.memory
    original = memory.memory_grads

    def observe(*args):
        counts[-1] += args[1].shape[0] // memory.heads
        return original(*args)

    memory.memory_grads = observe
    try:
        for i, ids in enumerate(segments):
            t = torch.tensor([ids], device="cuda")
            p = torch.arange(i*512,(i+1)*512,device=t.device)[None]
            if i < 3:
                counts.append(0)
                model.model(input_ids=t,position_ids=p,use_cache=False)
                wrapper.commit()
                if i == 0:
                    early = wrapper.state.updates
                if wrapper.state.seq_index != (i+1)*512:
                    raise ValueError("Unexpected store position")
            else:
                # Use the full segment; score only the prediction before the answer.
                h = model.model(input_ids=t,position_ids=p,use_cache=False).last_hidden_state
                logits = model.lm_head(h[:, -2]).float()
        if counts != [512//chunk]*3:
            raise ValueError(f"Unexpected store update counts: {counts}")
        loss = F.cross_entropy(logits,t[:, -1])
        params = {n:p for n,p in wrapper.named_parameters() if p.requires_grad}
        grads = torch.autograd.grad(loss,early,retain_graph=True)
        if not all(torch.isfinite(g).all() for g in grads) or not any(g.count_nonzero() for g in grads):
            raise ValueError("Missing finite read gradient")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = {n:None if p.grad is None else p.grad.float().norm().item() for n,p in params.items()}
        for prefix in ("memory_q_proj","memory_k_proj","memory_v_proj") if chunk == 16 else ():
            if not any(v is not None and v>0 for n,v in norms.items() if n.startswith(prefix)):
                raise ValueError(f"No projection gradient: {prefix}")
        torch.nn.utils.clip_grad_norm_(params.values(),1,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        return dict(loss=loss.item(),store_updates=counts,gradient_norms=norms)
    finally:
        del memory.memory_grads
        wrapper.state = wrapper.pending = None


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--titans-upstream",type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True)
    if subprocess.check_output(["git","-C",str(a.titans_upstream),"rev-parse","HEAD"],text=True).strip() != UP:
        raise ValueError("Wrong Titans revision")
    sys.path.insert(0,str(a.titans_upstream.resolve()))
    data = episodes(AutoTokenizer.from_pretrained(MODEL,revision=REV))
    raw = json.dumps(data).encode()
    (a.output/"synthetic_manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,dtype=torch.float16,
                                                attn_implementation="eager").cuda().eval()
    layer = model.model.layers[3]
    attention = layer.self_attn
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    hashes = {}
    with (a.output/"training.jsonl").open("w") as log:
        for arm,chunk in (("A0",512),("A4",16)):
            layer.self_attn = attention
            torch.manual_seed(42)
            wrapper = install(model,arm)
            named = {n:p for n,p in wrapper.named_parameters() if p.requires_grad}
            opt = torch.optim.AdamW(named.values(),lr=1e-4,weight_decay=0)
            sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda i:min(1,(i+1)/10))
            for step, episode in enumerate(data,1):
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                row = update(model,wrapper,episode["segments"],opt,sched,chunk)
                row.update(arm=arm,step=step,seconds=time.monotonic()-start,
                           peak_cuda_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row,allow_nan=False)+"\n")
                log.flush()
                print(row,flush=True)
            path = a.output/f"{arm}.pt"
            torch.save(dict(parameters={n:p.detach().cpu() for n,p in named.items()},
                            optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
                            step=8,seed=42,arm=arm,chunk=chunk,model_revision=REV,
                            input_sha256=digest,source_commit=source,titans_revision=UP),path)
            restored = torch.load(path,map_location="cpu",weights_only=True)
            for n,p in named.items():
                torch.testing.assert_close(p.detach().cpu(),restored["parameters"][n],atol=0,rtol=0)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name,obj in (("checkpoint_hashes",hashes),("decision",dict(format=KIND,status="SMOKE_ONLY",
                        scientific_status="NOT_EVALUATED",source_commit=source,titans_revision=UP,
                        model_revision=REV,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                        input_sha256=digest,tokens_per_arm=8*2048))):
        (a.output/f"{name}.json").write_text(json.dumps(obj,indent=2))


if __name__ == "__main__":
    main()
