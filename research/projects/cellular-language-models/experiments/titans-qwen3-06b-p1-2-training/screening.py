"""P1.2-TR Stage 1: paired synthetic screening; confirmation stays sealed."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

import torch

from adapter import ARMS, install
from smoke import episodes, update, MODEL, REV, UP, KIND


@torch.no_grad()
def evaluate(model, wrapper, validation, donors):
    from titans_pytorch.neural_memory import NeuralMemState

    block = model.model.layers[3]

    def hidden(ids, offset):
        x = torch.tensor([ids], device="cuda")
        pos = torch.arange(offset, offset+len(ids), device=x.device)[None]
        return model.model(input_ids=x, position_ids=pos, use_cache=False).last_hidden_state

    def history(parts):
        wrapper.state = wrapper.pending = None
        for i, ids in enumerate(parts):
            hidden(ids, i*512)
            wrapper.commit()
        return wrapper.state

    weights = wrapper.memory.init_weights(1)
    slow = NeuralMemState(0, weights, None, None, tuple(t[:, None] for t in weights))
    correct = [history(e["segments"][:3]) for e in validation]
    shuffled = [history([e["shuffled"], *e["segments"][1:3]]) for e in validation]
    rows = []
    try:
        for i, episode in enumerate(validation):
            query = episode["segments"][3]
            target = query[-1]

            def score(state, override=None):
                wrapper.state, wrapper.read_override = state, override
                logits = model.lm_head(hidden(query,1536)[:, -2]).float()[0]
                lp = logits.log_softmax(0)
                return dict(nll=-lp[target].item(), exact=int(logits.argmax().item()==target),
                            candidate_correct=int(episode["labels"][logits[episode["labels"]].argmax().item()]==target))

            block.self_attn = wrapper.attention
            empty = score(None)
            block.self_attn = wrapper
            ds = [score(correct[j]) for j in donors[i]]
            rows.append(dict(index=i,no_memory=empty,w0=score(slow),correct=score(correct[i]),
                             shuffled=score(shuffled[i]),zero_read=score(correct[i],"zero"),donors=ds,
                             donor_mean={k:mean(d[k] for d in ds) for k in ds[0]}))
        return rows
    finally:
        block.self_attn = wrapper
        wrapper.state = wrapper.pending = wrapper.read_override = None


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--titans-upstream", type=Path, required=True)
    a = p.parse_args()
    if subprocess.check_output(["git","-C",str(a.titans_upstream),"rev-parse","HEAD"], text=True).strip()!=UP:
        raise ValueError("Incorrect Titans checkout")
    sys.path.insert(0, str(a.titans_upstream.resolve()))
    a.output.mkdir(parents=True)
    def save(name, value):
        path = a.output/name
        path.write_text(json.dumps(value, indent=2, allow_nan=False))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REV)
    train = episodes(tok,128,42)
    valid = episodes(tok,32,20261119,{k for e in train for k in e["keys"]})
    donors = [[(i+j)%len(valid) for j in (1,7,13)] for i in range(len(valid))]
    digest = save("synthetic_manifest.json",dict(train=train,validation=valid))
    save("donor_assignments.json",donors)
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    save("config.json",dict(source_commit=source,model_revision=REV,titans_revision=UP,
         arms=ARMS,steps=128,seed=42,evaluate_every=32,token_budget_per_arm=128*2048))
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,dtype=torch.float16,
                                                attn_implementation="eager").cuda().eval()
    block = model.model.layers[3]
    native = block.self_attn
    summaries, hashes = {}, {}
    with (a.output/"training.jsonl").open("w") as stream:
        for arm,(chunk,_) in ARMS.items():
            block.self_attn = native
            torch.manual_seed(42)
            wrapper = install(model,arm)
            named = {n:v for n,v in wrapper.named_parameters() if v.requires_grad}
            opt = torch.optim.AdamW(named.values(),lr=1e-4,weight_decay=0)
            sched = torch.optim.lr_scheduler.LambdaLR(opt,lambda i:min(1,(i+1)/10))
            for step, ep in enumerate(train,1):
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                row = update(model,wrapper,ep["segments"],opt,sched,chunk)
                peak = torch.cuda.max_memory_allocated()
                stream.write(json.dumps(dict(row,arm=arm,step=step,seconds=time.monotonic()-started,
                                             peak_cuda_bytes=peak),allow_nan=False)+"\n")
                stream.flush()
                print(arm,step,row["loss"],round(time.monotonic()-started,2),flush=True)
                if peak > 13*2**30:
                    raise RuntimeError("13 GiB PyTorch allocation target exceeded")
                if step % 32 == 0:
                    rows = evaluate(model,wrapper,valid,donors)
                    save(f"{arm}-validation-{step:03}.json",rows)
                    summaries[arm] = {c:{m:mean(r[c][m] for r in rows) for m in rows[0][c]}
                        for c in ("correct","donor_mean","w0","shuffled","zero_read","no_memory")}
            path = a.output/f"{arm}.pt"
            torch.save(dict(parameters={n:p.detach().cpu() for n,p in named.items()},
                            optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state(),arm=arm,step=128,seed=42,chunk=chunk,
                            retrieve_chunk=512,data_sha256=digest,model_revision=REV,tokenizer_revision=REV,
                            titans_revision=UP,source_commit=source),path)
            readback = torch.load(path,map_location="cpu",weights_only=True)
            for n,param in named.items():
                torch.testing.assert_close(readback["parameters"][n],param.detach().cpu(),rtol=0,atol=0)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            del readback, opt, sched, named, wrapper
    save("checkpoint_hashes.json",hashes)
    save("statistics.json",summaries)
    candidates = [arm for arm,s in summaries.items() if all(
        s["correct"]["nll"] < s[c]["nll"] and s["correct"]["candidate_correct"] > s[c]["candidate_correct"]
        for c in ("w0","donor_mean","shuffled"))]
    save("decision.json",dict(format=KIND,status="SCREENING_COMPLETE",scientific_status="SCREENING_ONLY",
         confirmation_candidates=candidates,test="SEALED",source_commit=source))
    (a.output/"RESULTS.md").write_text("# Stage 1 screening\n\nCandidates: "+str(candidates)+
        "\n\nDirectional validation evidence only. No confirmation or natural-language training.\n")


if __name__ == "__main__":
    main()
