"""Single launch: real-model preflight followed by 20 synthetic engineering steps."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import itertools
from adapter import MODEL, REVISION, TITANS, SEGMENT, install
from preflight import require, verify, verify_segment
from evaluation import evaluate, natural_regression
from protocol import mixed


def build(tok, seed, lo, labels, count):
    rng = random.Random(seed)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    filler = enc(" There was an unrelated observation.\n") * SEGMENT
    keys = rng.sample(range(lo, lo+10000), count*2)
    rows = []
    for i in range(count):
        a,b = rng.sample(labels,2)
        k,j = keys[2*i:2*i+2]
        prompt = f"What value belongs to code {k}?\nCode {k}:"
        target = enc(" "+a)
        require(len(target)==1 and enc(prompt+" "+a)==enc(prompt)+target,
                "Answer token boundary mismatch")
        parts = [enc(f"Code {k}: {a}\nCode {j}: {b}\n"),
                 enc("Nothing else was recorded."), enc(prompt)+target]
        require(all(len(x)<=SEGMENT for x in parts), "Segment overflow")
        parts = [filler[:SEGMENT-len(x)]+x for x in parts]
        require(all(x not in parts[1]+parts[2][:-1] for x in enc(" "+a)+enc(" "+b)),
                "Answer leaked into distractor or query")
        rows.append(dict(keys=[k,j], values=[a,b], tokens=parts))
    return rows


@torch.no_grad()
def recurrence(m, device):
    from titans_pytorch.neural_memory import softclamp_grad_norm
    g = torch.Generator(device=device).manual_seed(441)
    def sample(n):
        return torch.randn(1,n,1,128,device=device,generator=g)
    q,k,v = sample(16),sample(8),sample(8)
    ctx = torch.zeros(1,1,1024,device=device)
    _,_,before = m(q,k,v,context=ctx)
    coefficients = {}
    handles = [mod.register_forward_hook(lambda _,a,y,name=name: coefficients.update({name:y.sigmoid()}))
               for name,mod in (("beta",m.to_momentum),("decay",m.to_decay_factor))]
    original = m.memory_grads
    gradients = []
    def capture(*args):
        result = original(*args)
        gradients.extend(result[0])
        return result
    m.memory_grads = capture
    try:
        _,_,after = m(q,k,-v,state=before,context=ctx,seq_offset=1)
    finally:
        del m.memory_grads
        for handle in handles:
            handle.remove()
    beta = coefficients["beta"].reshape(8,1,1)
    decay = coefficients["decay"].reshape(8,1,1)
    initial = m.init_weights(1)
    for w0,w,s,grad,observed_w,observed_s in zip(initial,before.states[0],before.states[1],
                                                gradients,after.states[0],after.states[1],strict=True):
        expected_s = beta*s[0] - softclamp_grad_norm(grad,m.max_grad_norm)
        expected_w = (1-decay)*w + decay*w0 + expected_s
        torch.testing.assert_close(observed_s[0],expected_s,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(observed_w,expected_w,atol=1e-6,rtol=1e-5)
    return "PASS"


@torch.no_grad()
def identity_probe(model, cell, rows, device):
    """Measure actual native K→V compatibility; do not select an arm."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    attention = cell.attention
    trial = deepcopy(cell.memory)
    trial.initialize_kv_memory(attention.k_proj.weight, attention.v_proj.weight, identity=True)
    weights = trial.init_weights(1)
    scores = []
    def capture(module, args, kwargs):
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        rope = kwargs["position_embeddings"]
        k = attention.k_norm(attention.k_proj(x).unflatten(-1, (8,128))).transpose(1,2)
        _, k = apply_rotary_pos_emb(k, k, *rope)
        k = k[0].float() * 128**-.25
        v = attention.v_proj(x).unflatten(-1,(8,128)).transpose(1,2)[0].float()
        pred = trial.memory_forward(weights, k)
        scores.append(dict(mse=(pred-v).square().mean().item(), zero_mse=v.square().mean().item(),
            cosine=torch.nn.functional.cosine_similarity(pred,v,dim=-1).mean().item()))
    block = model.model.layers[22]
    block.self_attn = attention
    handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for row in rows:
            ids = torch.tensor([row["tokens"][0]],device=device)
            model.model(input_ids=ids,use_cache=False)
    finally:
        handle.remove()
        block.self_attn = cell
    require(len(scores)==len(rows), "Missing identity probe evidence")
    return scores


FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def fineweb(tok, count, *, offset=0):
    from datasets import load_dataset
    stream = load_dataset(FINEWEB, name="sample-10BT", revision=FINEWEB_REV,
                          split="train", streaming=True)
    rows = []
    for entry in itertools.islice(stream, offset, offset + max(200, count * 10)):
        ids = tok(entry["text"], add_special_tokens=False, truncation=True,
                  max_length=3*SEGMENT)["input_ids"]
        if len(ids) == 3*SEGMENT and not any(row["tokens"] == [
            ids[i:i+SEGMENT] for i in range(0, len(ids), SEGMENT)] for row in rows):
            rows.append(dict(source="fineweb_edu", tokens=[
                ids[i:i+SEGMENT] for i in range(0,len(ids),SEGMENT)]))
            if len(rows) == count:
                return rows
    raise RuntimeError("Insufficient distinct FineWeb-Edu windows")


def train_step(model, cell, row, optimizer, device):
    from transformers import DynamicCache
    cell.reset_episode()
    optimizer.zero_grad(set_to_none=True)
    early = None
    natural = row.get("source") == "fineweb_edu"
    losses = []
    for segment, ids in enumerate(row["tokens"]):
        cache = DynamicCache()
        for offset, token in enumerate(ids):
            # The last token is supervision, never written before prediction.
            if segment == 2 and offset == 127:
                break
            x = torch.tensor([[token]],device=device)
            h = model.model(input_ids=x,past_key_values=cache,use_cache=True,
                            position_ids=torch.tensor([[offset]],device=device)).last_hidden_state
            if natural:
                target = ids[offset+1] if offset+1 < SEGMENT else row["tokens"][segment+1][0]
                next_id = torch.tensor([target],device=device)
                losses.append(torch.nn.functional.cross_entropy(model.lm_head(h[:, -1]).float(), next_id))
            if segment == offset == 0:
                early = cell.state.states[0]
    require(cell.state.seq_index == 383, "Expected 383 causal writes")
    loss = torch.nn.functional.cross_entropy(model.lm_head(h[:, -1]).float(),
                torch.tensor([row["tokens"][-1][-1]],device=device))
    if natural:
        loss = torch.stack(losses).mean()
    require(torch.isfinite(loss), "Nonfinite loss")
    grads = torch.autograd.grad(loss, early, retain_graph=True)
    require(all(torch.isfinite(g).all() for g in grads) and any(g.count_nonzero() for g in grads),
            "Invalid early recurrent gradient")
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grad = torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
    require(grad > 0 and all(p.grad is None for p in cell.attention.parameters()), "Gradient check failed")
    optimizer.step()
    cell.reset_episode()
    return dict(loss=loss.item(), grad_norm=grad.item(),
                early_grad_norms=[g.norm().item() for g in grads])


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("upstream",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    require(subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()==TITANS,
            "Wrong Titans revision")
    sys.path.insert(0,str(a.upstream.resolve()))
    torch.manual_seed(42)
    device="cuda"
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16,
        attn_implementation="eager").to(device).eval()
    cell=install(model,"G1-A")
    preflight=verify(model,cell,device)
    preflight["full_segment_boundary"]=verify_segment(model,cell,device)
    preflight["momentum_and_decay"]=recurrence(cell.memory,device)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    train=build(tok,42,10000,"ABCD",10)
    validation=build(tok,43,30000,"EFGH",8)
    # Test domain is reserved only; no test examples are generated.
    require(not ({k for r in train for k in r["keys"]} & {k for r in validation for k in r["keys"]}),
            "Entity overlap")
    require(not ({v for r in train for v in r["values"]} & {v for r in validation for v in r["values"]}),
            "Value overlap")
    identity_rows = identity_probe(model,cell,validation,device)
    compatible = all(r["mse"] < r["zero_mse"] and r["cosine"] > 0 for r in identity_rows)
    preflight["identity"] = dict(status="COMPATIBLE_ON_PROBE" if compatible else "NOT_APPLICABLE",
        rule="Every validation segment must beat zero MSE and have positive mean cosine", rows=identity_rows)
    web = fineweb(tok,10)
    train = mixed(train, web)
    raw=json.dumps(dict(train=train,validation=validation, fineweb_revision=FINEWEB_REV,
                       test_domain=dict(keys=[50000,59999],values="IJKL")))
    (a.output/"data_manifest.json").write_text(raw)
    (a.output/"preflight.json").write_text(json.dumps(preflight,indent=2))
    memory, fusion = [], []
    for name,param in cell.memory.named_parameters():
        if param.requires_grad:
            (fusion if name.startswith("history_mix") else memory).append(param)
    def make_optimizer():
        return torch.optim.AdamW([dict(params=memory,lr=5e-4),dict(params=fusion,lr=5e-5)],
                                 betas=(.9,.95),weight_decay=.01)
    opt=make_optimizer()
    params={n:p for n,p in cell.memory.named_parameters() if p.requires_grad}
    resume_path=a.output/"resume-step19.pt"
    def checkpoint(path, step):
        torch.save(dict(parameters={n:p.detach().cpu() for n,p in params.items()},
                        optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),step=step),path)
    with (a.output/"training.jsonl").open("w") as stream:
        for step,row in enumerate(train,1):
            started=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            result=train_step(model,cell,row,opt,device)
            result.update(step=step,seconds=time.monotonic()-started,
                          peak_bytes=torch.cuda.max_memory_allocated())
            stream.write(json.dumps(result,allow_nan=False)+"\n")
            stream.flush()
            print(result,flush=True)
            if step == 19:
                checkpoint(resume_path,19)
            if result["peak_bytes"] > 13*2**30:
                raise RuntimeError("Smoke peak allocation exceeded 13 GiB")
    path=a.output/"smoke.pt"
    checkpoint(path,20)
    expected=torch.load(path,map_location="cpu",weights_only=True)
    restored=torch.load(resume_path,map_location="cpu",weights_only=True)
    with torch.no_grad():
        for n,p in params.items():
            p.copy_(restored["parameters"][n])
    opt=make_optimizer()
    opt.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["cpu_rng"])
    torch.cuda.set_rng_state(restored["cuda_rng"])
    replay=train_step(model,cell,train[restored["step"]],opt,device)
    torch.testing.assert_close(replay["loss"],result["loss"],atol=1e-6,rtol=1e-6)
    for n,p in params.items():
        torch.testing.assert_close(p.detach().cpu(),expected["parameters"][n],atol=1e-7,rtol=1e-6)
    for key,state in opt.state_dict()["state"].items():
        for name,tensor in state.items():
            torch.testing.assert_close(tensor.cpu(),expected["optimizer"]["state"][key][name],
                                       atol=1e-7,rtol=1e-6)
    (a.output/"resume.json").write_text(json.dumps(dict(status="PASS", restored_step=19,
        replayed_step=20, continuous_loss=result["loss"], resumed_loss=replay["loss"])))
    regression = natural_regression(model,cell,fineweb(tok,1,offset=3000),device)
    paired = evaluate(model, cell, validation[:2], device)
    (a.output/"evaluation.json").write_text(json.dumps(dict(synthetic=paired,fineweb=regression),indent=2,allow_nan=False))
    (a.output/"decision.json").write_text(json.dumps(dict(format="minicells.titans-gate0.v1",
        status="SMOKE_ONLY", scientific_status="NOT_EVALUATED", formal_ready=True, arm="G1-A",
        identity=preflight["identity"]["status"], resume_equivalence="PASS",
        fineweb_revision=FINEWEB_REV, synthetic_steps=10, fineweb_steps=10,
        steps=20,model_revision=REVISION,titans_revision=TITANS,gpu=torch.cuda.get_device_name(),
        data_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":
    main()
