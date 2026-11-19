"""Gate 0 checks on the actual pinned Qwen and Titans; no optimizer steps."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from adapter import MODEL, REVISION, TITANS, install


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(model, cell, device):
    memory = cell.memory
    require(not any(p.requires_grad for p in model.parameters()
                    if id(p) not in {id(x) for x in memory.parameters()}), "Backbone is trainable")
    g = torch.Generator(device=device).manual_seed(42)
    def sample(heads):
        return torch.randn((1, heads, 1, 128), generator=g, device=device)
    q, k, v = sample(16), sample(8), sample(8)
    context = torch.zeros(1, 1, 1024, device=device)
    with torch.no_grad():
        first, gate, a = memory(q,k,v,context=context)
        _, _, b = memory(q,k,-v,context=context)
        require(a.seq_index == b.seq_index == 1, "First token update count")
        require(not first.count_nonzero() and not gate.count_nonzero(), "First token exposed memory")
        second, _, c = memory(q,k,v,state=a,context=context,seq_offset=1)
        alternative, _, d = memory(q,k,-v,state=a,context=context,seq_offset=1)
        prior_swap, _, _ = memory(q,k,v,state=b,context=context,seq_offset=1)
        torch.testing.assert_close(second, alternative, rtol=0, atol=0)
        require(not torch.equal(second, prior_swap), "Previous value had no effect on read")
        require(c.seq_index == d.seq_index == 2, "Second token update count")
        require(any(not torch.equal(x,y) for x,y in zip(c.updates,d.updates)), "No fast update")
        # Compare GQA against an independent per-head application of the MLP.
        flattened = (q * 128**-.25).transpose(1,2).flatten(2)
        direct, _ = memory.read_only((flattened,None,None),a)
        weights = [w[:, -1].repeat_interleave(2, dim=0) for w in a.updates]
        manual = memory.memory_forward(weights, flattened.reshape(16,1,128)).reshape(1,1,-1)
        torch.testing.assert_close(direct, manual, rtol=1e-5, atol=1e-6)

    cell.reset_episode()
    # Two short segments: actual native cache grows within a segment and
    # starts empty at the boundary; the Cell and its graph survive.
    from transformers import DynamicCache
    earliest = None
    for index in range(4):
        if index % 2 == 0:
            cache = DynamicCache()
            require(cache.get_seq_length() == 0, "Cache reset failed")
        ids = torch.tensor([[100+index]], device=device)
        output = model(input_ids=ids, past_key_values=cache, use_cache=True,
                       position_ids=torch.tensor([[index%2]], device=device))
        require(cache.get_seq_length() == index%2+1, "Cache did not advance")
        require(cell.position == cell.state.seq_index == index+1, "Cell did not persist")
        if index == 0:
            earliest = cell.state.updates
    loss = output.logits.float().square().mean()
    slow = tuple(memory.memory_model_parameters)
    gradients = torch.autograd.grad(loss, (*earliest,*slow), allow_unused=True)
    def check_slice(values):
        require(all(x is not None and torch.isfinite(x).all() for x in values), "Missing/nonfinite gradient")
        require(any(x.count_nonzero() for x in values), "Zero gradient")
        return [x.norm().item() for x in values]
    result = dict(early_fast_grad_norms=check_slice(gradients[:len(earliest)]),
                  slow_grad_norms=check_slice(gradients[len(earliest):]))
    cell.reset_episode()
    require(cell.state is None and cell.position == 0, "Episode reset failed")
    return result


def main():
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("upstream", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    revision = subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()
    require(revision == TITANS, "Wrong Titans revision")
    sys.path.insert(0, str(a.upstream.resolve()))
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,
        dtype=torch.float32 if a.device=="cpu" else torch.float16,
        attn_implementation="eager").to(a.device).eval()
    evidence = verify(model,install(model,"G1-A"),a.device)
    a.output.mkdir(parents=True)
    (a.output / "decision.json").write_text(json.dumps(dict(format="minicells.titans-gate0.v1", status="PARTIAL_GATE0", model_revision=REVISION,
        titans_revision=revision, device=a.device, **evidence), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
