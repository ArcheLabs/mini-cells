"""Exercise counterfactual evaluation and the outer gradient on real Qwen."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import episodes, manifest
from diagnostics import compare
from evaluate import build_history, encode_history, evaluate_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer-location-v1"))
from execution import answer_loss, forward_episode
from integration import MODEL, REVISION, build_memory, install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="eager",
    ).eval().cuda()
    wrapper = install(model, 22, build_memory(args.upstream, chunk_size=4, accelerated=False))
    params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, betas=(.9, .95), weight_decay=.01)
    for step, episode in enumerate(episodes("train")):
        if step == 5:
            break
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        item = episode["queries"][0]
        history = encode_history(tokenizer, episode["facts"], episode["distractors"])
        prompt = encode(f"Question: {item['prompt']}\nAnswer:")
        answer = encode(" " + item["answer"])
        segments = [(ids, [-100] * len(ids)) for ids in history]
        segments.append((prompt + answer, [-100] * len(prompt) + answer))
        if any(len(ids) > 128 for ids, _ in segments):
            raise ValueError("Smoke episode exceeds the segment limit")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        logits, targets = forward_episode(model, segments, wrapper=wrapper, query_write=False)
        loss = answer_loss(logits, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        print({"step": step + 1, "loss": loss.item(), "grad_norm": norm.item(),
               "seconds": time.monotonic() - started,
               "peak_bytes": torch.cuda.max_memory_allocated()}, flush=True)
        wrapper.reset_episode()
    selected, donor, *_ = episodes("validation")
    item = selected["queries"][0]
    correct = build_history(model, wrapper, encode_history(
        tokenizer, selected["facts"], selected["distractors"]))
    alternative = build_history(model, wrapper, encode_history(
        tokenizer, item["counterfactual_facts"], selected["distractors"]))
    selected["queries"] = [item]
    report = {
        "status": "ENGINEERING_SMOKE_COMPLETE",
        "diagnostics": compare(model, wrapper, correct, alternative,
                               encode(f"Question: {item['prompt']}\nAnswer:"), 22),
        "evaluation": evaluate_episode(model, wrapper, tokenizer, selected, donor),
        "manifests": [manifest(split) for split in ("train", "validation", "test")],
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2))
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
