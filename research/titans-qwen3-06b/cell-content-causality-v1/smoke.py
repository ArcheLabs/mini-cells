"""Exercise counterfactual evaluation and the outer gradient on real Qwen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import episodes, manifest
from diagnostics import compare
from evaluate import build_history, encode_history, evaluate_episode
from summarize import summarize

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
    def encode_training(episode):
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        item = episode["queries"][0]
        history = encode_history(tokenizer, episode["facts"], episode["distractors"])
        prompt = encode(f"Question: {item['prompt']}\nAnswer:")
        answer = encode(" " + item["answer"])
        segments = [(ids, [-100] * len(ids)) for ids in history]
        segments.append((prompt + answer, [-100] * len(prompt) + answer))
        if any(len(ids) > 128 for ids, _ in segments):
            raise ValueError(f"Training segment lengths: {[len(ids) for ids, _ in segments]}")
        return segments
    training = [encode_training(episode) for episode in episodes("train")]
    print({"stage": "TOKEN_BUDGET_VERIFIED", "episodes": len(training),
           "max_segment_tokens": max(len(ids) for row in training for ids, _ in row)},
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="eager",
    ).eval().cuda()
    wrapper = install(model, 22, build_memory(args.upstream, chunk_size=4, accelerated=False))
    params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, betas=(.9, .95), weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / 3 if step < 3 else
        (1 + math.cos(math.pi * (step - 3) / 97)) / 2,
    )
    output = args.output
    (output / "metadata.json").write_text(json.dumps({
        "model_revision": REVISION, "titans_revision": "c60b4efaa3a3c6df4deebd57cbccf75947b87cad",
        "seed": 42, "layer": 22, "chunk_size": 4, "accelerated_scan": False,
        "detach_inner_grads": True, "detach_mem_state": False, "segment_limit": 128,
        "outer_updates": 100, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "splits": [manifest(split) for split in ("train", "validation", "test")],
    }, indent=2))
    started_run = time.monotonic()

    def record(name, item):
        with (output / name).open("a") as stream:
            stream.write(json.dumps(item) + "\n")

    def assess(split, step, one=False):
        rows = list(episodes(split))
        for index, episode in enumerate(rows[:1] if one else rows):
            if one:
                episode["queries"] = episode["queries"][:1]
            donor = rows[(index + 1) % len(rows)]
            scores = evaluate_episode(model, wrapper, tokenizer, episode, donor)
            record(f"{split}.jsonl", {"step": step, "episode": index, "scores": scores})
            if index == 0:
                print({"stage": split, "step": step}, flush=True)

    def checkpoint(step):
        example = next(episodes("validation"))
        cell, position = build_history(model, wrapper, encode_history(
            tokenizer, example["facts"], example["distractors"]))
        path = output / f"step_{step:03}.pt"
        # Save the entire dynamic state as a plain mapping for portable loading.
        torch.save({
            "step": step, "parameters": {name: p.detach().cpu() for name, p in
                                         wrapper.named_parameters() if p.requires_grad},
            "cell": cell._asdict(), "cell_position": position,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
        }, path)
        record("checkpoints.jsonl", {"step": step, "path": path.name,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    checkpoint(0)
    assess("validation", 0)
    for step, segments in enumerate(training, 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        logits, targets = forward_episode(model, segments, wrapper=wrapper, query_write=False)
        loss = answer_loss(logits, targets)
        loss.backward()
        bank = wrapper.memory.memory_model_parameters
        if any(p.grad is None for p in bank):
            raise RuntimeError("Active Titans parameters lack gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        row = {"step": step, "loss": loss.item(), "grad_norm": norm.item(),
               "seconds": time.monotonic() - started,
               "peak_bytes": torch.cuda.max_memory_allocated()}
        record("training.jsonl", row)
        print(row, flush=True)
        wrapper.reset_episode()
        del logits, targets, loss
        if step == 5:
            selected = next(episodes("validation"))
            item = selected["queries"][0]
            correct = build_history(model, wrapper, encode_history(
                tokenizer, selected["facts"], selected["distractors"]))
            alternative = build_history(model, wrapper, encode_history(
                tokenizer, item["counterfactual_facts"], selected["distractors"]))
            values = compare(model, wrapper, correct, alternative,
                             tokenizer.encode(f"Question: {item['prompt']}\nAnswer:",
                                              add_special_tokens=False), 22)
            record("diagnostics.jsonl", values)
            del correct, alternative
            assess("validation", step, one=True)
            (output / "engineering.json").write_text(json.dumps({"status": "PASS", "step": step}))
        if step in (25, 50, 100):
            checkpoint(step)
            assess("validation", step)

    assess("test", 100)
    summarize(output)
    (output / "duration.json").write_text(json.dumps({"seconds": time.monotonic() - started_run}))
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
