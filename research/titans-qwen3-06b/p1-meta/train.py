"""Train P1 on revision-pinned, document-isolated natural language."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation import evaluate
from p1 import episode_loss, install

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
UPSTREAM = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path, help="UTF-8 JSONL with one text per document")
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, choices=(20, 200), default=20)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()
    import sys
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=args.upstream, text=True).strip()
    if revision != UPSTREAM:
        raise ValueError("Incorrect Titans revision")
    sys.path.insert(0, str(args.upstream.resolve()))
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    raw = args.documents.read_bytes()
    samples, seen = [], set()
    for line in raw.splitlines():
        text = json.loads(line)["text"]
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer.encode(text, add_special_tokens=False)[:2048]
        if len(ids) == 2048:
            samples.append(ids)
        if len(samples) == 208:
            break
    if len(samples) != 208 or len({tuple(ids) for ids in samples}) != 208:
        raise ValueError("Need 208 distinct complete document prefixes")
    args.output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()
    wrapper = install(model)
    named = {n: p for n, p in wrapper.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(named.values(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / 10))
    config = {"dataset_source": args.dataset_source, "dataset_revision": args.dataset_revision,
              "data_sha256": hashlib.sha256(raw).hexdigest(), "model_revision": REVISION,
              "titans_revision": revision, "seed": 42, "layer": 3, "chunk": 512,
              "trainable": list(named), "gpu": torch.cuda.get_device_name()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    for step, ids in enumerate(samples[:args.steps], 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses = episode_loss(model, wrapper, torch.tensor([ids], device="cuda"))
        losses.mean().backward()
        groups = (wrapper.memory.memory_model_parameters, wrapper.memory.history_mix_proj.parameters())
        for group in groups:
            if not any(p.grad is not None and p.grad.count_nonzero() for p in group):
                raise RuntimeError("Missing memory or fusion gradient")
        norm = torch.nn.utils.clip_grad_norm_(named.values(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        row = {"step": step, "chunk_nll": losses.detach().tolist(), "gradient_norm": norm.item(),
               "peak_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - started}
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(row, flush=True)
    torch.save({"parameters": {n: p.detach().cpu() for n, p in named.items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "global_step": args.steps, "configuration": config,
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()},
               args.output / f"step_{args.steps:03}.pt")

    report = evaluate(model, wrapper, samples[200:])
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
