"""Run the paired 100-update screen and retain checkpoint-level diagnostics."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from episodes import digest, encode, generate
from evaluate import evaluate
from execution import answer_loss, forward_episode
from integration import LAYERS, MODEL, REVISION, build_memory, install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    train = generate("train", 100)
    validation = generate("validation", 8)
    training = [encode(episode, tokenizer) for episode in train]
    for episode in validation:
        encode(episode, tokenizer)
        encode(episode, tokenizer, donor=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).eval().cuda()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "data.json").write_text(json.dumps({
        "model_revision": REVISION, "train": digest(train), "validation": digest(validation),
    }))
    for layer in LAYERS:
        torch.manual_seed(42)
        wrapper = install(model, layer, build_memory(args.upstream))
        named = {name: p for name, p in wrapper.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(named.values(), lr=.001, weight_decay=.01,
                                      betas=(.9, .95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (step + 1) / 3 if step < 3 else
            (1 + math.cos(math.pi * (step - 3) / 97)) / 2,
        )
        folder = args.output / f"layer_{layer}"
        folder.mkdir()
        for step in range(101):
            if step in (0, 25, 50, 100):
                torch.save({"step": step, "parameters": named,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state()},
                           folder / f"step_{step:03}.pt")
                report = [evaluate(model, wrapper, layer, episode, tokenizer)
                          for episode in validation]
                (folder / f"validation_{step:03}.json").write_text(json.dumps(report))
            if step == 100:
                break
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits, targets = forward_episode(model, training[step], wrapper=wrapper)
            loss = answer_loss(logits, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(named.values(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            wrapper.reset_episode()
            with (folder / "metrics.jsonl").open("a") as log:
                entry = {"update": step + 1, "loss": loss.item(), "grad_norm": norm.item(),
                         "seconds": time.monotonic() - started, "targets": targets.numel()}
                log.write(json.dumps(entry) + "\n")
                print(entry, flush=True)
        model.model.layers[layer].self_attn = wrapper.attention


if __name__ == "__main__":
    main()
