"""Execute the frozen P1.1 protocol with the original P1 episode loss."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from corpus import CORPUS, SOURCE, select
from decision import classify
from evaluation import evaluate_p1_1
from protocol import SEEDS, donor_assignments


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train import MODEL, REVISION, UPSTREAM
    from p1 import episode_loss, install

    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=args.upstream, text=True).strip()
    if actual != UPSTREAM:
        raise ValueError("Unexpected Titans revision")
    sys.path.insert(0, str(args.upstream.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    def rows(split):
        return load_dataset(SOURCE, "wikitext-103-raw-v1", revision=CORPUS,
                            split=split, streaming=True)

    documents, manifest = select(rows("train"), rows("validation"),
                                 lambda text: tokenizer.encode(text, add_special_tokens=False),
                                 holdout_size=8 if args.smoke else 64)
    manifest["model_revision"] = REVISION
    save("data_manifest.json", manifest)
    holdout = documents["primary_holdout"]
    assignments = donor_assignments(len(holdout), 1 if args.smoke else 3)
    save("donor_assignments.json", assignments)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16, attn_implementation="eager",
    ).eval().cuda()
    results, baseline = {}, None
    seeds, steps = (SEEDS[:1], 20) if args.smoke else (SEEDS, 200)
    save("config.json", {"seeds": seeds, "steps": steps, "layer": 3, "chunk": 512,
                          "lr": 1e-4, "warmup_steps": 10, "titans_revision": UPSTREAM})
    for seed in seeds:
        torch.manual_seed(seed)
        wrapper = install(model)
        named = {n: p for n, p in wrapper.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(named.values(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / 10))
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        hashes = {}
        for step in range(steps + 1):
            if step in (0, 20, 200):
                path = folder / f"step_{step:03}.pt"
                torch.save({"step": step, "parameters": {n: p.detach().cpu() for n, p in named.items()},
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, path)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                for name, parameter in named.items():
                    torch.testing.assert_close(checkpoint["parameters"][name], parameter.detach().cpu(),
                                               rtol=0, atol=0)
                del checkpoint
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                evaluated = evaluate_p1_1(model, wrapper, holdout, assignments, baseline=baseline)
                if baseline is None:
                    baseline = [{key: row[key] for key in ("full_attention", "chunked_attention")}
                                for row in evaluated["documents"]]
                    save("baseline.json", baseline)
                save(f"seed_{seed}/validation_step_{step:03}.json", evaluated)
            if step == steps:
                results[seed] = evaluated["documents"]
                break
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            audit = {} if step == 0 else None
            token_ids = torch.tensor([documents["train"][step]], device="cuda")
            loss = episode_loss(model, wrapper, token_ids, audit=audit).mean()
            loss.backward()
            gate_grad = wrapper.memory.history_mix.grad
            if gate_grad is None or not gate_grad.count_nonzero():
                raise RuntimeError("Gate has no training gradient")
            norm = torch.nn.utils.clip_grad_norm_(named.values(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            row = {"step": step + 1, "loss": loss.item(), "gradient_norm": norm.item(),
                   "seconds": time.monotonic() - started, "peak_bytes": torch.cuda.max_memory_allocated(),
                   "meta_gradient": audit}
            print({"seed": seed, **row}, flush=True)
            with (folder / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        save(f"seed_{seed}/checkpoint_hashes.json", hashes)
        model.model.layers[3].self_attn = wrapper.attention
    decision = classify(results, engineering_pass=True, smoke=args.smoke)
    save("statistics.json", decision.get("statistics", {}))
    save("decision.json", {"format": "minicells.titans-p1-1.v1", **decision})
    (args.output / "RESULTS.md").write_text(
        "# P1.1 实验结果\n\n"
        f"运行模式：{'工程 Smoke' if args.smoke else '三 Seed 正式验证'}。\n\n"
        f"判定：{decision['status']}。\n\n"
        "逐文档 NLL、Gate 和读取诊断见各 Seed 的 validation JSON。"
        "配对置信区间和科学门槛见 statistics.json 和 decision.json。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
