"""Frozen single-layer random binding diagnostic; no outer training."""
import argparse
import os
import random
from statistics import mean
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn.functional import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1-2"))
from replay import MODEL, REV, UP, ART, SEEDS, capture, same_tree, verify, restore, save, sha, finite

KIND = "minicells.titans-p1-3-diagnostic.v1"
WORDS = " apple pear lemon peach plum mango grape melon lime cherry banana orange".split()


@torch.no_grad()
def evaluate(model, wrapper, tokenizer, cases):
    rng = random.Random(20261119)
    device = next(model.parameters()).device

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    candidates = [encode(" " + word) for word in WORDS]
    if any(len(x) != 1 for x in candidates) or len({x[0] for x in candidates}) != len(WORDS):
        raise ValueError("Candidate vocabulary must contain distinct single tokens")

    def forward(ids, start):
        tokens = torch.tensor([ids], device=device)
        return model(input_ids=tokens, position_ids=torch.arange(start, start + len(ids),
                     device=device)[None], use_cache=False).logits

    rows = []
    for episode in range(cases):
        keys = rng.sample(range(10000, 99999), 4)
        words = rng.sample(WORDS, 4)
        ids = [candidates[WORDS.index(w)][0] for w in words]
        shifted = words[1:] + words[:1]

        def write(values):
            tokens, locations = [], []
            for key, word in zip(keys, values):
                prefix = f"Code {key}:"
                whole = encode(prefix + " " + word)
                if whole != encode(prefix) + candidates[WORDS.index(word)]:
                    raise ValueError("Value token boundary changed")
                tokens.extend(whole)
                locations.append(len(tokens) - 1)
                tokens.extend(encode("\n"))
            filler = encode(" Information follows.\n") * 512
            pad = 512 - len(tokens)
            wrapper.state = None
            forward(filler[:pad] + tokens, 0)
            targets = wrapper.pending[2][0, [pad + p for p in locations]].reshape(4, 8, 128)
            targets = targets[:, :, None, :].expand(-1, -1, 4, -1).reshape(4, 2048)
            wrapper.commit()
            return wrapper.state, targets

        correct, prototypes = write(words)
        donor, _ = write(shifted)
        for i, key in enumerate(keys):
            query = encode(f"What value is assigned to code {key}?\nCode {key}:")
            qkv = None
            for condition, state in (("correct", correct), ("swapped", donor), ("empty", None)):
                wrapper.state = state
                with capture(wrapper.memory, write=False) as observed:
                    logits = finite(forward(query, 512)[0, -1].float())
                if qkv is None:
                    qkv = wrapper.pending
                else:
                    same_tree(qkv, wrapper.pending)
                prediction = logits[ids]
                # Empty bypasses memory. A zero read gives tied candidates; score ties fractionally.
                read = observed["read"][0, -1] if state is not None else torch.zeros_like(prototypes[0])
                scores = finite(normalize(prototypes.float()) @ normalize(read.float(), dim=0))
                winners = scores == scores.max()
                rows.append(dict(episode=episode, keys=keys, values=words, query_index=i,
                    condition=condition, cosine_scores=scores.tolist(),
                    retrieval_credit=float(winners[i]) / int(winners.sum()),
                    lm_candidate_correct=int(prediction.argmax()) == i,
                    lm_log_probability=logits.log_softmax(0)[ids[i]].item(),
                    swapped_retrieval_credit=float(winners[(i+1)%4]) / int(winners.sum())))
        print(f"Binding episodes {episode+1}/{cases}", flush=True)
    wrapper.state = wrapper.pending = None
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from p1 import install
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "artifact", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    git = lambda p: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p, text=True).strip()
    if git(args.upstream) != UP:
        raise ValueError("Wrong Titans revision")
    verify(args.artifact)
    sys.path.insert(0, str(args.upstream.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float16,
                                                attn_implementation="eager").eval().cuda()
    block = model.model.layers[3]
    native = block.self_attn
    summary = {}
    for seed in SEEDS[:1] if args.smoke else SEEDS:
        block.self_attn = native
        torch.manual_seed(seed)
        wrapper = install(model).eval()
        for step in (200,) if args.smoke else (0, 200):
            checkpoint = args.artifact / f"seed_{seed}/step_{step:03}.pt"
            restored = restore(wrapper, checkpoint, step)
            rows = evaluate(model, wrapper, tokenizer, 1 if args.smoke else 16)
            key = f"seed_{seed}_step_{step:03}"
            save(args.output / f"{key}.json", rows)
            summary[key] = dict(conditions={c: {k: mean(r[k] for r in rows if r["condition"] == c)
                for k in ("retrieval_credit", "swapped_retrieval_credit", "lm_candidate_correct",
                          "lm_log_probability")} for c in ("correct", "swapped", "empty")},
                checkpoint_sha256=sha(checkpoint), restored=restored)
    save(args.output / "summary.json", summary)
    lines = ["# Random binding diagnostic", "",
             "| Checkpoint | Condition | Direct probe Recall@1 | LM candidate accuracy | Mean target log P |",
             "|---|---|---:|---:|---:|"]
    for key, entry in summary.items():
        for condition, m in entry["conditions"].items():
            lines.append(f"| {key} | {condition} | {m['retrieval_credit']:.3f} | "
                         f"{m['lm_candidate_correct']:.3f} | {m['lm_log_probability']:.5f} |")
    lines += ["", "Direct probe: cosine against contextualized write V, repeated across the four Q heads per KV group.",
              "This is a geometry probe; low scores alone cannot rule out information in another representation.",
              "Empty has four tied retrieval candidates (0.25 credit). No outer training."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    save(args.output / "decision.json", dict(format=KIND, status="SMOKE_ONLY" if args.smoke else "DIAGNOSTIC_ONLY",
         artifact_commit=ART, source_commit=git(HERE), model_revision=REV, upstream_revision=UP,
         outer_updates=0, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         kaggle_script_version_id=os.getenv("KAGGLE_SCRIPT_VERSION_ID")))


if __name__ == "__main__":
    main()
