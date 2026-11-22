"""Teacher-forced recall on the frozen MoM-Native Cells acceptance facts."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from acceptance import A, B, QUESTION
from backend import Backbone
from cells import Cells

NAMES = ("Wangari Maathai", "Jane Goodall", "Rachel Carson", "Aung San Suu Kyi")
YEARS = ("2004", "2002", "2005", "1997")
CASES = (
    ("founder", "Who founded the Green Belt Movement?",
     "The Green Belt Movement was founded by", NAMES, ("A",)),
    ("prize_year", "In what year did Wangari Maathai receive the Nobel Peace Prize?",
     "Wangari Maathai received the Nobel Peace Prize in", YEARS, ("B",)),
    ("joint", QUESTION,
     "The founder of the Green Belt Movement and the year she received the Nobel Peace Prize were",
     ("Wangari Maathai, 2004", "Wangari Maathai, 2002", "Jane Goodall, 2004",
      "Rachel Carson, 2005"), ("A", "B")),
)


def clone_cache(backbone, cache, reset_conv=False, reset_position=False):
    layers = copy.deepcopy(list(cache.to_legacy_cache()))
    if reset_conv:
        for layer in layers:
            layer["conv_state"] = tuple([None, None] for _ in range(3))
    return backbone.cache_type.from_legacy_cache(
        layers, seen_tokens=0 if reset_position else cache.get_seq_length())


@torch.inference_mode()
def score(backbone, prompt, answers, cache_factory):
    def encode(text):
        return backbone.tokenizer.encode(text, add_special_tokens=False)

    context = encode(prompt)
    rows = []
    for answer in answers:
        target = encode(answer)
        if not context or not target:
            raise ValueError("Prompt and candidate must tokenize to nonempty sequences")
        caches = cache_factory()
        # Same fresh branch for every teacher-forced candidate.
        hidden = backbone.step(context + target[:-1], caches, [1 / len(caches)] * len(caches))
        logits = backbone.model.lm_head(hidden[:, len(context)-1:]).float()
        logp = logits.log_softmax(-1)[0, range(len(target)), target].tolist()
        if not all(math.isfinite(value) for value in logp):
            raise FloatingPointError("Nonfinite answer scores")
        total = sum(logp)
        rows.append(dict(answer=answer, token_ids=target, token_logp=logp,
                         total_logp=total, mean_logp=total / len(target)))
    gold = rows[0]
    return dict(candidates=rows, gold_sequence_probability=math.exp(gold["total_logp"]),
                prompt_token_ids=context,
                **{key: dict(gold_rank=1 + sum(row[key] > gold[key] for row in rows[1:]),
                             gold_tie=any(row[key] == gold[key] for row in rows[1:]),
                             margin=gold[key] - max(row[key] for row in rows[1:]))
                   for key in ("total_logp", "mean_logp")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "snapshot", "tokenizer", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--restore", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fresh-process", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if not args.restore:
        out.mkdir(parents=True)
    def write(name, data):
        (out / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Match the accepted T4 path even when the diagnostic runs on a newer GPU.
    backbone = Backbone(args.source, args.snapshot, args.tokenizer, dtype="float16")
    cells = Cells(out / ("imported" if args.restore else "cells"), backbone)
    full, states, ids = {}, {}, {}
    for label, text in (("A", A), ("B", B)):
        bundle = out / (label + ".zip")
        if args.restore:
            meta = cells.import_bundle(bundle.read_bytes())
        else:
            meta = cells.create(label)
            tokens = backbone.tokenizer.encode(text, add_special_tokens=False)
            live = backbone.stream({})
            for start in range(0, len(tokens), 32):
                backbone.step(tokens[start:start+32], [live], [1.0])
            full[label] = live
            meta = cells.store({**meta, "revision": 1, "parent_revision": 0,
                                "tokens_written": len(tokens)}, backbone.persistent(live))
            bundle.write_bytes(cells.path(meta["cell_id"]).read_bytes())
        ids[label] = meta["cell_id"]
        states[label] = cells.read(ids[label])[1]

    rows = []
    for name, question, cloze, candidates, selected in CASES:
        for template, prompt in (("cloze", cloze), ("qa", f"Question: {question}\nAnswer:")):
            print(name, template, "restored" if args.restore else "local", flush=True)
            endings = [" " + c + ("." if template == "cloze" else "") for c in candidates]
            factories = {"recurrent_only": lambda: [backbone.stream(states[k]) for k in selected]}
            if not args.restore:
                factories.update(no_cell=lambda: [backbone.stream({})],
                    continuous=lambda: [clone_cache(backbone, full[k]) for k in selected])
                if args.ablation:
                    factories.update(reset_conv=lambda: [clone_cache(backbone, full[k], reset_conv=True)
                                                         for k in selected],
                        reset_position=lambda: [clone_cache(backbone, full[k], reset_position=True)
                                                for k in selected])
                if len(selected) == 2:
                    factories.update({k + "_only": lambda k=k: [backbone.stream(states[k])] for k in selected})
            if args.restore:
                factories["restored_recurrent_only"] = factories.pop("recurrent_only")
            arms = {k: score(backbone, prompt, endings, factory) for k, factory in factories.items()}
            reference = arms.get("no_cell")
            if reference:
                for value in arms.values():
                    value["delta_vs_no_cell"] = {
                        key: dict(gold=value["candidates"][0][key] - reference["candidates"][0][key],
                                  margin=value[key]["margin"] - reference[key]["margin"])
                        for key in ("total_logp", "mean_logp")}
            rows.append(dict(case=name, template=template, prompt=prompt, arms=arms))
            write("restored.json" if args.restore else "queries.json", rows)
    if args.restore:
        return
    write("environment.json", dict(gpu=torch.cuda.get_device_name(), **backbone.identity,
          facts={"A": A, "B": B}, ablation=args.ablation,
          cells={k: cells.read(v)[0] for k, v in ids.items()}))
    if args.fresh_process:
        del factories, full, cells, backbone, live
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, __file__, str(args.source), str(args.snapshot),
                        str(args.tokenizer), str(out), "--restore"], check=True)
        restored = json.loads((out/"restored.json").read_text())
        comparison = []
        for direct, imported in zip(rows, restored, strict=True):
            a = direct["arms"]["recurrent_only"]
            b = imported["arms"]["restored_recurrent_only"]
            comparison.append(dict(case=direct["case"], template=direct["template"],
                gold_total_logp_delta=b["candidates"][0]["total_logp"]-a["candidates"][0]["total_logp"],
                mean_margin_delta=b["mean_logp"]["margin"]-a["mean_logp"]["margin"]))
        write("restore-comparison.json", comparison)
    write("decision.json", dict(status="DIAGNOSTIC_COMPLETE", scope="DEVELOPMENT_ONLY",
                                primary_template="cloze", fresh_process=args.fresh_process))


if __name__ == "__main__":
    main()
