"""Hold query tokens and positions constant across memory interventions."""
import math

import torch
from torch.nn import functional as F

from p1 import CHUNK, LAYER


@torch.no_grad()
def evaluate(model, wrapper, documents):
    block = model.model.layers[LAYER]
    device = next(model.parameters()).device

    def logits(ids, offset):
        inputs = torch.tensor([ids], device=device)
        positions = torch.arange(offset, offset + len(ids), device=device)[None]
        return model(input_ids=inputs, position_ids=positions, use_cache=False).logits

    def history(ids):
        wrapper.state = None
        for offset in range(0, len(ids) - CHUNK, CHUNK):
            logits(ids[offset:offset + CHUNK], offset)
            wrapper.commit()
        state = wrapper.state
        wrapper.pending = None
        return state

    rows = []
    try:
        for index, ids in enumerate(documents):
            offset = len(ids) - CHUNK
            target = torch.tensor(ids[offset + 1:], device=device)

            def score(values):
                return F.cross_entropy(values[0, -CHUNK:-1].float(), target).item()

            correct = history(ids)
            donor = history(documents[(index + 1) % len(documents)])
            block.self_attn = wrapper.attention
            row = {"full_attention": score(logits(ids, 0)),
                   "chunked_attention": score(logits(ids[offset:], offset))}
            block.self_attn = wrapper
            for name, state in (("correct", correct), ("donor", donor),
                                ("empty", None), ("reset", None)):
                wrapper.state = state
                row[name] = score(logits(ids[offset:], offset))
            row["delta_empty"] = row["empty"] - row["correct"]
            row["delta_donor"] = row["donor"] - row["correct"]
            rows.append(row)
    finally:
        wrapper.state = None
        wrapper.pending = None
        block.self_attn = wrapper
    averages = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    return {"rows": rows, "mean": averages, "correct_perplexity": math.exp(averages["correct"])}
