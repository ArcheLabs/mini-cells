"""Matched final-chunk NLL with cached, read-only historical states."""
from statistics import mean

import torch
from torch.nn.functional import cross_entropy

CHUNK = 512


@torch.no_grad()
def evaluate_p1_1(model, wrapper, documents, donor_assignments, *, baseline=None):
    block = model.model.layers[3]
    device = next(model.parameters()).device
    saved = block.self_attn, wrapper.state, wrapper.pending, wrapper.read_override

    def forward(ids, offset):
        tokens = torch.as_tensor(ids, device=device)[None]
        positions = torch.arange(offset, offset + len(ids), device=device)[None]
        return model(input_ids=tokens, position_ids=positions, use_cache=False).logits

    histories, rows = [], []
    try:
        block.self_attn = wrapper
        wrapper.read_override = None
        for document in documents:
            if len(document) != 4 * CHUNK:
                raise ValueError("P1.1 requires 2048-token episodes")
            wrapper.state = None
            for offset in (0, CHUNK, 2 * CHUNK):
                forward(document[offset:offset + CHUNK], offset)
                wrapper.commit()
            histories.append(wrapper.state)
        for index, document in enumerate(documents):
            targets = torch.as_tensor(document[-CHUNK + 1:], device=device)
            query = document[-CHUNK:]

            def score(logits):
                return cross_entropy(logits[0, -CHUNK:-1].float(), targets).item()

            def intervene(state, override=None):
                wrapper.state = state
                wrapper.read_override = override
                return score(forward(query, 3 * CHUNK))

            if baseline is None:
                block.self_attn = wrapper.attention
                row = {"full_attention": score(forward(document, 0)),
                       "chunked_attention": score(forward(query, 3 * CHUNK))}
                block.self_attn = wrapper
            else:
                row = {key: baseline[index][key]
                       for key in ("full_attention", "chunked_attention")}
            row["correct"] = intervene(histories[index])
            row["empty"] = intervene(None)
            if abs(row["empty"] - row["chunked_attention"]) > 1e-6:
                raise RuntimeError("Empty Memory differs from native chunked attention")
            row["zero_read"] = intervene(histories[index], "zero")
            donors = [intervene(histories[donor]) for donor in donor_assignments[index]]
            row.update({f"donor_{i}": value for i, value in enumerate(donors, 1)})
            row["donor_mean"] = mean(donors)
            rows.append(row)
        return {"format": "minicells.titans-p1-1.evaluation.v1", "documents": rows,
                "mean": {key: mean(row[key] for row in rows) for key in rows[0]}}
    finally:
        block.self_attn, wrapper.state, wrapper.pending, wrapper.read_override = saved
