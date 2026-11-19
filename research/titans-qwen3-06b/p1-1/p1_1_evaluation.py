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

    histories, rows, diagnostics = [], [], []
    observed = []
    gate_observed = []

    def capture_read(_module, _args, output):
        gate_observed.append(output.detach())

    original_read = wrapper.memory.read_only

    def trace_read(*args, **kwargs):
        output = original_read(*args, **kwargs)
        observed.append(output[0].detach())
        return output
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
            reads, gates = [], []
            if index < 16:
                wrapper.memory.read_only = trace_read
                handle = wrapper.memory.history_mix_proj.register_forward_hook(capture_read)
                try:
                    for state in (histories[index], *(histories[j] for j in donor_assignments[index])):
                        observed.clear()
                        gate_observed.clear()
                        intervene(state)
                        reads.append(observed[0].float())
                        gate = (gate_observed[0].float() +
                                wrapper.memory.history_mix.flatten()).sigmoid()
                        gates.append({"mean": gate.mean().item(),
                                      "below_001": (gate < .01).float().mean().item(),
                                      "above_05": (gate > .5).float().mean().item(),
                                      "above_099": (gate > .99).float().mean().item()})
                finally:
                    handle.remove()
                    del wrapper.memory.read_only
                diagnostics.append({
                    "document_index": index, "gate": gates,
                    "read_cosine_correct_donor": [
                        torch.nn.functional.cosine_similarity(reads[0].flatten(), donor.flatten(),
                                                              dim=0).item()
                        for donor in reads[1:]],
                    "read_rms_delta": [
                        (reads[0] - donor).square().mean().sqrt().item() for donor in reads[1:]],
                })
            row["correct"] = intervene(histories[index])
            row["empty"] = intervene(None)
            if abs(row["empty"] - row["chunked_attention"]) > 1e-6:
                raise RuntimeError("Empty Memory differs from native chunked attention")
            row["zero_read"] = intervene(histories[index], "zero")
            donors = [intervene(histories[donor]) for donor in donor_assignments[index]]
            row.update({f"donor_{i}": value for i, value in enumerate(donors, 1)})
            row["donor_mean"] = mean(donors)
            rows.append(row)
        return {"format": "minicells.titans-p1-1.evaluation.v1", "documents": rows, "diagnostics": diagnostics,
                "mean": {key: mean(row[key] for row in rows) for key in rows[0]}}
    finally:
        block.self_attn, wrapper.state, wrapper.pending, wrapper.read_override = saved
