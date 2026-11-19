"""Paired read-only synthetic evaluation for the Gate 1 smoke."""
import torch
from adapter import LAYER


@torch.no_grad()
def evaluate(model, cell, rows, device):
    from transformers import DynamicCache

    def segment(ids):
        cache = DynamicCache()
        for offset, token in enumerate(ids):
            hidden = model.model(input_ids=torch.tensor([[token]],device=device),
                past_key_values=cache, use_cache=True,
                position_ids=torch.tensor([[offset]],device=device)).last_hidden_state
        return model.lm_head(hidden[:, -1]).float()[0]

    states = []
    block = model.model.layers[LAYER]
    try:
        for row in rows:
            cell.reset_episode()
            for ids in row["tokens"][:2]:
                segment(ids)
            states.append(cell.state)
        report = []
        for i, row in enumerate(rows):
            query, target = row["tokens"][-1][:-1], row["tokens"][-1][-1]
            donor = (i+1) % len(rows)
            record = dict(episode=i, donor_index=donor)
            for name,state in (("correct",states[i]),("empty",None),
                               ("donor",states[donor]),("zero_read",states[i]),("no_memory",None)):
                cell.state,cell.read_only,cell.zero_read = state,True,name=="zero_read"
                block.self_attn = cell.attention if name=="no_memory" else cell
                logits = segment(query)
                if cell.state is not state:
                    raise RuntimeError("Evaluation replaced Cell state")
                record[name] = dict(nll=-logits.log_softmax(0)[target].item(),
                                    exact=int(logits.argmax().item()==target))
            report.append(record)
        return report
    finally:
        block.self_attn = cell
        cell.reset_episode()
