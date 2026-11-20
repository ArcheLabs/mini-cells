"""Train a bounded memory residual on frozen Qwen answer likelihood."""
import time

import torch
from torch import nn
from torch.nn import functional as F


def evaluate_fusion(adapter, tokenizer, memory, data, encode, deadline):
    device = next(adapter.model.parameters()).device
    hidden_size = adapter.model.config.hidden_size
    projection = nn.Linear(64, hidden_size, bias=False, device=device)
    gate = nn.Parameter(torch.tensor(-2.1972246, device=device))
    optimizer = torch.optim.AdamW([*projection.parameters(), gate], lr=.001)

    def fixture(rows, control):
        features, _ = encode(adapter, tokenizer, rows)
        captured = []
        with torch.no_grad():
            memory(features, control=control, capture=captured)
            reads = torch.cat([read.reshape(len(rows), 4, 32).mean(1)
                               for _, _, read in captured], dim=-1).to(device)
            result = []
            for (_, question, value), recall in zip(rows, reads):
                prompt = tokenizer.encode(question, add_special_tokens=False)
                answer = tokenizer.encode(' ' + value, add_special_tokens=False)
                ids = torch.tensor([prompt + answer], device=device)
                hidden, _ = adapter.encode(ids[:, :-1])
                result.append((hidden[:, len(prompt)-1:].float(), ids[:, len(prompt):], recall))
        return result

    def losses(rows, enabled):
        terms = []
        hits = count = 0
        for hidden, targets, recall in rows:
            if enabled:
                residual = projection(recall)
                rms = residual.square().mean().add(1e-8).sqrt()
                hidden = hidden + gate.sigmoid() * hidden.square().mean(-1, keepdim=True).sqrt() * residual / rms
            logits = adapter.logits(hidden.to(adapter.model.lm_head.weight.dtype)).float()
            terms.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum'))
            hits += int(logits.argmax(-1).eq(targets).sum())
            count += targets.numel()
        return sum(terms)/count, hits, count

    updates = 0
    for rows in data['train'][:8]:
        if time.monotonic() >= deadline:
            return {'status': 'SKIPPED_BUDGET', 'updates': updates}, None
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = losses(fixture(rows, 'correct'), True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*projection.parameters(), gate], 1., error_if_nonfinite=True)
        optimizer.step()
        updates += 1
    records = []
    for rows in data['test'][:4]:
        if time.monotonic() >= deadline:
            return {'status': 'SKIPPED_BUDGET', 'updates': updates, 'rows': records}, None
        with torch.no_grad():
            record = {}
            for arm in ('native', 'correct', 'empty', 'donor'):
                loss, hits, count = losses(fixture(rows, 'correct' if arm == 'native' else arm), arm != 'native')
                record[arm] = dict(nll=float(loss), correct=hits, tokens=count)
            records.append(record)
    return {'status': 'INCONCLUSIVE', 'updates': updates, 'rows': records,
            'scope': 'answer-token diagnostic; base-language regression unmeasured'}, dict(
                projection=projection.state_dict(), gate=gate.detach(), optimizer=optimizer.state_dict())
