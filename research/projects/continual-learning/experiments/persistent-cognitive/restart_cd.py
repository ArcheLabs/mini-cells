"""Verify trained-state-free Qwen continuation across independent processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

from smoke_cd import setup
from cell import Cell, CellSlot, restore, snapshot, state_hash
from session import Session, _tensors, advance, restore_session, snapshot_session


@torch.no_grad()
def worker(path, role):
    adapter, retrofit, identity = setup()
    model = adapter.model
    ids = torch.tensor([[101, 307, 509, 701, 907, 1103]], device='cuda')
    fresh = lambda: adapter.cache_type(adapter.config)
    if role == 'source':
        cell = Cell.create(retrofit.memories, identity)
        slot = CellSlot(cell)
        session = Session.create(cell, fresh())
        _, session = advance(session, slot, model, retrofit, ids[:, :4], identity)
        _, session = advance(session, slot, model, retrofit, ids[:, 4:5], identity)
        snapshot(slot.current, path/'cell')
        snapshot_session(session, path/'session')
        logits, final = advance(session, slot, model, retrofit, ids[:, 5:], identity)
        save_file({'logits': logits.logits.cpu().contiguous()}, path/'expected.safetensors')
        record = dict(pid=os.getpid(), state_hash=state_hash(slot.current), position=final.position)
    else:
        cell = restore(path/'cell', retrofit.memories, identity)
        if _tensors(Session.create(cell, fresh()).runtime_cache):
            raise RuntimeError('Cell restore populated Session state')
        # Shape probe uses unrelated tokens. Restored values come only from the snapshot.
        layout, _ = retrofit(model, None, input_ids=torch.zeros_like(ids[:, :4]),
                             use_cache=True, memory_enabled=False)
        session = restore_session(path/'session', cell, fresh(), _tensors(layout.past_key_values))
        slot = CellSlot(cell)
        logits, final = advance(session, slot, model, retrofit, ids[:, 5:], identity)
        expected = load_file(path/'expected.safetensors')['logits']
        torch.testing.assert_close(logits.logits.cpu(), expected, atol=1e-5, rtol=1e-5)
        record = dict(pid=os.getpid(), state_hash=state_hash(slot.current), position=final.position,
                      logits_max_abs=float((logits.logits.cpu()-expected).abs().max()))
    (path/f'{role}.json').write_text(json.dumps(record))


def main(output):
    path = output.parent
    reports = {}
    for role in ('source', 'target'):
        subprocess.run([sys.executable, __file__, str(output), role], check=True, timeout=240)
        reports[role] = json.loads((path/f'{role}.json').read_text())
        print(f'{role} worker exited successfully', flush=True)
    source, target = reports.values()
    if source['pid'] == target['pid'] or any(source[k] != target[k] for k in ('state_hash', 'position')):
        raise RuntimeError('Restart identity or continuation differs')
    output.write_text(json.dumps(dict(status='PCA_CD_RESTART_SMOKE_PASS', **reports,
        training_updates=0, cross_window_recall='UNVERIFIED', phase_ab='PARTIAL'), indent=2))


if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        worker(output.parent, sys.argv[2])
    else:
        main(output)
