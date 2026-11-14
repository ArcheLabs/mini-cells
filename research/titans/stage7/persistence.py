"""Verify portable Titans shards in a fresh interpreter with independent Qwen query encoding."""
import json
from pathlib import Path
import sys

import torch


def restore(path):
    from run_stage7 import Memory, Qwen35Adapter, MODEL_ID, REVISION, encode
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    from run_stage7 import BASE

    sys.path.insert(0, str(BASE.parents[2] / 'research/projects/cellular-language-models/experiments'
                          / 'epclm-006-cross-session-cell-portability'))
    from cell_artifact import import_cell

    payload = torch.load(path / 'fixture.pt', weights_only=True, map_location='cpu')
    adapter = Qwen35Adapter.load()
    if adapter.identity != payload['identity']['model']:
        raise ValueError('Restored backbone identity differs')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    (_, queries, _), _ = encode(adapter, tokenizer, json.loads((path / 'packets.json').read_text()))
    model = Memory().cuda()
    model.load_state_dict(payload['weights'])
    for index, shard in enumerate(model.shards):
        holder = SimpleNamespace(memory=shard, parameters=shard.parameters)
        cell = import_cell(path / f'shard-{index}', holder, payload['identity'])
        query, expected = payload['reads'][index]
        with torch.no_grad():
            fresh = torch.nn.functional.normalize(
                model.address[index](queries[:, index*4:index*4+4]), dim=-1).flatten(0, 1)[None]
            torch.testing.assert_close(fresh.cpu(), query, atol=1e-5, rtol=1e-4)
            actual = shard.recall(fresh, cell).cpu()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    (path / 'restore.json').write_text(json.dumps({'status': 'PASS', 'shards': 2}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    restore(Path(sys.argv[1]))
