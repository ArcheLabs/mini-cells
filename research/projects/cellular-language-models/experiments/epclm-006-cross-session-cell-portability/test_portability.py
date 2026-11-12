"""Focused checks for sealed splits, identity and fail-closed tensors."""
import json
import uuid

import pytest
import torch

from portable_model import PortableModel,P
from portability_data import datasets,episode
from cell_artifact import CellSlot,compatibility,export_cell
from harness import tensors
from data import digest,file_hash
from e0 import write
from portability_run import schedule


def test_dataset_budget_and_split_isolation():
    train,val,banks,manifest=datasets('fixture-tokenizer')
    assert len(schedule(train))==32
    assert sum(len(r['input_ids']) for r in train)==P['input_tokens']
    assert sum(sum(r['query_mask']) for r in train)==P['query_targets']
    assert len(val)==32 and all(len(rows)==64 for rows in banks.values())
    for row in train:
        assert not any(264<=v<520 for v in row['target_ids'])
        assert set(row['facts']).isdisjoint(row['unrelated_facts'])
    assert manifest['test_status']=='SEALED_NOT_ACCESSED'
    with pytest.raises(ValueError,match='sealed'):
        episode('test',0,0)


def test_rehashed_bad_shape_rejected_without_detaching_good_cell(tmp_path):
    torch.set_num_threads(1)
    model=PortableModel(True).eval()
    identity=compatibility(model,'fixture')
    row=episode('validation',62900,0)
    with torch.no_grad():
        cell=model.acquire(tensors(model,row)['source_segments'])
    good=tmp_path/'good'
    bad=tmp_path/'bad'
    for path in (good,bad):
        export_cell(cell,path,identity,uuid.uuid4().hex)
    slot=CellSlot(model,identity)
    original=slot.replace(good)
    state=torch.load(bad/'state.pt',weights_only=True)
    name=next(iter(state['fast']))
    state['fast'][name]=state['fast'][name][...,:1]
    torch.save(state,bad/'state.pt')
    hashes={f:file_hash(bad/f) for f in ('manifest.json','state.pt')}
    write(bad/'integrity.json',dict(files=hashes,artifact_sha256=digest(hashes)))
    with pytest.raises(ValueError,match='shape'):
        slot.replace(bad)
    assert slot.cell is original
    manifest=json.loads((bad/'manifest.json').read_text())
    (bad/'manifest.json').write_text(json.dumps({**manifest,'revision':999}))
    with pytest.raises(ValueError,match='integrity'):
        slot.replace(bad)
    assert slot.cell is original
