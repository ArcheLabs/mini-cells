"""Exercise real Qwen attention and Titans updates with small CPU weights."""
from pathlib import Path
import sys

import pytest
import torch

from minicells_runtime import MultiCellRuntime
from minicells_runtime.qwen35 import Qwen35Adapter

EXPERIMENTS = Path(__file__).resolve().parents[1] / (
    'research/projects/cellular-language-models/experiments')
sys.path.insert(0, str(EXPERIMENTS / 'epclm-006-cross-session-cell-portability'))
from cell_artifact import export_cell, import_cell, state_hash
from memory import NeuralMemory


def test_independent_runtime_and_cell_roundtrip(tmp_path):
    pytest.importorskip("transformers")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'],
    )
    backbone = Qwen35Adapter(Qwen3_5ForCausalLM(config), {'fixture': 'qwen35'})
    memory = NeuralMemory(32)
    projection, gate = torch.nn.Linear(32, 32, bias=False), torch.nn.Linear(32, 1)
    runtime = MultiCellRuntime(backbone, memory, projection, gate)
    for name in ('A', 'B', 'C'):
        runtime.create(name)
    ids = torch.tensor([[2, 3, 4]])
    before = {name: state_hash(runtime.get(name)) for name in runtime.list()}
    baseline = runtime.generate(ids)
    learned = runtime.generate(ids, read_cells=('A', 'B'), update_cell='C',
                               learning_payload=(ids, ids.flip(1)))
    torch.testing.assert_close(learned['logits'], baseline['logits'], atol=0, rtol=0)
    assert learned['read_revisions'] == {'A': 0, 'B': 0}
    assert state_hash(runtime.get('C')) != before['C']
    assert all(state_hash(runtime.get(name)) == before[name] for name in ('A', 'B'))
    read = runtime.generate(ids, read_cells=('A', 'B', 'C'))
    assert not torch.equal(read['logits'], baseline['logits'])
    runtime.clone('C', 'clone')
    runtime.fork('C', 'fork')
    parent = state_hash(runtime.get('C'))
    runtime.generate(ids, update_cell='clone', learning_payload=(ids, ids))
    assert state_hash(runtime.get('C')) == state_hash(runtime.get('fork')) == parent

    path = tmp_path / 'bundle'
    identity = {'fixture': 'shared-identical-weights'}
    runtime.export_bundle(learned['runtime_state'], ('A', 'C'), path,
                          identity=identity, exporter=export_cell)
    fresh = MultiCellRuntime(backbone, memory, projection, gate)
    restored = fresh.import_bundle(path, identity=identity, importer=import_cell)
    assert restored.processed == 3
    assert restored.cache.get_seq_length() == 0
    for actual, original in zip(restored.cache.recurrent_states,
                                learned['runtime_state'].cache.recurrent_states):
        if original is not None:
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
    assert state_hash(fresh.get('C')) == parent
    assert fresh.generate(ids, runtime_state=restored, read_cells=('C',))['logits'].isfinite().all()
    state = path / 'runtime' / 'state.safetensors'
    state.write_bytes(state.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='snapshot'):
        backbone.import_runtime_state(path / 'runtime')
