"""Exercise the real process boundary and reject incompatible or damaged Cells."""
import pytest

from migration007 import import_cell, run
from preflight007 import direct_fixture


def test_fresh_process_migration(tmp_path):
    output = tmp_path / 'migration'
    result = run(output)
    assert result['status'] == 'PORTABILITY_PASS', result.get('error')
    assert result['source_exit_status'] == result['target_exit_status'] == 0
    assert result['source']['pid'] != result['target']['pid']
    assert result['source']['updates'] == 128
    assert result['target']['updates'] == 0
    assert result['comparison']['source_top1'] == result['comparison']['target_top1'] == 5
    assert result['comparison']['state_integrity']

    model, *_ = direct_fixture()
    artifact = output / 'cell'
    identity = result['target']['compatibility']
    for field in ('backbone_checkpoint_id', 'projection_version'):
        with pytest.raises(ValueError, match='compatibility mismatch'):
            import_cell(artifact, model, {**identity, field: 'incompatible'})
    with (artifact / 'state.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='integrity mismatch'):
        import_cell(artifact, model, identity)
