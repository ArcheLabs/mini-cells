"""Bundle integrity, publication retry immutability and ledger corruption guards."""
from pathlib import Path
import sys,json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from b0_integrity import freeze_bundle,verify_bundle


def test_bundle_retry_is_byte_identical_and_mutation_rejected(tmp_path):
    (tmp_path/'checkpoint.pt').write_bytes(b'opaque fixture, never deserialized')
    (tmp_path/'metrics.json').write_text('{"updates":1}')
    package,files=freeze_bundle(tmp_path);before=package.read_bytes()
    verify_bundle(package,files);freeze_bundle(tmp_path);assert package.read_bytes()==before
    (tmp_path/'publication.json').write_text('{"publication_status":"VERIFIED"}')
    freeze_bundle(tmp_path);assert package.read_bytes()==before
    (tmp_path/'metrics.json').write_text('{"updates":2}')
    with pytest.raises(ValueError,match='immutable'):freeze_bundle(tmp_path)


def test_bundle_corruption_is_rejected(tmp_path):
    (tmp_path/'metrics.json').write_text('{}');package,files=freeze_bundle(tmp_path)
    package.write_bytes(b'corrupt archive')
    with pytest.raises(Exception):verify_bundle(package,files)
