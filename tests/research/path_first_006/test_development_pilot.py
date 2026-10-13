import pytest
from exp006 import development_pilot as p

def test_pilot_config_is_dev_only():
    short=p.proposed(1000000);long=p.proposed(10000000)
    assert short["training"]["steps"]==500 and long["training"]["steps"]==5000
    assert short["training"]["warmup_steps"]==50
    assert short["data"]["split"]=="train"
    assert short["authorization"]["confirmation"] is False
    assert short["seeds"]["confirmation"]==[]
    with pytest.raises(ValueError):p.proposed(2000000)

def test_pilot_identity_separate(monkeypatch):
    monkeypatch.setattr(p,"source_commit",lambda:"a"*40)
    c=p.proposed(1000000);c["data"]["manifest_sha256"]="b"*64
    i=p.ident(c,"R2",6060611,"a"*40)
    assert i["role"]=="development" and i["run_class"]=="UNREGISTERED_DEV_PILOT"
    with pytest.raises(ValueError):p.ident(c,"R2",6060611,"c"*40)

def test_no_implicit_replacement(tmp_path):
    folder=tmp_path/"pilot";(folder/"assets").mkdir(parents=True)
    with pytest.raises(FileExistsError):p.prepare_pilot(folder,1000000)
