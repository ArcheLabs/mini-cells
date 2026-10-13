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

def test_unique_pilot_selection_cross_split_duplicates_and_audit():
    from exp006.data import select_unique_pilot_stories,assert_disjoint,story_digest
    rows=[{'text':s} for s in ('A','A','B','C','A','D','D','E','reserved')]
    train,dev,audit=select_unique_pilot_stories(iter(rows),(0,3),(3,5),8)
    assert train==['A','B','C'] and dev==['D','E']
    assert audit['duplicates_removed']==3 and audit['raw_rows_scanned']==8
    assert audit['raw_row_indices']=={'train':[0,2,3],'development':[5,7]}
    assert_disjoint(list(map(story_digest,train)),list(map(story_digest,dev)))
    assert (train,dev,audit)==select_unique_pilot_stories(iter(rows),(0,3),(3,5),8)

def test_unique_pilot_does_not_read_reserved_boundary():
    from exp006.data import select_unique_pilot_stories
    def rows():
        yield {'text':'same'}
        yield {'text':'same'}
        raise AssertionError('reserved row accessed')
    with pytest.raises(ValueError,match='insufficient unique'):
        select_unique_pilot_stories(rows(),(0,1),(1,2),2)

def test_asset_preparation_reuses_strict_encoder_after_pilot_dedup(tmp_path,monkeypatch):
    import sys,json,types
    from exp006.assets import prepare
    from exp006.config import digest
    from exp006.data import encode_stories
    class Tokenizer:
        def __init__(self,*args):pass
        def train_from_iterator(self,texts,*args):assert list(texts)==['A','B']
        def save(self,path):
            from pathlib import Path
            Path(path).write_text('{}')
        def get_vocab_size(self):return 8192
        def token_to_id(self,token):return 1
        def encode(self,text):return types.SimpleNamespace(ids=[2]*130)
    fake=types.SimpleNamespace(Tokenizer=Tokenizer,
        models=types.SimpleNamespace(BPE=lambda **kw:None),
        pre_tokenizers=types.SimpleNamespace(ByteLevel=lambda **kw:None),
        decoders=types.SimpleNamespace(ByteLevel=lambda:None),
        trainers=types.SimpleNamespace(BpeTrainer=lambda **kw:None))
    fake.pre_tokenizers.ByteLevel.alphabet=lambda:[]
    monkeypatch.setitem(sys.modules,'tokenizers',fake)
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=lambda *a,**kw:iter(
        [{'text':s} for s in ('A','A','B','C','A','D','D','E')])))
    source=tmp_path/'source.parquet';source.write_bytes(b'fixture')
    c=p.proposed(1000000)
    c['data'].update(train_range=[0,3],development_range=[3,5],confirmation_range=[8,9],tokenizer_stories=2)
    result=prepare(c,tmp_path/'assets',source,digest(source))
    assert result['story_selection_audit']['duplicates_removed']==3
    windows=json.loads((tmp_path/'assets/windows.json').read_text())
    assert windows['train']['stories']==3 and windows['development']['stories']==2
    assert not result['confirmation_materialized']
    with pytest.raises(ValueError,match='duplicate stories'):
        encode_stories(['A','A'],Tokenizer(),125)
