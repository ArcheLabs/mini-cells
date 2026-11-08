import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from data import MQARConfig, episode, batch, dataset_manifest, digest, train_bpe, TOKENS, WIKITEXT


def test_mqar_alignment_reproducibility_and_two_generalization_regimes():
    for regime in ('unseen_pairs','unseen_symbols'):
        c=MQARConfig(bindings=16,length=96,queries=1,interference=.5,regime=regime)
        sets={s:[episode(c,s,i) for i in range(16)] for s in ('train','validation','test')}
        assert sets['train'][0]==episode(c,'train',0)
        dataset_manifest(c.__dict__,sets,'fixture-tokenizer')
        supports={s:{tuple(p) for r in rows for p in r['bindings']} for s,rows in sets.items()}
        assert not supports['train'] & supports['test']
        for rows in sets.values():
            for r in rows:
                assert r['query_positions']==[95]
                assert r['input_ids'][-3:]==[5,r['input_ids'][-2],6]
                k=r['input_ids'][-2]-8
                assert r['labels'][-1]==264+dict(r['bindings'])[k]
                assert sum(v!=-100 for v in r['labels'])==1
                assert sum(r['query_mask'])==1
        if regime=='unseen_symbols':
            symbols=lambda s:{v for r in sets[s] for p in r['bindings'] for v in p}
            assert not symbols('train') & symbols('test')
    c=MQARConfig(queries=4)
    r=episode(c,'train',0)
    assert sum(r['query_mask'])==4 and r['query_positions'][-1]==127
    b=batch([r,episode(c,'train',1)])
    old=b.input_ids[1].clone();b.input_ids[0,0]=42
    assert torch.equal(old,b.input_ids[1])
    with pytest.raises(ValueError):batch([r,r])
    with pytest.raises(ValueError):MQARConfig(length=3)


def test_tokenizer_fixed_ids_train_only_adapter_and_manifest(tmp_path,monkeypatch):
    import data
    # This is a tokenizer fixture, not WikiText training evidence.
    texts=['small training fixture with repeated language words']*2
    tok=train_bpe(texts,tmp_path/'tokenizer.json',vocab_size=800)
    assert [tok.token_to_id(t) for t in TOKENS]==list(range(536))
    assert tok.get_vocab_size()==800
    calls=[]
    def fake(*args,**kw):
        calls.append(kw)
        return iter([{'text':'\n'},{'text':'some train line\r\n'}])
    import datasets
    monkeypatch.setattr(datasets,'load_dataset',fake)
    assert data.wikitext_prefix('train',1)==['some train line\n']
    assert WIKITEXT['revision'] in calls[0]['data_files']['train'][0]
    assert '/wikitext-103-raw-v1/train-' in calls[0]['data_files']['train'][0]
    with pytest.raises(ValueError):data.wikitext_prefix('test',1)
    assert len(calls)==1
