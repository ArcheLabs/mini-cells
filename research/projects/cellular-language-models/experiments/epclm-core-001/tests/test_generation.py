"""Seven generation contracts plus two focused adapter/asset checks; no training/benchmark."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from model import Core,Config
from runtime import Runtime,weight_hash
from generation import GenerationSession
from generative_baselines import NoCommunicationCore


class Tokenizer:
    def token_to_id(self,name):return {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3}[name]
    def encode(self,text):return SimpleNamespace(ids=[4+(ord(c)%8) for c in text])
    def decode(self,ids,skip_special_tokens=True):return ''.join(chr(97+t-4) for t in ids if t>=4)


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713);model=Core(Config(vocab=12,cells=4,dim=8)).eval()
    for p in model.parameters():p.requires_grad_(False)
    tokenizer=Tokenizer();identity={'weights_sha256':weight_hash(model),'checkpoint_sha256':'fixture','transition':'NORMAL'}
    return model,tokenizer,identity


def new(fixture,eos=False):
    model,tokenizer,identity=fixture
    return Runtime(model).generation(tokenizer,identity,'tokenizer-fixture',713,eos_trained=eos)


def test_prefill_manual_equivalence(fixture):
    from generative_preflight import check_generation
    assert all(value=='PASS' for value in check_generation().values())
    model,tokenizer,_=fixture;runtime=new(fixture);result=runtime.prefill('abc')
    state=model.init_state(1)
    for token in [2]+tokenizer.encode('abc').ids:state,logits=model.step(state,[token])
    assert torch.equal(state,runtime.state) and torch.equal(logits,result['logits'])
    assert result['consumed_tokens']==4
    assert new(fixture).prefill('')['consumed_tokens']==1


def test_exactly_once_consumption(fixture,monkeypatch):
    model,_,_=fixture;seen=[];inject=model.inject
    def record(state,token):seen.extend(token);return inject(state,token)
    monkeypatch.setattr(model,'inject',record)
    runtime=new(fixture);runtime.prefill([2,4,5]);result=runtime.generate(5)
    assert seen==[2,4,5]+result['token_ids']
    assert runtime.events==8 and runtime.internal_steps==32


def test_read_is_immutable(fixture):
    runtime=new(fixture);runtime.prefill('abc')
    state=runtime.state.clone();rng=runtime.rng.get_state().clone();events=runtime.events;steps=runtime.internal_steps
    assert torch.equal(runtime.read(),runtime.read()) and torch.equal(runtime.state,state)
    assert torch.equal(runtime.rng.get_state(),rng) and (runtime.events,runtime.internal_steps)==(events,steps)


def test_snapshot_continuity(fixture):
    for settings in ({'temperature':0.},{'temperature':.8,'top_k':5,'top_p':.95}):
        a=new(fixture);a.prefill('abc');whole=a.generate(64,**settings)
        b=new(fixture);b.prefill('abc');first=b.generate(20,**settings);snapshot=b.snapshot()
        c=new(fixture);c.restore(snapshot);tail=c.generate(44)
        assert whole['token_ids']==first['token_ids']+tail['token_ids'] and torch.equal(a.state,c.state)
        assert torch.equal(a.rng.get_state(),c.rng.get_state())
        bad=copy.deepcopy(snapshot);bad['rng_state'][0]^=1
        with pytest.raises(ValueError):c.restore(bad)
        d=new(fixture);d.tokenizer_identity='wrong'
        with pytest.raises(ValueError):d.restore(snapshot)


def test_session_isolation(fixture):
    a=new(fixture);b=new(fixture);a.prefill('abc');b.prefill('de')
    state=b.state.clone();events=b.events;a.generate(4);a.reset()
    assert a.state is None and a.events==0
    a.prefill('fresh')
    assert torch.equal(b.state,state) and b.events==events
    fresh=new(fixture);fresh.prefill('fresh');assert torch.equal(fresh.state,a.state)


def test_eos_stop_semantics(fixture,monkeypatch):
    model,_,_=fixture
    def fixed(token):
        def read(state):
            logits=torch.full((len(state),12),-20.,device=state.device);logits[:,token]=20.;return logits
        return read
    monkeypatch.setattr(model,'read',fixed(3))
    runtime=new(fixture,eos=True);runtime.prefill('');result=runtime.generate(10)
    assert result['token_ids']==[3] and result['text']=='' and result['stop_reason']=='EOS' and runtime.events==2
    assert runtime.generate(10)['generated_tokens']==0
    restored=new(fixture,eos=True);restored.restore(runtime.snapshot());assert restored.generate(1)['generated_tokens']==0
    monkeypatch.setattr(model,'read',fixed(4))
    runtime=new(fixture,eos=True);runtime.prefill('');result=runtime.generate(10,stop_sequences=['aa'])
    assert result['token_ids']==[4,4] and result['text']=='' and result['stop_reason']=='STOP_SEQUENCE'
    assert runtime.events==3


def test_no_history_replay(fixture,monkeypatch):
    model,_,_=fixture;seen=[];step=model.step
    def one(state,tokens,*args,**kwargs):
        assert len(tokens)==1;seen.extend(tokens);return step(state,tokens,*args,**kwargs)
    monkeypatch.setattr(model,'step',one)
    runtime=new(fixture);runtime.prefill([4,5,6]);first=runtime.generate(3);second=runtime.generate(2)
    assert seen==[2,4,5,6]+first['token_ids']+second['token_ids']
    assert not any(k in runtime.__dict__ for k in ('prompt_tokens','history','kv_cache','attention_cache'))
    assert runtime.state.shape==(1,4,8)
    # The new trained baseline is true center-only, not legacy neighbors-as-self C0.
    nc=NoCommunicationCore(model.config);nc.load_state_dict(model.state_dict())
    with torch.no_grad():
        state=torch.randn(1,4,8);normalized=nc.norm(state)
        expected=state+.1*nc.update(torch.cat((normalized,nc.center(normalized)),-1))
        assert torch.equal(nc.evolve(state,1),expected)


def test_gc_recovery_keeps_training_identity(tmp_path,monkeypatch):
    # One focused recovery regression: extra original assets, preferred version,
    # and an explicit opt-in before current-source optimizer checkpoints can return.
    import hashlib,json
    from frozen_model import recover_frozen_assets,BaselineIntegrityError
    base=Path(__file__).resolve().parents[1];root=next(p for p in base.parents if (p/'pyproject.toml').exists())
    monkeypatch.syspath_prepend(str(base.parent/'epclm-002-a0-persistent-state-recall/scripts'))
    import kaggle_output_recovery as backend
    sha=lambda data:hashlib.sha256(data).hexdigest()
    files={'C1.pt':b'trusted-original-c1','F0.pt':b'trusted-original-flat','assets/train.json':b'[]',
           'assets/tokenizer.json':b'{}','assets/development.json':b'[]','assets/manifest.json':b'{}'}
    identity={'source':'training-source','protocol':'training-protocol','data':sha(files['assets/manifest.json'])}
    checkpoint={'namespace':'training-pair','path':'C1.pt','sha256':sha(files['C1.pt'])}
    manifest={'identity':identity,'files':{k:sha(v) for k,v in files.items()}}
    payload={'training-pair/'+k:v for k,v in files.items()};payload['training-pair/recovery.json']=json.dumps(manifest).encode()
    visited=[]
    class API:
        def metadata(self):return {'version_number':9}
        def page(self,version,token):
            visited.append(version);return {'files':[{'file_name':n} for n in payload] if version==7 else []}
    def fetch(api,inventory,names,destination):
        for name in names:
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload[name])
    monkeypatch.setattr(backend,'_load_backend',lambda root:(API(),fetch))
    output=tmp_path/'gc';assets=tmp_path/'cache';progress_id={'source':'gc-source','protocol':'gc-plan','data':identity['data']}
    recover_frozen_assets(root,assets,identity,checkpoint,float('inf'),progress_output=output,progress_identity=progress_id,
        preferred_versions=[7],extra_asset_hashes={'assets/train.json':None,'F0.pt':sha(files['F0.pt'])},allow_training_progress=True)
    assert visited==[9,7] and (assets/'assets/train.json').exists() and (assets/'F0.pt').read_bytes()==files['F0.pt']
    result=json.dumps({'status':'COMPLETE','identity':progress_id}).encode()
    payload.clear();payload.update({'gc/decision.json':result,'gc/NO-COMM.pt':b'current-source-checkpoint'})
    payload['gc/recovery.json']=json.dumps({'identity':progress_id,'files':{k[3:]:sha(v) for k,v in payload.items()}}).encode()
    with pytest.raises(BaselineIntegrityError,match='unexpected progress'):
        recover_frozen_assets(root,tmp_path/'other',identity,checkpoint,float('inf'),progress_output=output,progress_identity=progress_id,preferred_versions=[7])
    terminal=recover_frozen_assets(root,tmp_path/'another',identity,checkpoint,float('inf'),progress_output=output,
                                  progress_identity=progress_id,preferred_versions=[7],allow_training_progress=True)
    assert terminal=='DIAGNOSTIC_TERMINAL' and (output/'NO-COMM.pt').read_bytes()==b'current-source-checkpoint'


@torch.inference_mode()
def test_existing_transformer_reference_adapter():
    # Only the new bounded-context adapter is checked; Native T1 is reused intact.
    from generative_baselines import TransformerReference,freeze
    torch.set_num_threads(1)
    reference=freeze(TransformerReference(vocab=12,context=3))
    identity={'weights_sha256':weight_hash(reference),'transition':'T1_REFERENCE_FIXTURE'}
    runtime=Runtime(reference).generation(Tokenizer(),identity,'tokenizer-fixture',713,True)
    runtime.prefill([4,5,6])
    expected=reference.native(torch.tensor([[4,5,6]]))[:,-1]
    assert torch.equal(runtime.read(),expected) and runtime.state.shape==(1,1,4)
    restored=Runtime(reference).generation(Tokenizer(),identity,'tokenizer-fixture',713,True)
    restored.restore(runtime.snapshot())
    assert torch.equal(restored.read(),expected)
    assert reference.reference_config['context_length']==3 and reference.reference_config['kv_cache'] is False
