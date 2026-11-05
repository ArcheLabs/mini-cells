"""Tiny automatic forward-only CPU gate, no data downloads or optimizer."""
from types import SimpleNamespace
import torch
from model import Core,Config
from runtime import Runtime,weight_hash
from generative_baselines import freeze


class ContractTokenizer:
    def token_to_id(self,name):return {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3}[name]
    def encode(self,text):return SimpleNamespace(ids=[4+ord(c)%8 for c in text])
    def decode(self,ids,skip_special_tokens=True):return ''.join(chr(97+t-4) for t in ids if t>=4)


@torch.inference_mode()
def check_generation():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713);core=freeze(Core(Config(vocab=12,cells=4,dim=8)))
        tokenizer=ContractTokenizer();identity={'weights_sha256':weight_hash(core),'kind':'synthetic'}
        def new(eos=False):return Runtime(core).generation(tokenizer,identity,'synthetic-tokenizer',713,eos)
        seen=[];inject=core.inject
        def consume(state,ids):seen.extend(ids);return inject(state,ids)
        core.inject=consume
        runtime=new();runtime.prefill('abc');ids=[2]+tokenizer.encode('abc').ids
        state=core.init_state(1)
        for token in ids:state,logits=core.step(state,[token])
        if not torch.equal(state,runtime.state) or not torch.equal(logits,runtime.read()):raise ValueError('prefill contract failed')
        before=runtime.state.clone();rng=runtime.rng.get_state().clone();events=runtime.events
        runtime.read();runtime.read()
        if not torch.equal(before,runtime.state) or not torch.equal(rng,runtime.rng.get_state()) or runtime.events!=events:raise ValueError('read mutated runtime')
        seen.clear();generated=runtime.generate(4)
        if seen!=generated['token_ids'] or len(seen)!=4:raise ValueError('exactly-once/no-replay contract failed')
        core.inject=inject
        for config in ({'temperature':0.},{'temperature':.8,'top_k':5,'top_p':.95}):
            a=new();a.prefill('abc');whole=a.generate(8,**config)
            b=new();b.prefill('abc');first=b.generate(3,**config);c=new();c.restore(b.snapshot());tail=c.generate(5)
            if whole['token_ids']!=first['token_ids']+tail['token_ids'] or not torch.equal(a.state,c.state):raise ValueError('snapshot continuation failed')
        other=new();other.prefill('other');before=other.state.clone();runtime.prefill('fresh')
        if not torch.equal(before,other.state):raise ValueError('session reset leaked state')
        read=core.read
        def eos_read(state):
            values=state.new_full((1,12),-20.);values[:,3]=20.;return values
        core.read=eos_read;terminal=new(True);terminal.prefill('');value=terminal.generate(4)
        if value['token_ids']!=[3] or terminal.generate(1)['generated_tokens']!=0:raise ValueError('EOS contract failed')
        def stop_read(state):
            values=state.new_full((1,12),-20.);values[:,4]=20.;return values
        core.read=stop_read;terminal=new(True);terminal.prefill('');value=terminal.generate(4,stop_sequences=['aa'])
        if value['token_ids']!=[4,4] or value['text']!='':raise ValueError('stop sequence contract failed')
        core.read=read
    return dict.fromkeys(['prefill','exactly_once','read_immutable','snapshot_greedy_and_sampling','session_isolation','eos_stop','no_history_replay'],'PASS')
