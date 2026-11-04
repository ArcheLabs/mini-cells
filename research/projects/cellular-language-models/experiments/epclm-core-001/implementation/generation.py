"""Stateful generation session over the existing Runtime event executor."""
import hashlib
import json
import math
import time
import torch
from runtime import Runtime,weight_hash
from sampling import sample


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def snapshot_hash(snapshot):
    rng=snapshot['rng_state'].contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256((snapshot['core']['sha256']+digest(snapshot['generation'])).encode()+rng).hexdigest()


class GenerationSession(Runtime):
    def __init__(self,core,tokenizer,checkpoint_identity,tokenizer_identity,seed=713,eos_trained=True):
        super().__init__(core)
        if core.training or any(p.requires_grad or p.dtype!=torch.float32 for p in core.parameters()):raise ValueError('generation requires frozen eval model')
        self.tokenizer=tokenizer
        self.checkpoint_identity=json.loads(json.dumps(checkpoint_identity))
        self.tokenizer_identity=tokenizer_identity
        self.model_hash=weight_hash(core)
        if checkpoint_identity.get('weights_sha256')!=self.model_hash:raise ValueError('generation weight identity differs')
        from dataclasses import asdict
        self.config_hash=digest({'runtime':asdict(core.config),'reference':getattr(core,'reference_config',None)})
        self.bos=tokenizer.token_to_id('<bos>');self.eos=tokenizer.token_to_id('<eos>') if eos_trained else None
        self.pad=tokenizer.token_to_id('<pad>')
        if self.bos is None or (self.eos is not None and self.eos==self.bos):raise ValueError('invalid trained special tokens')
        self.seed=seed;self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.state=None;self.output_tokens=[];self.stopped=False;self.stop_reason=None
        self.generation_config={'temperature':0.,'top_k':0,'top_p':1.,'stop_sequences':[]}
        self.prompt_token_count=0;self.deadline=float("inf")

    def _check(self):
        if self.state is None:raise ValueError('prefill or restore a generation session first')

    @torch.inference_mode()
    def _consume(self,token):
        if time.monotonic()>=self.deadline:raise TimeoutError('generation execution budget exhausted')
        if not isinstance(token,int) or not 0<=token<self.core.config.vocab:raise ValueError('invalid token ID')
        with torch.autocast(device_type=self.state.device.type,enabled=False):
            self.state,logits=super().step(self.state,[token])
        if not torch.isfinite(self.state).all() or not torch.isfinite(logits).all():raise FloatingPointError('non-finite generation state/logits')
        return logits

    def reset(self,session='generation'):
        self.state=None;self.events=self.internal_steps=0;self.session=session
        self.output_tokens=[];self.stopped=False;self.stop_reason=None;self.prompt_token_count=0
        self.rng.manual_seed(self.seed)
        self.generation_config={'temperature':0.,'top_k':0,'top_p':1.,'stop_sequences':[]}

    @torch.inference_mode()
    def prefill(self,prompt='',session='generation'):
        ids=self.tokenizer.encode(prompt).ids if isinstance(prompt,str) else list(prompt)
        if not ids or ids[0]!=self.bos:ids=[self.bos]+ids
        if any(token in (self.pad,self.eos,self.bos) for token in ids[1:]):raise ValueError('prompt contains document boundary/padding token')
        self.reset(session);self.state=self.core.init_state(1)
        for token in ids:logits=self._consume(token)
        self.prompt_token_count=len(ids)
        return {'logits':logits,'consumed_tokens':len(ids)}

    @torch.inference_mode()
    def read(self):
        self._check()
        with torch.autocast(device_type=self.state.device.type,enabled=False):logits=self.core.read(self.state)
        if not torch.isfinite(logits).all():raise FloatingPointError('non-finite readout')
        return logits

    @torch.inference_mode()
    def evolve(self,steps):
        self._check()
        with torch.autocast(device_type=self.state.device.type,enabled=False):self.state=super().evolve(self.state,steps)
        if not torch.isfinite(self.state).all():raise FloatingPointError('non-finite extra evolution')
        return self.state

    def _decoded(self):
        # Hide document control tokens, but keep <unk> visible for quality review.
        visible=[t for t in self.output_tokens if t not in (self.pad,self.bos,self.eos)]
        return self.tokenizer.decode(visible,skip_special_tokens=False)

    def _text(self):
        text=self._decoded()
        for stop in self.generation_config['stop_sequences']:
            if stop in text:text=text[:text.index(stop)]
        return text

    @torch.inference_mode()
    def generate(self,max_new_tokens=128,temperature=None,top_k=None,top_p=None,stop_sequences=None):
        self._check()
        if not isinstance(max_new_tokens,int) or max_new_tokens<0:raise ValueError('invalid generation length')
        config=dict(self.generation_config)
        for key,value in [('temperature',temperature),('top_k',top_k),('top_p',top_p),('stop_sequences',stop_sequences)]:
            if value is not None:config[key]=value
        if not all(isinstance(s,str) and s for s in config['stop_sequences']):raise ValueError('stop sequences must be nonempty strings')
        # Validate even zero-length/stopped calls without advancing RNG or state.
        if (not math.isfinite(config['temperature']) or config['temperature']<0 or not isinstance(config['top_k'],int)
                or config['top_k']<0 or not 0<config['top_p']<=1):raise ValueError('invalid sampling configuration')
        self.generation_config=json.loads(json.dumps(config));generated=[];entropies=[];raw_entropies=[]
        logits=self.read() if max_new_tokens and not self.stopped else None
        for _ in range(max_new_tokens):
            if self.stopped:break
            probabilities=logits.float().softmax(-1)
            raw_entropies.append(float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()))
            token,entropy=sample(logits,self.rng,config['temperature'],config['top_k'],config['top_p'],(self.pad,self.bos))
            logits=self._consume(token)  # Includes EOS once, then no more events in a stopped session.
            self.output_tokens.append(token);generated.append(token);entropies.append(entropy)
            if self.eos is not None and token==self.eos:self.stopped=True;self.stop_reason='EOS'
            elif any(s in self._decoded() for s in config['stop_sequences']):
                self.stopped=True;self.stop_reason='STOP_SEQUENCE'
        return {'token_ids':generated,'text':self._text(),'generated_tokens':len(generated),
                'stop_reason':self.stop_reason or 'MAX_NEW_TOKENS','stopped':self.stopped,
                'mean_sampling_entropy':sum(entropies)/len(entropies) if entropies else None,
                'mean_raw_token_entropy':sum(raw_entropies)/len(raw_entropies) if raw_entropies else None,
                'events':self.events,'internal_steps':self.internal_steps}

    def snapshot(self):
        self._check()
        meta={'format':'minicells.generation-session.v1','checkpoint_identity':self.checkpoint_identity,
              'tokenizer_identity':self.tokenizer_identity,'config_hash':self.config_hash,
              'generation_config':self.generation_config,'seed':self.seed,'output_tokens':self.output_tokens,
              'stopped':self.stopped,'stop_reason':self.stop_reason,'prompt_token_count':self.prompt_token_count,
              'eos_trained':self.eos is not None}
        snapshot={'core':super().snapshot(self.state),'generation':json.loads(json.dumps(meta)),
                  'rng_state':self.rng.get_state().clone()}
        snapshot['sha256']=snapshot_hash(snapshot)
        return snapshot

    def restore(self,snapshot):
        meta=snapshot['generation']
        if (snapshot_hash(snapshot)!=snapshot['sha256'] or meta['format']!='minicells.generation-session.v1'
                or meta['checkpoint_identity']!=self.checkpoint_identity or meta['tokenizer_identity']!=self.tokenizer_identity
                or meta['config_hash']!=self.config_hash or meta['eos_trained']!=(self.eos is not None)):
            raise ValueError('generation snapshot identity/checksum mismatch')
        generator=torch.Generator(device='cpu');generator.set_state(snapshot['rng_state'])
        state=super().restore(snapshot['core'])
        self.state=state;self.rng=generator;self.seed=meta['seed'];self.generation_config=json.loads(json.dumps(meta['generation_config']))
        self.output_tokens=list(meta['output_tokens']);self.stopped=meta['stopped'];self.stop_reason=meta['stop_reason']
        self.prompt_token_count=meta['prompt_token_count']
        return state
