"""E0 adapter and bounded M1 loop; reuse existing blocks and atomic checkpoints."""
from pathlib import Path
import json
import random
import sys
import time
import torch
from torch.nn import functional as F
from data import batch, digest

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p/'pyproject.toml').exists())
CORE = BASE.parent/'epclm-core-001'
sys.path[:0] = [str(CORE/'gc002-fair-architecture'), str(CORE/'implementation'),
               str(BASE.parent/'native-clm-v0/implementation')]
from cached_transformer import CachedTransformer, Config, rope
from trainer import save_checkpoint, load_checkpoint, rng, restore_rng


class E0(CachedTransformer):
    """Full causal attention over the supplied sequence; no persistent training KV.

    Retains existing parameters/initialization. Config.window is inert here;
    there is no architectural history window or inherited cached inference API.
    """
    def forward(self, ids):
        c = self.config
        if ids.ndim != 2 or ids.shape[1] == 0: raise ValueError('nonempty [B,T] required')
        h = self.token_embedding(ids); b,t,_ = h.shape
        positions = torch.arange(t,device=ids.device); d = c.dim//c.heads
        for block in self.blocks:
            x=block.n1(h);a=block.attn
            q=rope(a.q(x).view(b,t,c.heads,d).transpose(1,2),positions)
            k=rope(a.k(x).view(b,t,c.kv_heads,d).transpose(1,2),positions)
            v=a.v(x).view(b,t,c.kv_heads,d).transpose(1,2)
            n=c.heads//c.kv_heads
            y=F.scaled_dot_product_attention(q,k.repeat_interleave(n,1),v.repeat_interleave(n,1),
                                           dropout_p=0.0,is_causal=True)
            h=h+a.o(y.transpose(1,2).contiguous().view(b,t,c.dim))
            h=h+block.ffn(block.n2(h))
        return self.lm_head(self.final_norm(h))

    def cached(self,*args,**kwargs):
        raise ValueError('M1 E0 supports complete-sequence execution only')


def masked_loss(logits, labels):
    valid=labels!=-100
    if logits.shape[:2]!=labels.shape or not valid.any(): raise ValueError('invalid/empty supervision')
    return F.cross_entropy(logits[valid],labels[valid],reduction='sum')/valid.sum()


class Sampler:
    """Episode adapter using existing propose/resolve/ledger commit semantics."""
    def __init__(self,rows,seed,batch_size):
        if len(rows)<batch_size or batch_size<1:raise ValueError('dataset smaller than batch')
        self.rows=rows;self.batch_size=batch_size;self.identity=digest(rows)
        self.random=random.Random(seed);self.order=list(range(len(rows)));self.random.shuffle(self.order)
        self.cursor=0;self.pending=None

    def snapshot(self):
        if self.pending is not None:raise ValueError('unresolved sampler transaction')
        return {'identity':self.identity,'batch_size':self.batch_size,'order':self.order.copy(),
                'cursor':self.cursor,'rng':self.random.getstate()}

    def restore(self,s):
        if self.pending is not None or (s['identity'],s['batch_size'])!=(self.identity,self.batch_size):
            raise ValueError('sampler identity/pending mismatch')
        self.order=s['order'].copy();self.cursor=s['cursor'];self.random.setstate(s['rng'])

    def propose(self):
        before=self.snapshot()
        if self.cursor==len(self.rows):self.random.shuffle(self.order);self.cursor=0
        indices=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=len(indices)
        self.pending=before
        return [self.rows[i] for i in indices]

    def resolve(self,accepted,ledger,accounting=None):
        if self.pending is None:raise ValueError('no pending transaction')
        before=self.pending;self.pending=None
        if accepted:
            for k,v in accounting.items():ledger[k]+=v
            ledger['committed_updates']+=1
        else:self.restore(before)


def parameters(model):
    names={n:p.numel() for n,p in model.named_parameters()}
    return {'static_parameters':sum(names.values()),'embedding_parameters':model.token_embedding.weight.numel(),
            'tied_embedding_head':model.lm_head.weight is model.token_embedding.weight,
            'by_module':{'embedding':sum(v for k,v in names.items() if k.startswith('token_embedding')),
                         'blocks':sum(v for k,v in names.items() if k.startswith('blocks')),
                         'final_norm':sum(v for k,v in names.items() if k.startswith('final_norm'))}}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


@torch.no_grad()
def evaluate(model,rows,batch_size=4):
    device=next(model.parameters()).device;model.eval();loss_sum=0.;count=0;correct=0;queries=0;predictions={}
    for i in range(0,len(rows),batch_size):
        b=batch(rows[i:i+batch_size],device);logits=model(b.input_ids);valid=b.labels!=-100
        n=int(valid.sum());loss_sum+=float(masked_loss(logits,b.labels))*n;count+=n
        q=b.query_mask;pred=logits.argmax(-1)
        correct+=int(((pred==b.labels)&q).sum());queries+=int(q.sum())
        for p in pred[q].cpu().tolist():predictions[str(p)]=predictions.get(str(p),0)+1
    if not count:raise ValueError('empty evaluation')
    return {'nll':loss_sum/count,'supervised_targets':count,'query_targets':queries,
            'query_exact_match':correct/queries if queries else None,'query_prediction_counts':predictions}


def fit(model,rows,validation,config,out,identity,deadline=float('inf'),stop_after=None):
    """FP32 short-run adapter, one atomic committed payload per update.

    No scheduler is used in this development smoke (constant LR). Restore the
    last atomic payload on rejected/mutated updates, including RNG and sampler.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'checkpoint.pt'
    device=next(model.parameters()).device;sampler=Sampler(rows,config['seed'],config['batch'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.95),weight_decay=.1)
    ledger={k:0 for k in ('input_tokens','processed_tensor_tokens','padding_tokens','supervised_target_tokens','query_target_tokens','committed_updates')}
    history=[];initial=None
    def save():
        save_checkpoint({'format':'minicells.epclm004.m1-checkpoint.v1','identity':identity,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(device.type=='cuda'),
            'sampler':sampler.snapshot(),'ledger':ledger,'history':history,'initial_evaluation':initial},path)
    def restore():
        nonlocal ledger,history,initial
        state=load_checkpoint(path)
        if state['identity']!=identity:raise ValueError('checkpoint source/config/data/tokenizer differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        sampler.pending=None;sampler.restore(state['sampler']);restore_rng(state['rng'])
        ledger=state['ledger'];history=state['history'];initial=state['initial_evaluation']
    if path.exists():restore()
    else:
        initial=evaluate(model,validation,config['batch']);save()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    limit=config['updates'] if stop_after is None else min(stop_after,config['updates'])
    while ledger['committed_updates']<limit:
        if time.monotonic()>=deadline:break
        samples=sampler.propose();b=batch(samples,device);model.train();optimizer.zero_grad(set_to_none=True)
        started=time.perf_counter()
        if device.type=='cuda':torch.cuda.synchronize(device)
        try:
            logits=model(b.input_ids);loss=masked_loss(logits,b.labels)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.requires_grad]
            if any(g is None for g in grads) or not torch.stack([g.isfinite().all() for g in grads]).all():
                raise FloatingPointError('missing/nonfinite gradients; no clip/update')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            before=model.token_embedding.weight.detach().clone() if ledger['committed_updates']<2 else None
            optimizer.step()
            if not torch.stack([p.isfinite().all() for p in model.parameters()]).all():
                raise FloatingPointError('nonfinite updated parameters')
            changed=float((model.token_embedding.weight.detach()-before).norm()) if before is not None else None
            if before is not None and not changed>0:raise FloatingPointError('optimizer did not change embedding')
            if device.type=='cuda':torch.cuda.synchronize(device)
            q=b.query_mask;pred=logits.detach().argmax(-1)
            n=int((b.labels!=-100).sum());qn=int(q.sum())
            row={'update':ledger['committed_updates']+1,'loss':float(loss.detach()),'gradient_norm':float(norm),
                 'parameter_change_norm':changed,'input_sha256':digest(b.input_ids.cpu().tolist()),
                 'supervised_targets':n,'query_targets':qn,'compute_seconds':time.perf_counter()-started}
            accounting={'input_tokens':b.input_tokens,'processed_tensor_tokens':b.input_ids.numel(),
                        'padding_tokens':b.input_ids.numel()-b.input_tokens,'supervised_target_tokens':n,'query_target_tokens':qn}
            sampler.resolve(True,ledger,accounting);history.append(row);save()
        except Exception:
            # Includes optimizer mutations and interrupted/failed checkpoint writes.
            # The previous complete atomic envelope remains authoritative.
            sampler.pending=None;restore();optimizer.zero_grad(set_to_none=True)
            raise
    final=evaluate(model,validation,config['batch'])
    result={'status':'COMPLETE' if ledger['committed_updates']==config['updates'] else 'INCOMPLETE',
            'ledger':ledger,'initial_evaluation':initial,'validation':final,'parameters':parameters(model),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'checkpoint':{'sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size},
            'history':history,'hypotheses':'NOT_EVALUATED'}
    write(out/'metrics.json',result)
    # Derived log regenerated from the committed checkpoint, never authoritative.
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    return result
