"""Small synthetic candidate gates; no optimizer or corpus."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from mamba3_jax import Mamba3Config, Mamba3ForCausalLM


def check_mode(mimo=False):
    jax.config.update('jax_default_matmul_precision', 'highest')
    cfg=Mamba3Config(vocab_size=64,hidden_size=32,state_size=8,head_dim=8,
                     num_hidden_layers=1,chunk_size=4,is_mimo=mimo,mimo_rank=4)
    model=Mamba3ForCausalLM(cfg,rngs=nnx.Rngs(18001))
    x=jnp.asarray(np.random.default_rng(18001).integers(0,64,(2,9)),jnp.int32)
    metrics={}
    def equal(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all(),name+' nonfinite'
        np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-4,err_msg=name)
        metrics[name]=float(np.max(np.abs(a-b)))
    @nnx.jit
    def forward(m,ids):return m(ids)
    full=forward(model,x); equal('jit_eager',full['logits'],model(x)['logits'])
    for name,sizes in [('chunk',[4,4,1]),('decode',[4,1,1,1,1,1])]:
        cache=None;outputs=[];offset=0
        for size in sizes:
            out=model(x[:,offset:offset+size],cache=cache);cache=out['cache'];outputs.append(out['logits']);offset+=size
        equal(name,full['logits'],jnp.concatenate(outputs,axis=1))
        for i,(a,b) in enumerate(zip(jax.tree.leaves(full['cache']),jax.tree.leaves(cache))):equal(name+'_state_'+str(i),a,b)
    prefix=model(x[:,:4]);saved=jax.tree.map(lambda a:jnp.array(a),prefix['cache'])
    next_a=model(x[:,4:5],cache=prefix['cache']);next_b=model(x[:,4:5],cache=saved)
    equal('memory_restore',next_a['logits'],next_b['logits'])
    equal('reset',full['logits'],model(x)['logits'])
    equal('batch_isolation',full['logits'],jnp.concatenate([model(x[i:i+1])['logits'] for i in range(2)]))
    changed=x.at[:,5:].set((x[:,5:]+7)%64)
    equal('causality',full['logits'][:,:5],model(changed)['logits'][:,:5])
    loss,grads=nnx.value_and_grad(lambda m:m(x,labels=x)['loss'])(model)
    leaves=jax.tree.leaves(grads); arrays=[np.asarray(a) for a in leaves]
    assert arrays and all(np.isfinite(a).all() for a in arrays),'nonfinite gradient'
    norm=float(np.sqrt(sum(float(np.sum(a.astype(np.float64)**2)) for a in arrays)))
    assert np.isfinite(float(loss)) and norm>0
    # Cache size independent of processed prefix length; record actual shapes/dtypes.
    short=jax.tree.leaves(prefix['cache']);long=jax.tree.leaves(full['cache'])
    assert [(a.shape,a.dtype) for a in short]==[(a.shape,a.dtype) for a in long]
    return {'status':'PASS','config':dataclasses.asdict(cfg),'errors':metrics,'loss':float(loss),
            'gradient_norm':norm,'optimizer_updates':0,'dtype':'float32','matmul_precision':'highest',
            'cache':[{'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.size*a.dtype.itemsize} for a in long]}
