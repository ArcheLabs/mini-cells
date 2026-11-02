"""Analytic matmul decomposition; explicit extra memory scores, reads and GRU work."""
from bootstrap import old_flops,shared


def report(model,tokens=125,batch=1):
    # Same A2 projections/FFNs; no old X4C State module assumptions.
    d=model.dim;head=2*tokens*d*model.head.out_features
    attention=sum(8*tokens*d*d+4*d*shared.native._local_pairs(tokens,s.window) for s in model.stages)
    ffn=sum(6*tokens*d*s.ff.up.out_features for s in model.stages)
    chunks=(tokens+31)//32
    bridge=chunks*2*16*128*d
    memory_kv=chunks*3*4*16*d*d
    attention_read=3*4*tokens*16*d if model.read_attention else 0
    ffn_read=3*tokens*(4*d*d+4*16*d) if model.read_ffn else 0
    # GRU: input affine and hidden affine, 3 gates, 16 Cells per Chunk.
    write=chunks*16*6*128*(d+128)
    parts={'token_attention':attention,'ffn':ffn,'state_read_bridge':bridge,'state_read_kv':memory_kv,
           'state_read_attention_path':attention_read,'state_read_ffn_path':ffn_read,'state_write':write,'head':head}
    parts={k:int(v*batch) for k,v in parts.items()};total=sum(parts.values())
    return {'parts':parts,'forward_flops':total,'forward_flops_per_prediction_token':total/(tokens*batch),
            'parameter_count':sum(p.numel() for p in model.parameters()),'state_bytes_per_sample':8192,
            'convention':'MAC=2; matmul-heavy analytic estimate; includes final partial Chunk write; excludes normalization, gates, softmax and allocator traffic',
            'caveat':'local-pair Attention formula may undercount dense masked SDPA kernels; compare actual timings separately'}
