"""Reproduce legacy matmul estimates; separately account for added state/router paths."""
from common import native
from cell import SelectiveCellFFN


def report(model,tokens=125,batch=1):
    d=model.dim;parts={'attention':0,'dense_ffn':0,'cell_ffn':0,'router':0,'state_read':0,'state_write':0,'head':2*tokens*d*model.head.out_features}
    for i,s in enumerate(model.stages):
        attn=8*tokens*d*d+4*d*native._local_pairs(tokens,s.window)
        for j in range(s.iterations):
            if model.attention_masks[i][j]:parts['attention']+=attn
            if not model.ffn_masks[i][j]:continue
            if isinstance(s.ff,SelectiveCellFFN):
                f=s.ff;hidden=f.shared.up.out_features;rank=f.down.shape[-1]
                parts['cell_ffn']+=2*tokens*(6*d*hidden+4*d*rank+(2*128*d if f.stateful else 0))
                parts['router']+=2*tokens*(d*f.router.rank+16*f.router.rank)+(2*((tokens+31)//32)*16*128*f.router.rank if f.stateful else 0)
            else:parts['dense_ffn']+=6*tokens*d*s.ff.up.out_features
    if model.persistent:
        p=model.persistent;chunks=(tokens+31)//32;rank=p.query.out_features
        # Read: Q/out per token; K/V over 16 slots per Chunk; QK and AV products.
        parts['state_read']=2*tokens*(d*rank+rank*d+d+2*16*rank)+chunks*4*16*128*rank
        writes=tokens//32;factor=16 if p.kind=='cellular' else 1
        parts['state_write']=writes*(2*p.summary.weight.numel()+factor*2*sum(m.weight.numel() for m in (p.write_in,p.write_value,p.write_gate)))
    parts={k:int(v*batch) for k,v in parts.items()};forward=sum(parts.values())
    return {'parts':parts,'forward_flops':forward,'backward_flops_estimate':2*forward,'training_flops_estimate':3*forward,
            'forward_flops_per_prediction_token':forward/(batch*tokens),'sequence_length':tokens,'batch':batch,
            'convention':'MAC=2; legacy matmul-heavy estimate; backward=2x forward is an estimate, not measured autograd work',
            'excluded':'normalization, RoPE, activations, softmax, top-k, aggregation, optimizer and device traffic; not a complete scalar-operation FLOPs standard; local-pair formulas can undercount dense masked SDPA backend work',
            'state_read_cost_note':'parallel Chunk estimate; token-wise runtime recomputes K/V projections and may cost more',
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'relative_parameters_to_x4':sum(p.numel() for p in model.parameters())/1893912-1,
            'parameter_matched_within_5pct':abs(sum(p.numel() for p in model.parameters())/1893912-1)<=.05}
