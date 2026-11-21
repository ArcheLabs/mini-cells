"""Shared MoM cache operations. Query branches never commit their state."""
import hashlib
import io


def clone_tree(x, torch, device):
    if torch.is_tensor(x):
        return x.detach().to(device).clone()
    if isinstance(x, dict):
        return {k:clone_tree(v,torch,device) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return type(x)(clone_tree(v,torch,device) for v in x)
    if x is None or isinstance(x,(bool,int)):
        return x
    raise TypeError(type(x))


def export(cache, torch):
    return dict(layers=clone_tree(cache.to_legacy_cache(),torch,'cpu'),
                seen_tokens=cache.get_seq_length())


def fingerprint(state, torch):
    buffer=io.BytesIO()
    torch.save(state,buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def query_copy(cache, torch, restore):
    saved=export(cache,torch)
    return restore(type(cache),clone_tree(saved['layers'],torch,'cuda'),saved['seen_tokens'])
