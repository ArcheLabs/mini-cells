"""Isolated E1 reference with one chunk-local Qwen attention layer."""
from contextlib import contextmanager


class ChunkCache:
    """Expose only current-chunk KV to the selected native attention call."""

    def update(self, key, value, layer_idx, cache_kwargs=None):
        return key, value


@contextmanager
def chunk_local(model, index):
    """Temporarily restrict a single full-attention layer of a private model."""
    config = model.config
    if config.layer_types[index] != 'full_attention':
        raise ValueError('Select a full-attention layer')
    if config._attn_implementation != 'eager':
        raise ValueError('The prototype requires native eager attention')
    attention = model.model.layers[index].self_attn

    def local_inputs(module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        mask = kwargs.get('attention_mask')
        if mask is None:
            raise ValueError('Expected native eager causal mask')
        kwargs['attention_mask'] = mask[..., -hidden.shape[1]:]
        if kwargs.get('past_key_values') is not None:
            kwargs['past_key_values'] = ChunkCache()
        return args, kwargs

    handle = attention.register_forward_pre_hook(local_inputs, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
