"""Real tiny Qwen tests for the pinned eager backend."""
import pytest
import torch

from cell import Cell
from integration import Retrofit, RuntimeConfig


def test_native_parity_and_bounded_continuation():
    pytest.importorskip('transformers')
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'],
    )
    model = Qwen3_5ForCausalLM(config).eval()
    model.set_attn_implementation('eager')
    retrofit = Retrofit(model, RuntimeConfig(local_window=4))
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        original = model(input_ids=ids[:, :4], use_cache=True)
        local, _ = retrofit(model, None, input_ids=ids[:, :4], use_cache=True,
                            memory_enabled=False)
        torch.testing.assert_close(original.logits, local.logits)
        for name in ('recurrent_states', 'conv_states'):
            torch.testing.assert_close(getattr(original.past_key_values, name)[0],
                                       getattr(local.past_key_values, name)[0])
        for pos in (4, 5):
            local, _ = retrofit(model, None, input_ids=ids[:, pos:pos+1], use_cache=True,
                                memory_enabled=False, past_key_values=local.past_key_values,
                                position_ids=torch.tensor([[pos]]))
            assert local.past_key_values.get_seq_length() == 4
            assert local.logits.isfinite().all()
        cell = Cell.create(retrofit.memories, {k: k for k in
                           ('backbone', 'tokenizer', 'architecture', 'meta', 'layer_map')})
        attention = model.model.layers[1].self_attn
        calls = []
        handle = attention.q_proj.register_forward_hook(lambda *_: calls.append(1))
        try:
            output, updated = retrofit(model, cell, input_ids=ids[:, :2], use_cache=False)
        finally:
            handle.remove()
        assert calls == [1]
        assert updated.shards['1:0'].step == 2 and output.logits.isfinite().all()


def test_mask_membership_at_window_boundary():
    pytest.importorskip('transformers')
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'],
    )
    model = Qwen3_5ForCausalLM(config).eval()
    model.set_attn_implementation('eager')
    retrofit = Retrofit(model, RuntimeConfig(local_window=3))
    seen = []
    def observe(module, args, kwargs):
        seen.append(kwargs)
    attention = model.model.layers[1].self_attn
    # Existing hooks run before the retrofit's hook. Inspect the mask at the
    # native module boundary by wrapping its bound forward for this test.
    native = attention.forward
    def forward(*args, **kwargs):
        observe(attention, args, kwargs)
        return native(*args, **kwargs)
    attention.forward = forward
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    with torch.no_grad():
        result, _ = retrofit(model, None, input_ids=ids, use_cache=True, memory_enabled=False)
        actual = seen[-1]['attention_mask'][0, 0].eq(0)
        expected = torch.tensor([[max(0, i-2) <= j <= i for j in range(5)] for i in range(5)])
        torch.testing.assert_close(actual, expected)
        assert result.past_key_values.get_seq_length() == 3
