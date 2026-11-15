"""Explicit Titan decoder-layer bridge for the pinned upstream patch."""
from dataclasses import replace

from portable import _restore_memories, tree


def run_layers(layers, hidden, cell, session):
    """Stage a forward over Titan decoder layers without mutating the Cell."""
    from titan_llama import TitanLayerState

    session_offset, memory_offset = session.offsets(cell)
    _, tensors = tree(cell)
    memories = _restore_memories(cell, tensors, memory_offset)
    previous_kv = dict(session.attention_kv)
    next_kv = {}
    value_residual = None
    count = hidden.shape[1]
    for index, layer in enumerate(layers):
        state = TitanLayerState(memory=memories.get(index),
                                attention_kv=previous_kv.get(index), position=session_offset)
        hidden, state, value_residual = layer.forward_core(
            hidden, state, value_residual, memory_position=memory_offset)
        if layer.has_neural_memory:
            memories[index] = state.memory
        next_kv[index] = state.attention_kv
    return hidden, memories, replace(session, position=session_offset + count,
                                     attention_kv=tuple(next_kv.items()))
