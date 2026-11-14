"""Research-only segment lifecycle for the pinned Qwen dynamic cache."""

class SegmentState:
    """Own local KV and pending writes separately from committed Cells."""

    def __init__(self, cache, layers, commit, length=512):
        if length < 1 or not layers or len(set(layers)) != len(layers):
            raise ValueError('Invalid segment configuration')
        self.cache = cache
        self.layers = tuple(layers)
        self.commit = commit
        self.length = length
        self.position = 0
        self.pending_tokens = 0

    def advance(self, tokens):
        if not 0 < tokens <= self.length - self.pending_tokens:
            raise ValueError('Split calls at the segment boundary')
        self.position += tokens
        self.pending_tokens += tokens
        if self.pending_tokens == self.length:
            self.flush_segment()

    def flush_segment(self):
        if not self.pending_tokens:
            return
        # The caller builds all new shards before publishing their state.
        self.commit()
        for index in self.layers:
            self.cache.key_cache[index] = None
            self.cache.value_cache[index] = None
        self.pending_tokens = 0
