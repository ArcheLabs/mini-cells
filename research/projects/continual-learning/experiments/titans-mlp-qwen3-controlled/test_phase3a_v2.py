from collections import Counter

import torch

from phase3a_v2 import Adapter, contrastive, dataset, view


def test_balanced_disjoint_facts_and_ordered_spans():
    splits = dataset(42, (256, 64, 128))
    assert [len(rows) for rows in splits.values()] == [1024, 256, 512]
    for rows in splits.values():
        assert set(Counter(e for e, _, _ in rows).values()) == {4}
        assert set(Counter(r for _, r, _ in rows).values()) == {len(rows) // 8}
        for row in rows:
            for kind in ('write', 'query', 'held_write', 'held_query'):
                text, spans = view(row, kind)
                assert [text[a:b] for a, b in spans] == list(row[:2])
                assert spans[0][1] <= spans[1][0]
                assert row[2] not in text[:spans[1][1]]
    for column in (0, 2):
        ids = [{row[column] for row in rows} for rows in splits.values()]
        assert len(set.union(*ids)) == sum(map(len, ids))


def test_adapter_gradients_and_unit_norm():
    torch.manual_seed(42)
    adapter = Adapter()
    assert adapter.query[0].weight.data_ptr() != adapter.key[0].weight.data_ptr()
    features = torch.randn(4, 2048)
    q, k = (adapter.encode(role, features) for role in ('query', 'key'))
    value = adapter.encode('value', torch.randn(4, 1024))
    for result in (q, k, value):
        assert result.shape == (4, 256)
        torch.testing.assert_close(result.norm(dim=1), torch.ones(4))
    (contrastive(q, k) + contrastive(value, value)).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in adapter.parameters())
