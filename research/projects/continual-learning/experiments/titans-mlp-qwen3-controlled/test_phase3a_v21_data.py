from phase3a_v21_data import audit, dataset, view


def test_combinations_balance_and_split_isolation():
    splits = dataset(42, (256, 64, 128))
    assert splits == dataset(42, (256, 64, 128))
    for name, rows in splits.items():
        metrics = audit(rows)
        assert set(metrics['relations'].values()) == {len(rows) // 8}
        assert len(metrics['combinations']) == min(70, len(rows) // 4)
        assert len(metrics['cooccurrences']) == 28
    for column in (0, 2):
        ids = [{r[column] for r in rows} for rows in splits.values()]
        assert len(set.union(*ids)) == sum(map(len, ids))


def test_template_families_and_causal_spans():
    for row in dataset(42, (4, 0, 0))['train']:
        seen = set()
        held_out = set()
        for role in ('write', 'query'):
            for held, collection in ((False, seen), (True, held_out)):
                for variant in (0, 1):
                    text, (entity, relation) = view(row, role, variant, held)
                    assert text[slice(*entity)] == row[0]
                    assert entity[1] < relation[0]
                    assert row[2] not in text[:relation[1]]
                    collection.add(text)
        assert len(seen) == len(held_out) == 4
        assert seen.isdisjoint(held_out)
