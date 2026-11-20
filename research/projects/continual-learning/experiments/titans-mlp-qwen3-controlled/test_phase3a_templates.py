from phase3a_templates import SPLIT_FAMILIES, TEMPLATES


def test_registry_and_sealed_family():
    assert len(TEMPLATES) == len({t.template_id for t in TEMPLATES.values()}) == 64
    assert set(SPLIT_FAMILIES['train']) == {'A1', 'A2'}
    assert 'C' not in SPLIT_FAMILIES['validation']
    for template in TEMPLATES.values():
        text, spans = template.render('E123456', 'V543210')
        assert text[slice(*spans[0])] == 'E123456'
        assert text[slice(*spans[1])] == template.anchor
        assert spans[0][1] < spans[1][0]
        if template.role == 'write':
            assert spans[1][1] < spans[2][0]
            assert text[slice(*spans[2])] == 'V543210'
        else:
            assert 'V543210' not in text
