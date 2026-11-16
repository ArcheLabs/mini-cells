"""Canonical relation templates. Family C is reserved for formal Test."""
from dataclasses import dataclass

from phase3a import RELATIONS

NOUNS = ('project', 'group', 'assignment', 'asset', 'person', 'resource', 'asset', 'location')
DIRECT = ('work on', 'belong to', 'assigned to', 'operate', 'report to', 'use', 'manage', 'located in')


@dataclass(frozen=True)
class Template:
    relation: str
    role: str
    family: str
    text: str
    anchor: str

    @property
    def template_id(self):
        return f'{self.role}-{RELATIONS.index(self.relation)}-{self.family}'

    def render(self, entity, value):
        # The sentinel fields delimit spans before rendering any identifier.
        text = self.text.replace('{anchor}', self.anchor)
        start = text.index('{entity}')
        text = text.replace('{entity}', entity)
        anchor = text.index(self.anchor, start + len(entity))
        spans = [(start, start + len(entity)), (anchor, anchor + len(self.anchor))]
        if '{value}' in text:
            start = text.index('{value}')
            spans.append((start, start + len(value)))
            text = text.replace('{value}', value)
        return text, spans


def registry():
    result = {}
    for index, relation in enumerate(RELATIONS):
        noun, direct = NOUNS[index], DIRECT[index]
        auxiliary = 'is' if relation.startswith('is ') else 'does'
        writes = ('{entity} {anchor} {value}.',
                  'The record states that {entity} {anchor} {value}.',
                  'It is {entity} that {anchor} {value}.',
                  '{entity}, which {anchor} {value}, is listed in the record.')
        queries = ('Which ' + noun + ' ' + auxiliary + ' {entity} {anchor}?',
                   'Can you identify which ' + noun + ' {entity} {anchor}?',
                   'Identify the ' + noun + ' that {entity} {anchor}.',
                   'The requested ' + noun + ' is the one that {entity} {anchor}.')
        for i, family in enumerate(('A1', 'A2', 'B', 'C')):
            result['write', relation, family] = Template(relation, 'write', family, writes[i], relation)
            result['query', relation, family] = Template(
                relation, 'query', family, queries[i], direct if i == 0 else relation)
    return result


TEMPLATES = registry()
SPLIT_FAMILIES = {'train': ('A1', 'A2'), 'validation': ('A1', 'A2', 'B'),
                  'test': ('A1', 'A2', 'C')}


def view(row, role, family):
    entity, relation, value = row
    text, spans = TEMPLATES[role, relation, family].render(entity, value)
    return text, spans[:2]
