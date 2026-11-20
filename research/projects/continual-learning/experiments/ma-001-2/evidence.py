"""Evidence routing prototype. Discovery scores are not evidence labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    subject: str
    relation: str
    object: str


def assess(request, claims):
    """Exact normalized claim coverage; inputs are structured, not inferred."""
    if request in claims:
        return 3
    if any(c.subject == request.subject for c in claims):
        return 1
    return 0


def route(requests, candidates):
    """Each candidate is (event_id, cell_id, claims). Require all query facets."""
    primary, auxiliary, uncovered = set(), set(), []
    audit = []
    for request in requests:
        supported = False
        for event, cell, claims in candidates:
            grade = assess(request, claims)
            audit.append(dict(event_id=event, cell_id=cell, facet=request.__dict__, grade=grade))
            if grade == 3:
                primary.add(cell)
                supported = True
            elif grade:
                auxiliary.add(cell)
        if not supported:
            uncovered.append(request.__dict__)
    status = ('SUPPORTED' if requests and not uncovered else
              'RELATED' if primary or auxiliary else 'NO_MATCH')
    return dict(status=status, primary_cell_ids=sorted(primary) if status=='SUPPORTED' else [],
                auxiliary_cell_ids=sorted(auxiliary-primary if status=='SUPPORTED' else auxiliary|primary),
                uncovered=uncovered, evidence=audit)
