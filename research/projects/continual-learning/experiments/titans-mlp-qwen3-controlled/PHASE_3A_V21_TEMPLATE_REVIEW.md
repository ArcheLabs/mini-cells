# Template semantic review

Codex initial review: PASS for the rendered examples. Human review: pending.
The 64 rendered cases with canonical relation, template ID, role, family,
character offsets and pinned-tokenizer offsets are in
[phase3a_v21_templates.json](phase3a_v21_templates.json).

| Canonical relation | Query target | A1 anchor | Review |
|---|---|---|---|
| works on | project | work on | Preserves the object of the work relation |
| belongs to | group | belong to | Preserves membership |
| is assigned to | assignment | assigned to | Retains passive interpretation with auxiliary is |
| operates | asset | operate | Preserves operation target |
| reports to | person | report to | Preserves reporting direction |
| uses | resource | use | Preserves the object used |
| manages | asset | manage | Preserves management direction |
| is located in | location | located in | Retains the containing location |

A1 applies do-support or inversion. A2, B and C preserve the canonical
predicate in embedded or relative clauses. B's cleft and C's appositive
retain the same subject and object. Their surrounding record language
adds a framing context while the lookup target remains the canonical fact.
C was rendered for template audit using placeholder identifiers only.
Training selects A1/A2. Validation includes B. Formal Test remains sealed.
