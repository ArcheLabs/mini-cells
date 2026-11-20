# MA-001.5 — Missing-fact evidence probes

Continual Learning development experiment. Nine authored pairs remove the
queried actor, attribute or relation while preserving topic and claim. The
absent premise is insufficient evidence, not necessarily a contradiction.
Use the same pinned DeBERTa NLI as MA-001.3. Save three-way probabilities for
both members, strict pair ordering, and unsupported entailment argmax by
stratum. Argmax is a diagnostic, not a routing threshold. No model training
or threshold fitting. This isolates evidence scoring from retrieval and
does not test autonomous parsing or MA-002 generalization.

Run the Kaggle Python Script `kaggle_semantic.py` with CPU, Internet and
the existing GITHUB_TOKEN secret. Scoring results remain pending until run.
