# Safe Model Evolution

This project studies controlled, auditable modifications to pretrained models,
including mutation, compression, composition, and rollback. Its experiments
preserve the model lineage and evidence needed to assess both capability and
safety changes.

## Contents

- [`context/`](context/) preserves the original stage research records.
- [`experiments/`](experiments/) contains experiment-specific protocols,
  implementations, reports, and available results.

## Reading results

Use each experiment's frozen protocol and provenance records to interpret its
scope. A passing engineering check does not establish that safe model evolution
has been solved generally.
