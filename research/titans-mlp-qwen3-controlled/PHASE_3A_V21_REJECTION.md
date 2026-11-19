# Rejection calibration correction

The previous 64 nominal negatives included repeated unknown-relation
queries. The corrected generator emits each missing relation once per
known entity and uses separate unknown entity IDs. It checks rendered
uniqueness for A1, A2 and B. Complementary relation profiles keep each
validation partition and each negative type balanced. Checkpoint selection
and threshold calibration use disjoint entity sets.

The calibration gallery is the calibration partition's A1 write bank.
One threshold per learned method uses pooled unknown scores across A1,
A2 and B. Reports include FPR by family and negative class, known-query
coverage, and Recall@1 conditional on acceptance. If no known query is
accepted, selective Recall@1 is null. Existing retrieval figures using
the combined validation gallery are descriptive, separate from both
selection and rejection calibration.

Proposed independent Family C criteria for owner review: FPR at most 5%,
known coverage at least 80%, and selective Recall@1 at least 90%. These
are proposed operating targets and have not been frozen. The calibration
threshold must be applied unchanged to the independent assessment with
the same candidate-count protocol. Formal Test remains sealed.
