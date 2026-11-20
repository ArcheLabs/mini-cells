# D2 · Addressing

Define key a = normalize(A h_write), query b = normalize(A h_read),
value v = V h_value, and recall r = M_fast(b). Encode packets independently
of source session state. A symmetric encoder supplies a positive control
with identical key and query packets. Natural-language paraphrases form
a separate generalization stratum.

Measure address discrimination by matched cosine minus the largest
nonmatching cosine. Measure retrieval by candidate MRR, Recall@1 and
correct-minus-best-wrong score margin. Measure language use separately by
target NLL, rank and exact generation. Include hard negatives, unknown
queries and a rejection threshold fitted only on validation. Compare
correct, donor and empty states at equal update counts where applicable.
A lower NLL relative to no memory alone cannot identify content-specific use.
