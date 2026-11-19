# Stage 5 Cell portability

The T4 launcher exports two logical Cells using the existing portable
codec for every layer/head shard. A fresh process reloads the pinned
Qwen and verifies adapter identity, tensor hashes and fixed-query outputs
without source caches. Queries begin at position zero in the fresh
session. The fixture checks single and joint read, fork isolation and
independent online writes. Raw Cell artifacts remain in Kaggle Output;
the verification report and log are published to GitHub.

This is an engineering fixture with independently initialized Stage 4
memories. T4 execution, learned retrieval quality and cross-machine
migration are unverified.
