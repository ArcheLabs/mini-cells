# Architecture audit

The existing app already had the pinned Backbone, native Cell ZIP and resumable
write flow, isolated recurrent reads, FusionReader and one generation loop
per entry point. The SDK moves the three shared runtime modules into an
installable package. Compatibility aliases preserve existing app and research
imports. Native generation now uses a single helper. The demo delegates writes,
queries, ZIP handling and model lifetime to the SDK.

The residual experiment's Shared Read Output Norm was computed within shared
Hidden State evolution across all Cells. The SDK applies the same read-vector
norm and aggregation to one Cell at a time. There was no independent product
Selector, Top-K policy or calibrated rejection threshold for this score.
The default Top-2 and optional threshold are explicit engineering choices.
ResidualTrace stays available in the research code and production scoring
does not invoke the diagnostic kernel. Independent scoring is a fresh read
path and its scientific effectiveness still requires separate measurement.
