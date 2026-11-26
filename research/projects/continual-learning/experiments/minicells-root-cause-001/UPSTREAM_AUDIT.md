# Upstream audit · 2026-11-26

## Expert order

The standalone OpenSparseLLMs/MoM head audited at
`033057c2f80941cb0b70ef62a99dd7ddfc94bf52` still uses an unstable sort in
both the varlen `mom/layers/mom.py` implementation and the original
`mom_naive/layers/mom_gated_deltanet.py` path. The varlen integration moved
the pinned implementation into `mom_naive`. It also added a distinct
`MomAttention` implementation. [Standalone varlen change](https://github.com/OpenSparseLLMs/MoM/commit/c05f15845abb6aa41b8aea2e271cebab7ed1641a)
· [Varlen sort at audited head](https://github.com/OpenSparseLLMs/MoM/blob/033057c2f80941cb0b70ef62a99dd7ddfc94bf52/mom/layers/mom.py#L176)
· [Original sort at audited head](https://github.com/OpenSparseLLMs/MoM/blob/033057c2f80941cb0b70ef62a99dd7ddfc94bf52/mom_naive/layers/mom_gated_deltanet.py#L91).

FLA integrated its `MomAttention` and `MomConfig` in
`aefb21b30722b9846b4c3b5536112a47a26b2d88` (PR #442), included in v0.3.1.
That version still used the unstable sort. The later
`7a78a662c525a7ac10e5b6b4cdd54d275d141cf2` (PR #1628) uses stable sorting
for equal batch/expert keys. The equivalent one-line ordering change
addresses the same operation in the pinned MoM packer. The integrated
FLA model uses the `mom` model type and different packing and cache
interfaces. Using it with the current `mom_gated_deltanet` checkpoint
would require explicit compatibility validation.
[FLA integration](https://github.com/fla-org/flash-linear-attention/commit/aefb21b30722b9846b4c3b5536112a47a26b2d88)
· [v0.3.1 model config](https://github.com/fla-org/flash-linear-attention/blob/v0.3.1/fla/models/mom/configuration_mom.py)
· [Stable-order patch](https://github.com/fla-org/flash-linear-attention/commit/7a78a662c525a7ac10e5b6b4cdd54d275d141cf2).

Local CPU replay used the actual `transform` bodies, with identical input
positions and expert assignments in ten fixtures from lengths 5, 13, 20,
21 and 64. The newer FLA packer's validity mask excludes its padding values.

| Source | Chronological packing |
| --- | ---: |
| Pinned standalone MoM | 4/10 |
| FLA parent of #1628 | 4/10 |
| FLA #1628 merge commit | 10/10 |

PR #1628 reports 8 passed and 11 skipped on NVIDIA B300. It tests causality,
prefix consistency and the 64/65-token kernel boundary. The independently
reproduced FP16 generation discrepancy remained above tolerance after
stable sorting, so generation retained its skip. The PR's controlled
router-logit study links part of that discrepancy to near-tied expert
scores. It does not validate the pinned checkpoint, Tesla T4, or the
complete SDK stream. [Upstream test log and method](https://github.com/fla-org/flash-linear-attention/pull/1628).

## ShortConvolution

Commit `6146a30b9d8817440c8467c7fa43413c68b9cd87` (PR #535) introduces
chunked prefill and forwards prior convolution cache as `initial_state`
to the Triton multi-token call. This is the change relevant to our 5+7
probe. PR #568 (`b1d766994c7ac53c4d0a53a1b6e8f94de363abe1`) follows with
stateful convolution kernel and backward changes, including comparison
against a Torch oracle. v0.3.1 contains the cache-aware dispatch. Independent
validation on T4 with the pinned MiniCells stack remains pending.
[Chunk prefill change](https://github.com/fla-org/flash-linear-attention/commit/6146a30b9d8817440c8467c7fa43413c68b9cd87)
· [Cached convolution follow-up](https://github.com/fla-org/flash-linear-attention/commit/b1d766994c7ac53c4d0a53a1b6e8f94de363abe1)
· [v0.3.1 ShortConvolution](https://github.com/fla-org/flash-linear-attention/blob/v0.3.1/fla/modules/convolution.py#L903-L914).

### Single-token Triton weight layout

The pinned ShortConvolution passes its Conv1d weight with shape
`[D, 1, W]` to the single-token Triton fallback. Its update wrapper
takes the kernel width from `weight.shape[1]`, so it dispatches with
width 1. The update kernel uses that width for both the weight and cache
strides and writes back into the supplied cache. The multi-token path
flattens the weight to `[D, W]` and has a separate missing-history issue.
[Single-token dispatch](https://github.com/fla-org/flash-linear-attention/blob/a46f204c1c7ac5c70ee99168adef7682e6fbf2a3/fla/modules/convolution.py#L636-L652)
· [Width and layout in update wrapper](https://github.com/fla-org/flash-linear-attention/blob/a46f204c1c7ac5c70ee99168adef7682e6fbf2a3/fla/modules/convolution.py#L335-L367).

A CPU index-level reconstruction used the A.1 12-token, 16-channel input,
the four fixed weights, and the pinned dispatch's effective width and
flattened addressing. It reproduced all six reported T4 maximum errors
across the three split schedules, including zero final-cache error for
the two-chunk schedule. This supports separate diagnoses for the
single-token and multi-token paths.

## Delta recurrence and next gate

Phase A.1 measured all fused-recurrent Delta schedule differences within
0.002 absolute/relative tolerance. The Phase A.2 trace checks which
quantities diverge with the pinned 1.3B weights. Its full-model short
fixture still exercises fused recurrence. Any controlled sort or
convolution intervention will need an independent comparison against
that baseline. The Phase A gate remains closed.
