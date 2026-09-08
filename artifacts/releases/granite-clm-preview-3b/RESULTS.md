# Granite-CLM-Preview-3B Release

Status: **PASS**

> Release-engineering artifact; not a new continual-learning or safe-evolution decision.

## Source and structure

- Source: `ibm-granite/granite-3.1-3b-a800m-base`
- Revision: `d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a`
- Hidden layers: **32**
- Local experts: **40**
- Experts per token: **8**
- Safetensors bytes: **6,597,612,952**
- Tensor records: **290**
- Expert address records: **2560**

## Deterministic baseline controls

- Same-instance repeat parity: **PASS**
- Independent source reload parity: **PASS**

## CLM materialized parity

- Max absolute logit error: `0.0`
- Max absolute router error: `0.0`
- Router outputs compared: **96**
- Logit argmax identity: **True**
- Router Top-K identity: **True**
- Greedy token identity: **True**

## Runtime

- Device: `cuda:0`
- GPU: `Tesla T4`
- Visible CUDA devices: **2**
- Execution policy: `single_gpu_deterministic_reference_parity`
- Transformers: `4.46.0`
- Dtype: `torch.float16`
- End-to-end conversion/parity seconds: `232.632`

Published to `archelabs-org/native-clm-v0/granite-clm-preview-3b`.
