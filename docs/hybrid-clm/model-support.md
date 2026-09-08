# Model support

Support levels are explicit; inspection of a module does not imply that it can
be safely mutated.

| Family | Level | Notes |
|---|---|---|
| Granite MoE | **SUPPORTED** | Verified fused `gate_up_proj` + `down_proj`, inherited top-k router |
| Mixtral | UNSUPPORTED / planned | No v0.1 backend |
| Qwen MoE | UNSUPPORTED / planned | No v0.1 backend |
| DeepSeek MoE | UNSUPPORTED / planned | No v0.1 backend |
| Dense Transformer | UNSUPPORTED / research | No MoE placement contract |

Unknown architectures raise `UnsupportedArchitectureError`; there is no silent
best-effort mutation.
