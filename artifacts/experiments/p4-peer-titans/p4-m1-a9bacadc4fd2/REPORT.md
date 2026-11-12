# P4-M1.1 Tesla T4 smoke

Status PASS.
GPU Tesla T4, CUDA 13.0, driver 580.178.04.
Wall 9.77s, peak allocated 18,573,824 bytes.
Projected acquisition only 22.30 minutes for 20 mappings per configuration at 128 exposures, extrapolated from one eight-exposure episode per configuration.

| Geometry | Method | Bindings | Updates | Presentations | Acquisition seconds | Restore error |
|---|---|---:|---:|---:|---:|---:|
| gaussian | batch | 4 | 8 | 32 | 0.0238 | 0 |
| gaussian | sequential | 4 | 32 | 32 | 0.1171 | 0 |
| gaussian | chunk4 | 4 | 8 | 32 | 0.0247 | 0 |
| gaussian | chunk8 | 4 | 8 | 32 | 0.0269 | 0 |
| qr | batch | 4 | 8 | 32 | 0.0239 | 0 |
| gaussian | batch | 8 | 8 | 64 | 0.0236 | 0 |
| gaussian | sequential | 8 | 64 | 64 | 0.1826 | 0 |
| gaussian | chunk4 | 8 | 16 | 64 | 0.0473 | 0 |
| gaussian | chunk8 | 8 | 8 | 64 | 0.0238 | 0 |
| qr | batch | 8 | 8 | 64 | 0.0251 | 0 |
| gaussian | batch | 16 | 8 | 128 | 0.0247 | 0 |
| gaussian | sequential | 16 | 128 | 128 | 0.3792 | 0 |
| gaussian | chunk4 | 16 | 32 | 128 | 0.1007 | 0 |
| gaussian | chunk8 | 16 | 16 | 128 | 0.0512 | 0 |
| qr | batch | 16 | 8 | 128 | 0.0257 | 0 |
| gaussian | batch | 32 | 8 | 256 | 0.0276 | 0 |
| gaussian | sequential | 32 | 256 | 256 | 0.7247 | 0 |
| gaussian | chunk4 | 32 | 64 | 256 | 0.1901 | 0 |
| gaussian | chunk8 | 32 | 32 | 256 | 0.1003 | 0 |
| qr | batch | 32 | 8 | 256 | 0.0254 | 0 |
| gaussian | batch | 64 | 8 | 512 | 0.0263 | 0 |
| gaussian | sequential | 64 | 512 | 512 | 1.4285 | 0 |
| gaussian | chunk4 | 64 | 128 | 512 | 0.3713 | 0 |
| gaussian | chunk8 | 64 | 64 | 512 | 0.1870 | 0 |
