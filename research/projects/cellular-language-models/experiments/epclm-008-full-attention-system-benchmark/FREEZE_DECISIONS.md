# EPCLM-008 decisions before formal execution

## Backbone budget

Use the M2 architecture and the M1 WikiText-103 tokenizer. For resource
planning, evaluate the two historical EPCLM-004 budget points separately:
20M and 100M supervised tokens per backbone per seed. With two backbones
and three seeds, these imply 120M and 600M supervised tokens, respectively.
Both projections exclude the three 1024-update MAG runs and evaluation.
The original 20M budget was a screening budget. The original 100M budget
was a confirmation budget. Choosing either for EPCLM-008 requires an
explicit decision after profiling. The formal budget remains unset.

## Exact History interpretation

The current protocol candidate measures zero-shot exact recovery. Keep
Exact History examples out of backbone and MAG training under that
interpretation. An alternative is to include a disjoint synthetic
instruction-training split for all arms, which changes the training mix
and requires a separately specified paired budget. The generator and
scoring contract in EXACT_HISTORY_PROTOCOL.md apply to either design.
Choose the interpretation before freezing prompts, payload lengths,
split counts and decoding limits.

## Engineering evidence and next measurements

The Tesla T4 fixture passed. Every report listed in its GitHub publication
manifest was verified by byte length and SHA256. The fixture performed
zero official updates and did not benchmark the training loop or transfer
durable model weights. Measure full-size FP32 E0, E3 and MAG step time,
nonpadding targets and CUDA peaks before choosing a formal session plan.
Use fresh engineering checkpoints for interrupted-versus-uninterrupted
replay and durable storage readback. Keep the formal Test sealed.
