# Mutation format

Public mutation artifacts are directories containing:

```text
mutation.safetensors
cell_config.json
manifest.json
provenance.json
evaluation.json
```

`cell_config.json` is schema version 1 and records mutation type, backend,
placements, default alpha, and tensor target records. `manifest.json` pins the
base model/revision, architecture hash, tensor shapes/dtype, and the SHA256 of
the safetensors file. Provenance records source commit, protocol, dataset
identity/hash, seed, optimizer, steps, parameter count, creation time, and
engineering/formal status. `evaluation.json` contains publication metrics and
is not itself a scientific protocol.
