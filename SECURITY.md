# Security policy

Report security issues privately to the repository maintainers rather than
opening a public issue with exploit details. Mutation artifacts are treated as
untrusted input: v0.1 accepts safetensors plus JSON metadata, validates hashes,
shapes, pinned revisions, and architecture signatures, and does not deserialize
arbitrary pickle files.

Do not attach an artifact whose base model or revision is not independently
verified. Use the documented rollback check after an attach/detach lifecycle.
