# Hugging Face publication setup

HybridCLM artifact publication is a separate Kaggle → Hugging Face path. It
does not require GitHub write credentials and never changes formal seed state.

One-time setup:

1. Create a fine-grained Hugging Face write token scoped to the ArcheLabs
   repositories or organization that will host the Model, Space, and
   Collection.
2. Add the token to Kaggle Secrets under the exact name `HF_TOKEN`.
3. Confirm that the Kaggle account can create or update the configured Model
   repository, Space, and Collection.

The release notebook reads the secret through Kaggle's secrets client, never
prints or writes the token, and performs identity, artifact, and metric
validation before any upload. Existing repositories are treated as
idempotent only when their release manifest matches; incompatible content
aborts the publication.

Formal seeds remain `RESERVED_UNTOUCHED`. The first preview uses engineering
seed `26090501` only.
