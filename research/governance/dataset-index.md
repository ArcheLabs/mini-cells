# Immutable research inputs

CPU preparation runs in Actions, using the existing pinned TinyStories loader and train-only tokenizer. No dataset or tokenizer download occurs on Kaggle.

Each snapshot includes `train.npy`, `validation.npy`, `tokenizer.json`, `data.json`, and `assets-manifest.json`. The snapshot name contains its manifest hash; it is never updated in place. Exact file hashes, text hashes, token hashes, vocabulary, train/validation split, dataset revision and preprocessing definition are verified before GPU work.

Dataset creation uses the existing Kaggle API credential in Actions. Public TinyStories needs no HF token. Private/gated future inputs need their own approved CPU preparation definition; do not introduce runtime secrets.

Large checkpoints stay in Kaggle outputs and separate Actions checkpoint artifacts. No checkpoint is committed to Git. A longer retention Dataset export can be added to a separately authorized storage workflow.
