# Historical evaluation audit

The source baseline is the local clean Layer Location commit `0d7d132`.
Its E3 resets the Cell at the question segment but leaves query writes
enabled. Its evaluation scores argmax over teacher-forced answers in a
single segment, allowing earlier answers to enter later query contexts.
Its synthetic unknown entity includes the literal suffix `absent`.
The publication allowlist contains validation and training metrics but
excludes checkpoint files.

The new query helper restores a copy of the historical state independently
for each prediction, with writes off by default and no retained Qwen cache.
Generation starts with the prompt alone.
