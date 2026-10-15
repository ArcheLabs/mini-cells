# R2 change log

New independent R2 package/config/Notebook/tests/CI. STOP-anchored relative labels,
HT regression, confidence-filtered ranking, actual-visit reservoirs, grouped clipping,
shared scheduled exploration, typed candidate failures and diagnostics, transactional
checkpoint/event recovery, independent worker logs, partial aggregation and allowlisted
publication. R1 files and frozen outputs are unchanged. GPU results are not fabricated.

## Pilot fault-injection repair

The fixed J0 Shadow Path has no scorer policy gradient. Pilot now hooks all private
trainable tensors and verifies a real backward hook fired, including the J0 critic.
Failed workers write typed Pilot failure reports; parent collection preserves sibling
successes instead of reading a missing success file. The training algorithm/configuration
and R1 evidence remain unchanged.
