# Experiment 018b — Feedback-Isolated Pressure Recruitment

## Question

Can a one-cell capability tissue be recruited by a local computational-pressure signal that is causally isolated from newborn feedback?

## Why 018b

Experiment 018 showed that conditional participation improved retained-language behavior, but the state-novelty gate lost selectivity during adaptation. The sensor observed the already-adapted parent state, creating a possible positive feedback path:

`recruitment -> newborn influence -> parent state deviation -> recruitment`.

018b removes that path rather than tuning the novelty threshold.

## Matched policies

Each replicate trains one Phase-1 Growing CLM and clones the exact checkpoint into two policies using the same REVERSE_INC corpus, schedule, conservative initial fork, newborn-only phenotype optimization, and local structural schedule.

- **N** — Experiment-018 recurrent-step state-novelty recruitment, capped to one newborn.
- **P** — feedback-isolated pressure recruitment, capped to one newborn.

Both policies may connect or prune edges touching the newborn tissue. Neither may allocate another cell. Experiment 017 already established that one newborn is sufficient for this skill; fixing cell count isolates the sensor.

## Pressure sensor

P runs an old-only shadow organism containing exactly the Phase-1 cells, phenotype snapshot, and graph. It receives the same token input but no newborn phenotype, edges, activity, or messages.

At recurrent step `t`, parent-cell pressure is:

`rho[p,t] = RMS(reaction[p,t] + old_diffusion[p,t])`.

A Phase-1 retained-language calibration estimates a recurrent-step × cell mean, scale, and 95th-percentile threshold. Newborn recruitment is:

`e = floor + (1-floor) * sigmoid(beta * (rho - threshold) / scale)`.

The pressure sensor has no task label and no trainable router.

## Hard causal invariant

For a fixed input and Phase-1 snapshot, the shadow pressure trace must be exactly invariant to:

- newborn phenotype changes;
- newborn incident edges;
- forcing recruitment on or off.

Thus there is no `newborn -> sensor` causal path by construction.

As in 018, recruitment controls both structural conductance and metabolic participation. `e=0` must recover exact Phase-1 logits and `e=1` must recover the ordinary static adapted organism.

## Pre-registered full signal

`FEEDBACK_ISOLATED_PRESSURE_RECRUITMENT_SIGNAL` requires all of:

- P skill improvement is positive and at least 80% of N in >=2/3 replicates;
- P donor TinyStories NLL <=1.10× Phase-1 in >=2/3;
- P retention improves over N in >=2/3;
- old phenotype max drift <=1e-6 in all replicates;
- exactly one newborn cell in all replicates;
- P skill/language recruitment ratio >=2.0 and absolute gap >=0.20 in >=2/3;
- forcing P recruitment off removes >=50% of skill improvement in >=2/3;
- forcing P recruitment on worsens TinyStories NLL by >=0.05 in >=2/3;
- runtime shadow-pressure feedback delta <=1e-7 in all replicates;
- newborn tissue causal fraction >=50% in >=2/3;
- newborn-only graft recovers >=50% of skill improvement while recipient TinyStories NLL remains <=1.10× in >=2/3.

Diagnostic statuses distinguish selective pressure recruitment without retention, retention without pressure selectivity, partial signal, and no signal.

## Scope

This is a controlled one-skill, same-genome, same-checkpoint sensor experiment. It does not establish an optimal pressure statistic, general task routing, arbitrary real-world capability transplantation, cross-genome compatibility, or production compute efficiency.
