# Oracle trust Gate 0/1

## Definition
For a captured pre-action state from theta_k, execute the alternative once, then repeatedly choose from the same frozen epsilon-Boltzmann policy on the **new** state. Return terminal teacher-forced CE plus lambda times remaining non-STOP calls/B. The sunk prefix is excluded. STOP directly returns head CE; budget end is forced and has no phantom STOP action.

A deep-copied read-only model, policy/registry/config hashes and detached captured memory bind every sample to theta_k and pi_k before its optimizer update. The online model is never overwritten for candidate evaluation. Labels become invalid after a registry or model version update. No cross-step replay buffer exists.

At most two token-step coordinates are chosen independently of labels per optimizer step. Capture may yield fewer if STOP occurred earlier. For each: actual action and legal STOP are included with probability one; up to two other legal candidates are sampled uniformly without replacement with conditional inclusion probability m/N. The documented loss is self-normalized inverse-inclusion Huber, not an assertion of an unbiased full-action estimator.

Two suffix draws share **underlying uniform variates indexed by absolute depth**, then independently inverse-CDF each new distribution. They do not share selected actions or consume one candidate's branch RNG in another candidate. Live/proposal/suffix/shadow/evaluation generators are distinct and checkpointed.

## CPU contracts
The test suite checks all legal short-world actions including CTX/STOP, exact enumeration vs MC with 5 standard errors plus numerical tolerance, Bellman decomposition, immediate STOP CE, self-repetition/forced termination, finite-difference continuous directions, fixed-replay full/decode causality, and a real toy Cell/Oracle state-dependent ranking learned by the Q head. A homogeneous-Cell/uniform-suffix control yields equal exact costs, not invented specialization.

Negative tests mutate policy, weights, registry, state, target or label payload, attempt illegal reserved/early STOP, and inject accepted transaction failure. They fail closed. Snapshot candidate work is independent of the live RNG. Q supervision updates Connection only; continuous task gradients come from actual live CE.

Every candidate stores sample costs/variance, action set, actual/alternative action, conditional proposal probability, target/state/model/policy/registry hashes, suffix seed/uniform hash, full completion trace and its hash. Detailed records remain in Kaggle Output, while GitHub receives aggregate metadata only.

CPU success establishes implementation consistency in tested small worlds. It does **not** establish real-language Q calibration, variance, generalization, better routing or speed. The separate frozen-model GPU Pilot is still required and unrun.
