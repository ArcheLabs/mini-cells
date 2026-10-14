# Baseline fairness

All primary arms encode the same ten visible features at all N slots, own N*d persistent allocated state and N*d immutable encodings, use the same candidates, root information, two-coordinate output rule, maximum eight active sources/recipients and sixteen rounds. Common encoder/message/update/readout operators have identical initial weights for each seed where shapes match (component-specific RNG streams). Same optimizer, training seed, examples, final-checkpoint policy and update budgets. Differences in actual messages/update counts and latency are retained, not assumed equal.

B0: shared recurrent controller follows bounded input-dependent addresses; trajectory control replaces destination state using its initial encoding, without retaining a recipient's previously computed local state. Addressed input memory remains N*d. It is a deliberately strong state-addressing, no-persistent-cell-message-memory control; it is not a blind root-only RNN.
B1: fixed immutable-priority candidate ordering, no learned or runtime-state route; actual update hidden width is increased to compensate router capacity. No dummy unused parameters.
B2: truly trainable source/target bucket preferences, independent of computed runtime state. The biased ST estimator optimizes those preferences. Absolute-address bucket dependence may fail permutation; report it rather than exempting it.
C0: shared local state/candidate-state scoring MLP, shared messages and persistent receiver updates.
B3: centralized shared bilinear query controller with exactly the same source/candidate read permissions and persistent state, a different scoring family and much less score projection work. It cannot pool unrelated private states or scan N. Required before any positive H1 claim.

All parameters participate in task training. Exact counts, spread (<5%), state bytes, candidate/gather/message/update counts and full hard-forward latency are recorded. FLOPs estimates do not enforce latency equivalence: if C0 costs more, only quality gain can be reported. The shape of a local router is not a scientific advantage by definition; B3 may match or exceed it.

POOL is an additional strong global mean/root-pooling shortcut diagnostic with the same supervised budget. ALG is the exact public-rule executor for A with zero learned parameters; its CPU implementation has a distinct runtime scope. Neither is used to manufacture a favourable matched neural comparison.
