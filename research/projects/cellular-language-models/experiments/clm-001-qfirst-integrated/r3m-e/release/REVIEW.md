# R3m-E exploratory release

User-supplied cloud logs establish the completed engineering relay at source
`53fcde98c872324e70d36741742ff4862e3c61ba`, config
`3d641b26d4b70a5aabfd315bd35ac015652cea359bb234fb6b4b72d8b2ffbd99`.
The original extracted JSON reports and raw-log SHA-256/size are retained in `gate2/`. Reconstructing the uploaded review and single-file manifest from the supplied JSON exactly reproduces the pinned HF manifest SHA-256; `HF_REVIEW_MANIFEST.json` retains that independent byte-level check.
No GPU check or historical sampling replay was repeated to register this evidence.

Session A completed actual Proposal, ten-step Shadow and forced rejection/control;
rollback model and optimizer maximum difference was zero. Its real candidate was
effective and rejected. This is an engineering outcome, not a scientific result.
Session B restored relative step 4 and completed the remaining seven main/seven
Shadow updates in a distinct container. Model/optimizer maximum difference was zero,
with exact RNG/cursor/identity and unchanged transaction outcome. Private HF review
is pinned at revision `116af6327e5e083f0cac451a40bb61dff55122e0`.

The user requested the next step after the Gate 2 PASS report and the explicit
proposal to release four arms × three seeds × 600 continuation updates. This release
authorizes that exploratory development matrix and a user-launched Kaggle run.
It does not authorize Confirmation, original R3m registration, data re-selection,
a larger matrix or a change to model, training objective, task allocation, seeds,
Trust Region/Shadow acceptance, evaluation data or scientific budget.

The original engineering-only frozen config remains intact. The separate released
config pins its predecessor and the completed cloud proof. Launch reads that proof
from its immutable private HF revision and checks its bytes; it does not rerun
GPU Smoke or a test suite. The scientific config content is compared to its
predecessor with only release/status/authorization/execution metadata excluded.

Two independent worker processes use two visible GPUs and the existing R3m engine.
Candidate durability before scale scanning uses the existing R3i request/ack
protocol; only the coordinator writes to HF. Routine snapshots are throttled to
180 seconds; mandatory candidate and terminal persistence bypasses that interval.
Network I/O follows a brief locked local snapshot, so it does not hold the worker
checkpoint lock. There is no process runtime termination cap.

Every arm must finish before the terminal analysis stage and paired report barrier.
Reports include all twelve jobs. Missing jobs remain partial/inconclusive.
Signed per-Cell gradients use the exact parameter layout, including zero gradients
for unvisited parameters. Final GitHub publication reuses the existing current-main,
metadata-only publisher and remote commit verification, after training rather than
as a costly pre-training handshake. Weights remain in private HF and Kaggle Output.

CPU and simulated HF validation are engineering evidence only. The full exploratory
matrix has not been launched by Codex. Full runtime and final scientific outcomes
remain unmeasured. Reused historical Story Development remains exploratory data;
original R3m global isolation remains BLOCKED and Confirmation remains SEALED.
