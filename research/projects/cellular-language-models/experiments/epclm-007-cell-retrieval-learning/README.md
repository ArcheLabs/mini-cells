# EPCLM-007 Cell Retrieval Learning Validation

Run `OMP_NUM_THREADS=1 python preflight007.py /tmp/epclm007-new` in the existing PyTorch environment. Gate 0 checks shared Key and Value symbol coverage, disjoint pair partitions, a sealed Test namespace and exact oracle labels. The 1/8 reference is uniform random candidate selection. The split partition is deterministic and constrains admissible associations. Future learned screening must measure a no-Cell control rather than assume all controls have exactly uniform performance.

Gate 1 uses eight orthogonal keys, independent nonidentity target assignment, and 128 full-set writes with the unchanged frozen M3 update rule. Each retrieval ranks eight values by cosine. Its permutation control preserves the key and candidate sets and has no fixed points. The positive control requires eight correct retrievals before any E3 optimization. This fixed acquisition fixture is not a claim about all update schedules.

## Observed result

The dataset audit passed. The fixed K/V/Q positive control was BLOCKED at 5/8 correct versus 0/8 for the permuted Cell. Fast Weight delta was 4.7794. Full inner-loss and control histories are in cpu-preflight.json. Export/import parity for this fixture was downstream of the failed gate and remains pending. The frozen E3 adapter and candidate objective are implemented for the next gate. The E3 fixed-episode optimization, held-out screening, T4 and C0–C4 remain unexecuted.

EPCLM-006 remains INCONCLUSIVE. Its evaluation conflated unseen pairs with unseen symbols, included random-answer loss without a Cell, and used full-vocabulary decoding from a randomly initialized E3. This experiment uses shared symbol sets, candidate ranking, separated K/V/Q input paths and no Cell dropout. E3 is explicitly a frozen initialization. Neither the EPCLM-006 artifacts nor the frozen M3 update rule were changed. Diagnose inner acquisition before spending resources on E3.

## Screening entry

`python screen007.py /tmp/epclm007-screen-new` runs the staged CPU gates,
then requires a real T4 and a no-update budget probe. The Notebook invokes
this entry under the session deadline. CPU gate failure writes a BLOCKED
decision and stops. Raw results and checkpoints remain in Kaggle Output.

The final data protocol uses 256 shared Key symbols and 256 shared Value
symbols, shuffled disjoint pair partitions, 4,096 training episodes and
512 validation episodes. Each episode provides eight distinct candidates
and four scored queries. C2 rolls the assignment with no fixed points.
C0 scores candidates uniformly. Split membership can constrain admissible
pairs, so the uniform reference is a scoring control and not a bound on
a predictor given the partition. Confirmation and independent Test require
a separate protocol.

If the data and fixed-episode gates pass, the screening runner uses
1,024 batch-four updates with a frozen, randomly initialized E3. A
held-out gate requires C1 accuracy of at least 50%, improvement in both
accuracy and NLL, nonconstant candidate predictions, controls at or below
17.5%, and positive paired interval lower bounds versus both controls.
Only after that gate does the runner save a validation Cell and verify its
candidate logits in a fresh process. C3 is the source-side canonical read.
There is no session-conditioned feature path or learned gate in this
retrieval-only adapter. C4 checks checkpoint parity for one validation
episode. It does not constitute multi-seed portability confirmation.

The Direct Gate diagnostic remains below its 8/8 criterion at 5/8. Correct candidate ranks are
`[5,1,5,1,1,6,1,1]`. The runner's downstream paths have not executed on
T4. Two CPU tests passed for target alignment, the no-fixed-point
permutation, sealed Test, candidate ranking and paired statistics.

### Gate P — Cross-session Cell portability

Run `python migration007.py <new-output-directory>` from this directory.
The CPU FP32 run uses the frozen Direct Gate fixture and 128 full-set replay updates
in Source. Source must reproduce 5/8 before export. The launcher waits for Source
to exit before launching Target. Target rebuilds the seeded static parameters and
uses only the portable Cell for dynamic state. The fixture uses no tokenizer,
E3 backbone, or external checkpoint. The existing v1 artifact format is reused.
Gate P compares complete readouts, scores, candidate rankings, margins and
persistent-state hashes. The Direct Gate still requires 8/8.
