# Frozen before generator execution

A1 uses 44 Train and 11 Dev canonical facts. The explicit lists in config are selected by domain coverage, without model outputs. Each role covers zero operands, equal operands, sums below ten and sums at least ten. Eight explicit Train facts form Tiny Overfit. All operand orders and templates remain in the same fact role.

Copy uses disjoint numerical identities: four Dev numbers and the other fifteen Train numbers. This deliberately prevents exact Copy fact leakage. Its Train sampling weights follow A1 Train answer frequencies on eligible numbers (minimum one); the remaining answer-frequency and length mismatch must be reported, never tuned to quality. Copy is a numerical output control, not arithmetic generalization.

Two shared templates are trained. A third is held out for both tasks. All template/order variants are aggregated within fact, not independent observations. Decimal answers have no final period and end in EOS. The newline delimiter must tokenize as an exact prompt prefix; digit offsets must cover only the answer. Any violation stops preparation rather than masking prompt bytes as answers.

Tiny memorization threshold is 6/8 exact freely generated answers after exactly 200 updates. Pilot budget is fixed at 900 updates for each of nine jobs; it remains unreleased until separate review. A2/A3 remain prohibited.

Cell 1 means zero-based index 1 (the second Cell). The primary exploratory specialization contrast is the final-minus-anchor change in normalized A1-minus-Story replacement harm, F2 minus each of F0 and F1, using usage matching. Reference threshold 0.005 in at least two seeds with positive A1 replacement harm; all Cells, controls and times are reported. No inference of Connection value.
