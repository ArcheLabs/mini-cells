# EPCLM-005 M4-H4 Fast Screening

L1 uses the frozen M4 E5 shell and M2 transaction loop. F00/F10/F01/F11 keep SWA active while switching Mamba-1 and Titans independently. The disabled modules are resident and frozen. Every arm starts from the same untrained weights.

The Kaggle entry is [notebook.ipynb](notebook.ipynb). Select Tesla T4, enable Internet and the existing GITHUB_TOKEN secret, then Save & Run All. The notebook checks CPU and T4 fixture behavior, measures the actual L1 workload, and runs the four arms only if the forecast fits. Checkpoints are retained in Output. Attach complete saved Output to resume or retry publication. The execution and notebook budgets are 9,600 and 10,500 seconds respectively.

Local checks in the existing PyTorch environment:

```sh
python screening.py --output /tmp/h4-audit-new
python preflight.py /tmp/h4-cpu-new
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_screening.py -q
```

Session A ends before the first QUERY and pads with HOLE to a complete chunk. Session B begins with empty SWA KV and carries only the selected Mamba tensors and Titans Cell. Its suffix contains query markers, Keys, answer markers and HOLE. Query reads use read-only Titans. F11 evaluates all four persistence conditions. A new interpreter checks one full state save/restore per arm and all four F11 conditions against the trained static revision.

The official L1 budget is 128 training episodes of 512 tokens, 32 batch-four updates, 65,536 inputs and 1,024 query targets per arm. Validation has 32 episodes. Three distinct evaluation banks have 64 episodes each. Train and held-out pairs use disjoint generator partitions. M4's five local layers have a 156-token composed receptive field, excluding the current task's distant facts.

If L1 triggers the common floor/ceiling rule, the run records the trigger. Level 2 and Level 3 require a frozen accounting decision because 32 batch-four updates at longer contexts and the fixed 65,536-input budget cannot both be preserved. The launcher implements the L1 schedule. Evaluation records, conditions, interval estimates and gate weights are returned to a main-derived result branch after execution. Scientific review considers the paired quality threshold, gate behavior and resource cost.
