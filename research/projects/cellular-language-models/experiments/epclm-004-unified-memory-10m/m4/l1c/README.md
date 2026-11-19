# EPCLM-005 H4 L1-C

L1-C keeps the frozen E5 dimensions, the M3 learning rule and the M2
transactional trainer. The contextual model explicitly splits each training
batch at 272 + 240 tokens. The trainer receives aligned logits with Session A
excluded from loss. A writes SWA-contextual representations to Titans. B starts
new SWA and Mamba state and reads the carried Cell with meta-gradients enabled.
B performs zero Cell updates.

The [Kaggle notebook](notebook.ipynb) runs CPU and T4 preflight, then a
batch-four timing/VRAM probe without optimizer updates. Select Tesla T4,
enable Internet and the existing GITHUB_TOKEN, and Save & Run All. Keep the
complete Output for an exact-source resume or publication retry. The run
stops on a failed gate or budget violation. Actual screening is limited to
32 updates and 65,536 tokens per arm, with FP32 and TF32 disabled.

Local engineering verification in the existing PyTorch environment:

```sh
python checks_l1c.py /tmp/h4-l1c-preflight-new
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_l1c.py -q
```

The fresh-process check saves A's Cell and B's pre-query Mamba state
independently. It reconstructs B-local SWA from the B prefix and verifies the
continuation against the original logits. No Session A SWA or Mamba state
enters Session B. Each episode has four queries from A and four from B.
The scorer produces source-specific and joint EM, NLL, top-k, entropy, target
ranks, Token 432 diagnostics, query-specific gates, and F11 branch ablations.
The fixed Validation sample is profiled before training, and final
diagnostics use all three held-out banks. Intermediate eight-update
checkpoints follow the existing trainer policy without extra profiling passes.
