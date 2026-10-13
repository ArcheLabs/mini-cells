# Reproducible routing experiments

Consult `shared/python/experiments/registry.yaml` for status and experiment directories for
entry notebooks/configuration/protocols. `003-hard-cell-transition` is registered
and unrun. 002 has only development seed 42; its full matrix remains incomplete.
Historical source/config/artifact paths remain intact for replay and citations.
Shared new execution logic is in `src/minicells`; adapters reuse the frozen
001/002 components rather than redefining attention, tokenization or auth.

## Workflow

1. Run **Kaggle Auth Test** manually when checking API credentials. It lists
   kernels without launching a notebook or reserving GPU time. The only Actions
   Kaggle credential is `KAGGLE_API_TOKEN`; no username/key/JSON credential is needed.
2. Run **Experiment Notebook Preparation** for an exact repository commit and
   one registered development seed. It produces the canonical notebook, an
   exported script, metadata and hashes as an Actions artifact. Preparation is
   the default and consumes no Kaggle GPU quota.
3. Use the existing `bingjiangli/minicells` editor with the existing secret labels
   HF_TOKEN/GITHUB_TOKEN enabled. Replace its code with the prepared script and
   run there. The optional submit checkbox pushes to that same kernel and sets
   server timeout to one hour; it does not create another notebook.
4. Inspect the generated experiment_report.md and immutable GitHub results branch.
   Incomplete runs, missing measurements and failed publication remain visible.
   Download checkpoints from Kaggle outputs; only hashes go to GitHub.

Kaggle API authentication and notebook runtime secret authorization are distinct.
The public push schema offers no secret-label attachment argument. Accordingly,
the pipeline does not invent secret metadata or embed tokens in code. See the
[upstream Kaggle discussion](https://github.com/Kaggle/kaggle-cli/issues/582).
An API submission may still fail to read secrets; use the already-authorized
original editor. Repeated GPU smoke jobs are not part of this workflow.

GitHub publication reuses `research/projects/continual-learning/context/04-continual-learning/scripts/research/publish_core_validation_007.py` and
`publish_experiment_results.py`. A PAT requires ArcheLabs repository access and
**Contents read/write**, plus Metadata read. It needs no Actions, Workflows or
Pull requests permissions to push these result files. Organization approval,
membership and token policy still determine whether this repository is writable.

## Discipline

- Freeze config/protocol before observation; hash source, tokenizer, data and checkpoints.
- Use identical seeded windows/token budget for every arm, not equal wall time.
- Log training statistics every step and validation separately; compact console output.
- Count observed execution, document coverage; never rename a nominal FLOP estimate “actual”.
- Keep STOP and budget-forced termination distinct; report measured entropy and path lengths.
- Save partial evidence; never turn timeouts or missing arms into passing decisions.
- Previously observed development seeds are not untouched confirmation seeds.
- Respect the 55-minute application cap, dual-T4 registered schedule and original notebook identity.
- Do not merge, launch paid/GPU work or interpret the registry's unrun entries as results automatically.

Generated output/checkpoint directories are placeholders in Git; genuine evidence
is stored under `artifacts/experiments`, never fabricated in templates. Derived
plots are secondary to frozen protocol, decision JSON and raw per-step evidence.
