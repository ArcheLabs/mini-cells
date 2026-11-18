# Frozen associative recall diagnostic

Run the Python contents of kaggle_diagnostic.py in the existing Kaggle notebook with a T4, Internet and GITHUB_TOKEN. Integrated smoke runs one episode before the paired diagnostic. Results publish to GitHub using the P1.2 artifact workflow.

The same 16 deterministic episodes (four random code–value bindings each) are replayed for seeds 42–44 at checkpoints 0 and 200. Donors cycle the four values while retaining keys and candidate tokens. Each cell commits one 512-token write chunk; queries reuse positions and are read-only. The direct probe compares the observed read with the correct write's contextual V vectors. Language scoring reports candidate accuracy and full-vocabulary target log probability. Interpret candidate accuracy alongside the swapped binding and empty controls; the latter uses native attention with no history, not transcript replay. No outer updates or new layers.
