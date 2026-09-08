# Contributing

Keep public API changes small and tested. New model backends require an
explicit support level, architecture and numerical compatibility tests, and
must fail closed on unknown layouts. Keep reusable implementation in Python
packages; research scripts should remain thin dispatch/validation entrypoints.

Preserve historical protocols, artifacts, source commits, and reserved formal
seeds. Do not run formal research seeds as part of ordinary tests or CI. Run
the focused HybridCLM tests and the existing regression suite before opening a
pull request.
