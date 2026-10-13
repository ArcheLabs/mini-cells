# Research Experiments

Every experiment belongs to one project and lives at:

```text
research/projects/<project>/experiments/<experiment>/
```

That directory is the working unit for the study. It may contain a README,
protocol/configuration, implementation, notebook, reports, and experiment
artifacts. It need not contain empty placeholder subdirectories for assets it
does not use.

Cross-project policy and frozen validation records remain in
[`governance/`](./). Shared implementation utilities live under
[`shared/`](../../shared/); repository-level release and maintenance entrypoints
live under [`scripts/`](../../scripts/).

The registry is only a discovery index. The owning experiment's frozen protocol
and recorded source commit determine what was run and what can be concluded.
