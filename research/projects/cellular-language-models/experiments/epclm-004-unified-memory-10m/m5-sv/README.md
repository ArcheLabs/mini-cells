# EPCLM-004 M5-SV

Scientific validation of the frozen M5 mechanism; no new memory architecture and no M6.

Read [SPEC.md](SPEC.md), [baseline audit](audit/m5-baseline-audit.md), [REPORT.md](REPORT.md) and [implementation decisions](implementation_deviations.md). All new repository prose is English; historical documents are preserved.

The single [Notebook](notebook.ipynb) targets the existing Kaggle Notebook with Tesla T4, Internet and existing secrets. It proceeds automatically through protocol checks, frozen data, development diagnostics, bounded Pilot, immutable freeze, three paired seeds and independent Test. Each session is limited to3 hours. Attach the prior saved Output for subsequent sessions; exact source/config identity is checked automatically. Completed updates and evaluation Episodes are reused.

A failed learnability Pilot is INCONCLUSIVE and stops further training. A timeout leaves PARTIAL work and publishes it. All required Test cases must finish before a scientific verdict. A completed engineering stage cannot set H5=SUPPORTED.
