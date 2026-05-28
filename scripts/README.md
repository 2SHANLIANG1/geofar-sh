# Script Map

The scripts in this directory are copied or wrapped from the local research workspace and renamed for release readability.

| Release script | Original source path | Purpose |
|---|---|---|
| `run_fairness_controls.py` | `scripts/run_paper_fairness_controls.py` | Batch fairness-control variants for 13 scenes. |
| `evaluate_all.py` | `full_eval.py` | Offline rendering and metric computation entry point. |
| `aggregate_results.py` | `scripts/build_final_paper_tables.py` | Aggregate scene, dataset, and overall summaries. |
| `make_tables.py` | `scripts/paper_full/make_paper_tables.py` | Generate paper tables. |
| `make_figures.py` | `scripts/paper_full/make_paper_figures.py` | Generate paper figures. |
| `verify_geometry_frozen.py` | `scripts/verify_geometry_frozen_stage2.py` | Verify Stage 2 geometry freezing. |
| `measure_efficiency.py` | `scripts/collect_efficiency_evidence.py` | Collect FPS, memory, model size, and training-time evidence. |

Each copied script starts with a comment recording its original source path. Some command-line options may still reflect the internal development workflow; check `MISSING_FILES.md` before using the release package as a polished public API.



