# Results Included in This Release

This directory contains small, manuscript-relevant result summaries copied from the local `output/` tree. Large checkpoints, point clouds, raw rendered views, TensorBoard event files, and full training logs are excluded.

## Selected Sources

- `summary_overall_average.csv`, `summary_dataset_average.csv`, `summary_scene_level.csv`, `summary_ablation.csv`, `summary_efficiency.csv`
  - source: `output/paper_final_tables/`
  - reason: final consolidated paper tables for the 13 valid scenes and fairness-control comparison.
- `main_table_*.tex`, `appendix_per_scene_table.tex`
  - source: `output/paper_final_tables/`
  - reason: LaTeX-ready manuscript table artifacts.
- `efficiency_summary.csv`, `efficiency_per_scene.csv`, `efficiency_summary.md`
  - source: `output/paper_efficiency_evidence/`
  - reason: FPS, memory, model size, and Stage-2 timing evidence.
- `RUNNING_STATUS.md`, `failed_jobs.md`, `fairness_analysis.md`, `win_rate_summary.csv`
  - source: `output/paper_fairness_controls/`
  - reason: fairness-control status and auxiliary analysis.
- `geometry_frozen_summary.csv`, `geometry_frozen_per_scene.csv`, `geometry_frozen_summary.md`
  - source: `output/paper_geometry_frozen_verification/`
  - reason: verification evidence for frozen geometry in Stage 2.
- `cuda_fused_audit.md`
  - source: `output/cuda_fused_audit/`
  - reason: implementation reproducibility and CUDA-fused audit notes.

## Exclusions

Per-scene raw `results.json`, `metrics.json`, `time_memory.json`, and complete render folders were not copied to keep the release compact and free of generated heavy outputs. Re-run the scripts to regenerate them locally.



