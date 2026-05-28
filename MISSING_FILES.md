# Missing Files and Manual Checks

This file records items that were not safely inferred or not included in the cleaned GitHub release package.

## Must Confirm Before Public GitHub Upload

- **Author metadata:** `CITATION.cff` contains placeholder author and repository fields.
- **License choice:** `LICENSE` was copied from the local project if available, but the final public license should be confirmed by the authors.
- **Public repository URL:** no GitHub URL was available during packaging.
- **Exact tested environment:** `requirements.txt` and `environment.yml` contain dependency ranges and package names, not fully pinned tested versions.
- **CUDA/PyTorch build:** CUDA toolkit, driver, and PyTorch CUDA wheel versions must be adapted to the target machine.

## Reproducibility Files Not Included

- Public datasets are not included. Users must download Mip-NeRF360, Tanks&Temples, and DeepBlending separately.
- 30k checkpoints are not included.
- 40k refined checkpoints are not included.
- `.pth`, `.pt`, `.ckpt`, `.ply`, `.npy`, and `.npz` artifacts are excluded.
- Full render outputs and test-view image folders are excluded.
- TensorBoard event files and large training logs are excluded.

## CUDA Patch Status

- `cuda_changes/geofar_sh_cuda.patch` could not be reconstructed as a clean non-empty git patch from the local workspace because the rasterizer submodule git metadata was not available in a usable state.
- The release therefore includes copied real modified files under `cuda_changes/modified_files/`.
- Recommended manual action: regenerate a clean patch by comparing this project against the original 3DGS / `diff-gaussian-rasterization` baseline.

## Script API Checks

The release scripts were copied and renamed from local paper scripts. They contain source-path comments, but some command-line options may still reflect the local workflow.

Manual checks recommended:

- `scripts/run_fairness_controls.py`
- `scripts/evaluate_all.py`
- `scripts/aggregate_results.py`
- `scripts/make_tables.py`
- `scripts/make_figures.py`
- `scripts/verify_geometry_frozen.py`
- `scripts/measure_efficiency.py`

## Ablations Not Detected as Release-Ready Result Tables

- mask-free residual
- gate-free residual
- single-residual branch
- latent dimension 4/8/16 comparison

The current release includes evidence for SH-only, App-only, and full GeoFAR-SH ablations.

## Optional Additions

- A polished one-command installer for CUDA extensions.
- A small synthetic/demo scene that is legally redistributable.
- Pretrained tiny demo checkpoint, if size and license permit.
- Regenerated CUDA patch from a clean upstream baseline.
