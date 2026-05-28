# Geometry-Preserving Colour-Space Refinement for Appearance Enhancement in 3D Gaussian Splatting

This repository contains the release package for the paper:

**Geometry-Preserving Colour-Space Refinement for Appearance Enhancement in 3D Gaussian Splatting**

GeoFAR-SH is a post-convergence 3D Gaussian Splatting (3DGS) appearance refinement framework. It is designed to improve rendered appearance while keeping the converged 3D Gaussian geometry fixed.

## Method Summary

The pipeline follows a two-stage protocol:

1. **Stage 1:** train the original 3DGS model to 30k iterations.
2. **Stage 2:** resume from the 30k checkpoint and continue to 40k iterations.
3. During Stage 2, freeze Gaussian positions, scales, rotations, and opacities.
4. Disable densification and pruning during Stage 2.
5. Optimize only SH colour coefficients and a lightweight per-Gaussian appearance residual branch.
6. Replace only the per-Gaussian RGB before alpha compositing with the enhanced colour.
7. Keep projection, visibility determination, depth ordering, opacity accumulation, and alpha compositing unchanged.

This makes the method a geometry-preserving colour-space refinement strategy rather than a geometry correction method.

## Main Variants

The main paper experiments compare:

- `3DGS-30k`
- `3DGS-40k-cont`
- `SH-only-10k`
- `App-only-10k`
- `GeoFAR-SH / Ours`

## Datasets

The experiments use public datasets:

- Mip-NeRF360
- Tanks&Temples
- DeepBlending

Datasets are **not** included in this repository. Place them in a local data root such as:

```text
DATA_ROOT/
|-- mipnerf360/
|-- tandt/
`-- db/
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate geofar_sh
pip install -r requirements.txt
```

The provided environment file reflects the tested local setup where available. Some CUDA/PyTorch versions may need to be adapted to the user's GPU driver and CUDA toolkit.

The CUDA rasterizer modifications are documented in `cuda_changes/`. If you build from a clean 3DGS checkout, inspect `cuda_changes/geofar_sh_cuda.patch` and `cuda_changes/modified_files/`.

## Repository Layout

```text
github_release_geofar_sh/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- environment.yml
|-- .gitignore
|-- SECURITY_CHECK_REPORT.md
|-- REPRODUCIBILITY_NOTES.md
|-- reproduce_commands.md
|-- configs/
|-- scripts/
|-- geofar_sh/
|-- cuda_changes/
|-- docs/
|-- results/
`-- figures/
```

## Reproduction Overview

Command templates are provided in `reproduce_commands.md` for both Windows PowerShell and Linux bash.

The intended workflow is:

1. Prepare 3DGS 30k checkpoints for each scene.
2. Run single-scene Stage 2 refinement for debugging.
3. Run the 13-scene fairness-control batch.
4. Render and evaluate all variants offline.
5. Aggregate scene-level, dataset-level, and overall results.
6. Generate paper tables.
7. Generate qualitative comparison, zoom-in, and error-map figures.
8. Verify that geometry is frozen during Stage 2.
9. Measure FPS, GPU memory, model size, and Stage-2 training time.

## Code Availability Note

This release includes the core implementation, environment files, training and evaluation scripts, fairness-control protocol, CUDA-fused rasterisation documentation, aggregated results, and table and figure generation utilities needed to understand and reproduce the reported protocol.

Complete public datasets, full trained checkpoints, complete per-view render folders, and large TensorBoard logs are not included because of storage constraints and third-party dataset licensing considerations.

## Important Caveats

- This is a cleaned release package copied from a local research workspace.
- Local absolute paths were replaced with portable placeholders such as `PROJECT_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT`.
- See `REPRODUCIBILITY_NOTES.md` for details on included materials, excluded large assets, and reproduction instructions. See `SECURITY_CHECK_REPORT.md` for the release security scan summary.



