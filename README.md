# Geometry-Preserving Colour-Space Refinement for Appearance Enhancement in 3D Gaussian Splatting

This repository contains the release package for the paper:

**Geometry-Preserving Colour-Space Refinement for Appearance Enhancement in 3D Gaussian Splatting**

GeoFAR-SH is a post-convergence 3D Gaussian Splatting appearance refinement framework.
It improves rendered appearance while keeping the converged 3D Gaussian geometry fixed.

## Method Summary

The pipeline follows a two-stage protocol:

1. Stage 1 trains the standard 3DGS model to 30k iterations.
2. Stage 2 resumes from the 30k checkpoint and continues to 40k iterations.
3. During Stage 2, Gaussian positions, scales, rotations, and opacities are frozen.
4. Densification and pruning are disabled during Stage 2.
5. Stage 2 optimizes only SH colour coefficients and a lightweight per-Gaussian appearance residual branch.
6. The enhanced colour replaces only the per-Gaussian RGB before alpha compositing.
7. Projection, visibility determination, depth ordering, opacity accumulation, and alpha compositing remain unchanged.

This design makes GeoFAR-SH a geometry-preserving colour-space refinement strategy rather than a geometry correction method.

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

Datasets are not included in this repository.
Download them from their official sources and place them in a local data root such as:

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

The provided environment file reflects the tested local setup where available.
Some CUDA and PyTorch versions may need to be adapted to the user's GPU driver and CUDA toolkit.

The CUDA rasterizer modifications are documented in `cuda_changes/`.
If you build from a clean 3DGS checkout, inspect `cuda_changes/geofar_sh_cuda.patch` and `cuda_changes/modified_files/`.

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

## Repository Contents

This release includes:

- core implementation of GeoFAR-SH;
- environment files for Python and package setup;
- training and evaluation scripts;
- fairness-control protocol documentation and scripts;
- CUDA-fused rasterisation documentation;
- aggregated experimental results;
- table and figure generation utilities.

## Reproduction Overview

Command templates are provided in `reproduce_commands.md` for both Windows PowerShell and Linux bash.

The intended workflow is:

1. Prepare or train the 3DGS 30k checkpoint for each scene.
2. Run single-scene Stage 2 refinement for debugging.
3. Run the 13-scene fairness-control batch.
4. Render and evaluate all variants offline.
5. Aggregate scene-level, dataset-level, and overall results.
6. Generate paper tables.
7. Generate qualitative comparison, zoom-in, and error-map figures.
8. Verify that geometry is frozen during Stage 2.
9. Measure FPS, GPU memory, model size, and Stage-2 training time.

## Code Availability Note

This release contains the core method code, configuration files, reproduction scripts, result summaries, and figure/table generation tools needed to understand and reproduce the reported protocol.

Complete public datasets, full trained checkpoints, complete per-view render folders, and large TensorBoard logs are not included because of storage constraints and third-party dataset licensing considerations.

## Citation

If you use this repository or protocol, please cite the accompanying paper.
The citation metadata is provided in `CITATION.cff`.

## Important Caveats

- This is a cleaned release package copied from a local research workspace.
- Local absolute paths were replaced with portable placeholders such as `PROJECT_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT`.
- See `REPRODUCIBILITY_NOTES.md` for details on included materials, excluded large assets, and reproduction instructions.
- See `SECURITY_CHECK_REPORT.md` for the release security scan summary.
