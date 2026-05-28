# Reproducibility Notes

This repository is organized to support reproducible research for:

**Geometry-Preserving Colour-Space Refinement for Appearance Enhancement in 3D Gaussian Splatting**

The release package provides the source code, protocols, summaries, and generation utilities needed to inspect and reproduce the reported GeoFAR-SH workflow.

## Included Materials

The repository contains:

- **core implementation** for the GeoFAR-SH appearance refinement method;
- **environment files** including `environment.yml` and `requirements.txt`;
- **training and evaluation scripts** for Stage 2 refinement, rendering, metrics, aggregation, and analysis;
- **fairness-control protocol** documentation and scripts for the main variants;
- **CUDA-fused rasterisation documentation** and modified rasterizer files under `cuda_changes/`;
- **aggregated results** for scene-level, dataset-level, overall, ablation, efficiency, and geometry-freezing analyses;
- **table and figure generation utilities** for paper-facing quantitative tables and qualitative figures.

## Materials Not Included

For storage constraints and third-party dataset licensing considerations, this repository does not include:

- public datasets;
- full trained checkpoints;
- complete per-view render folders;
- large TensorBoard logs.

These materials can be regenerated or obtained through the standard public data and training workflow described in the repository.

## Dataset Preparation

Users should download the public datasets from their official sources and place them according to the path layout described in `README.md`, for example:

```text
<DATA_ROOT>/
|-- mipnerf360/
|-- tandt/
`-- db/
```

The experiments use:

- Mip-NeRF360;
- Tanks&Temples;
- DeepBlending.

## Checkpoint Reproduction

Checkpoints can be reproduced by following the command templates in `reproduce_commands.md`.

The intended workflow is:

1. Train or prepare the standard 3DGS 30k checkpoint for each scene.
2. Resume from the 30k checkpoint for Stage 2 refinement.
3. Run the fairness-control variants:
   - `3DGS-30k`;
   - `3DGS-40k-cont`;
   - `SH-only-10k`;
   - `App-only-10k`;
   - `GeoFAR-SH / Ours`.
4. Run offline evaluation with the provided evaluation scripts.
5. Aggregate results and regenerate tables and figures.

## Reproduction Commands

Use `reproduce_commands.md` for Windows PowerShell and Linux bash templates covering:

- single-scene Stage 2 refinement;
- 13-scene fairness-control runs;
- offline evaluation;
- result aggregation;
- table generation;
- figure generation;
- geometry-freezing verification;
- efficiency measurement.

All command templates use placeholders such as `<PROJECT_ROOT>`, `<DATA_ROOT>`, `<OUTPUT_ROOT>`, `<SCENE>`, and `<CHECKPOINT_30K>` so they can be adapted to different local systems.

## CUDA Rasterisation Notes

CUDA-fused rasterisation documentation is provided under `cuda_changes/`.

The documentation explains how GeoFAR-SH replaces the per-Gaussian RGB before alpha compositing while keeping projection, visibility, depth ordering, opacity accumulation, and alpha compositing unchanged.

## Result Summaries

The `results/` directory contains compact aggregated summaries and LaTeX table artifacts. These files are intended for inspection, manuscript table generation, and consistency checks without requiring the large intermediate training artifacts.
