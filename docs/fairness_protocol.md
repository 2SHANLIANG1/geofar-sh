# Fairness Protocol

The paper protocol is designed to isolate post-convergence appearance refinement from geometry changes.

## Shared Conditions

- All Stage 2 variants start from the same 30k 3DGS checkpoint for a given scene.
- All variants use the same train/test split.
- All variants use the same test views.
- Offline evaluation uses the same rendering and metric scripts.
- Metrics include PSNR, SSIM, and LPIPS.
- No test-time optimization is used.

## Variants

- `3DGS-30k`: original 3DGS checkpoint at 30k iterations.
- `3DGS-40k-cont`: continuation from 30k to 40k with original 3DGS parameter updates.
- `SH-only-10k`: Stage 2 refinement with geometry frozen and only SH colour coefficients optimized.
- `App-only-10k`: Stage 2 refinement with geometry frozen and only the appearance residual branch optimized.
- `GeoFAR-SH / Ours`: Stage 2 refinement with geometry frozen and both SH coefficients and the residual branch optimized.

## Frozen Stage 2 Rules

- Gaussian positions are frozen.
- Gaussian scales are frozen.
- Gaussian rotations are frozen.
- Gaussian opacities are frozen.
- Densification is disabled.
- Pruning is disabled.

## Evidence in This Release

- `results/summary_scene_level.csv`
- `results/summary_overall_average.csv`
- `results/summary_dataset_average.csv`
- `results/geometry_frozen_summary.csv`
- `results/geometry_frozen_per_scene.csv`



