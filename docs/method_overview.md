# Method Overview

GeoFAR-SH is a post-convergence appearance refinement method for 3D Gaussian Splatting. The central goal is to improve colour and view-dependent appearance while preserving the already converged Gaussian geometry.

The method starts from a standard 3DGS model trained to 30k iterations. A second refinement stage resumes from this checkpoint and continues training to 40k iterations. During this stage, Gaussian positions, scales, rotations, and opacities are frozen. Densification and pruning are disabled so the Gaussian set and spatial support remain fixed.

Only appearance-related parameters are optimized in Stage 2. These include SH colour coefficients and a lightweight per-Gaussian appearance residual branch. The residual branch predicts a bounded correction to the colour derived from SH features. Optional mask and gate terms can modulate the correction.

At rendering time, GeoFAR-SH replaces only the per-Gaussian RGB before alpha compositing. Projection, visibility, depth sorting, tile traversal, opacity accumulation, and alpha compositing follow the original 3DGS pipeline.

This separation makes the method useful when the 3DGS geometry is sufficiently converged but the appearance remains imperfect. It is not intended to repair missing geometry, large density errors, or incorrect camera/data preparation.

## Main Components

- Stage 1: standard 3DGS training to 30k.
- Stage 2: appearance-only refinement from the same 30k checkpoint.
- SH-only variant: optimize SH coefficients only.
- App-only variant: optimize residual appearance branch only.
- GeoFAR-SH/Ours: jointly optimize SH coefficients and residual branch.

## Implementation Pointers

- `geofar_sh/src/scene/gaussian_model.py`
- `geofar_sh/src/scene/appearance_residual.py`
- `geofar_sh/src/gaussian_renderer/__init__.py`
- `geofar_sh/src/train.py`
- `configs/paper/`



