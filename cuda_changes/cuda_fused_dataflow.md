# CUDA-Fused Data Flow

## Original 3DGS Colour Path

In standard 3DGS, per-Gaussian colour is obtained from spherical harmonics (SH) either on the Python/Torch side or in the rasterizer path, depending on the `convert_SHs_python` and rasterizer settings. The resulting RGB is then rasterized with the Gaussian's projected footprint, opacity, depth ordering, and alpha compositing.

## GeoFAR-SH Appearance Path

GeoFAR-SH keeps the geometric rasterization path unchanged and modifies only the colour value used before alpha compositing:

1. Read SH colour features from the Gaussian model.
2. Optionally evaluate SH colour in Python/Torch when `convert_SHs_python` or `appearance_compute_mode=torch_precompute` is selected.
3. Build appearance inputs using per-Gaussian appearance latent features and view-dependent signals.
4. Compute lightweight appearance residual terms, including diffuse/specular residual components where enabled.
5. Apply mask and gate terms where enabled.
6. Fuse the residual with the SH colour to produce an enhanced per-Gaussian RGB.
7. Pass the enhanced RGB as the colour used by the rasterizer before alpha compositing.

## What Changes

- Additional appearance latent parameters are introduced for Stage 2.
- Residual colour heads predict bounded colour corrections.
- Optional mask and gate terms modulate residual application.
- Gradients propagate to SH coefficients and appearance parameters during Stage 2.

## What Stays Fixed

The following 3DGS operations remain unchanged:

- projection
- visibility
- depth ordering
- tile traversal
- Gaussian positions
- Gaussian scales
- Gaussian rotations
- Gaussian opacities
- alpha compositing
- opacity accumulation

## Backward Pass

The backward path must propagate gradients from the rendered image loss to:

- SH colour coefficients
- appearance latent vectors
- residual head parameters
- mask/gate parameters when enabled

It should not update frozen geometry parameters during Stage 2. Geometry-freezing verification is documented in `results/geometry_frozen_summary.md`.



