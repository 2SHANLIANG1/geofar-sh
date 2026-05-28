# Figure Captions

**Figure 1.** Overall framework of GeoFAR-SH. A standard 3DGS model is first reconstructed from input views and used as a 30k-iteration checkpoint. During Stage 2, Gaussian positions, scales, rotations, and opacities are frozen, densification and pruning are disabled, and only SH colors and appearance residual parameters are refined. The enhanced color is injected into the rasterizer color path, while projection, visibility determination, and alpha compositing remain unchanged.

**Figure 2.** Residual module of GeoFAR-SH. The module takes the per-Gaussian appearance latent, viewing direction, and view distance as input, and predicts diffuse residual, specular residual, specular mask, and global gate through lightweight heads. The mask-gated fusion produces a controlled residual correction that is added to the base SH color.

**Figure 3.** Qualitative comparison on representative scenes. GeoFAR-SH improves local color consistency and detail preservation compared with the original 3DGS baseline and the 40k continuation baseline, while preserving the global scene structure.

**Figure 4.** Local zoom-in comparison. GeoFAR-SH mainly improves local complex appearance regions, including reflections, highlights, fine textures, and color-biased areas. The correction is concentrated in local residual appearance errors rather than large-scale geometric structures.

**Figure 5.** Error map comparison. The error map is computed as the average absolute RGB error between the rendered image and the ground-truth image. GeoFAR-SH reduces local errors in complex appearance regions while preserving the geometry-frozen reconstruction structure.

**Figure 6.** Fairness-control attribution analysis. The original 3DGS continuation baseline shows that additional training alone improves the 30k baseline, while the comparison with SH-only refinement indicates that geometry-frozen SH refinement explains most of the PSNR gain. GeoFAR-SH achieves the best overall average result with slightly improved perceptual quality.



