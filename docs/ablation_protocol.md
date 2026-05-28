# Ablation Protocol

The current release includes evidence for the main component ablations:

- `SH-only`: Stage 2 optimizes SH coefficients while geometry is frozen.
- `App-only`: Stage 2 optimizes the appearance residual branch while geometry is frozen.
- `Full GeoFAR-SH`: Stage 2 optimizes SH coefficients and the residual branch jointly while geometry is frozen.

The consolidated summary is included in:

- `results/summary_ablation.csv`
- `results/main_table_ablation.tex`

## Reserved Future Ablations

The following ablations were not detected as complete release-ready result tables in the local scan and should be manually added if needed:

- mask-free residual
- gate-free residual
- single-residual branch
- latent dimension 4/8/16 comparison

These missing or uncertain items are also listed in `MISSING_FILES.md`.



