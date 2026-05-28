# Numeric Consistency Report

## Data Sources

- `<PROJECT_ROOT>\output\paper_fairness_controls`
- `<PROJECT_ROOT>\output\paper_main_b_multiscene`
- `<PROJECT_ROOT>\output\paper_ablation_evidence`
- `<PROJECT_ROOT>\output\garden_stage2_ablation`

Selected final source root: `<PROJECT_ROOT>\output\paper_fairness_controls`

## Completeness

- Scene-method rows found: `65` / `65`
- 65 scene-variant pairs complete: `yes`

## Overall Averages

- `3DGS-30k`: PSNR `27.2860`, SSIM `0.8333`, LPIPS `0.2158`
- `3DGS-40k-cont`: PSNR `27.4009`, SSIM `0.8340`, LPIPS `0.2135`
- `SH-only-10k`: PSNR `27.5014`, SSIM `0.8351`, LPIPS `0.2136`
- `App-only-10k`: PSNR `27.4756`, SSIM `0.8351`, LPIPS `0.2138`
- `GeoFAR-SH`: PSNR `27.5023`, SSIM `0.8351`, LPIPS `0.2133`

## GeoFAR-SH Deltas

- vs `3DGS-30k`: Delta PSNR `0.2164`, Delta SSIM `0.0018`, Delta LPIPS `-0.0025`
- vs `3DGS-40k-cont`: Delta PSNR `0.1014`, Delta SSIM `0.0011`, Delta LPIPS `-0.0002`
- vs `SH-only-10k`: Delta PSNR `0.0009`, Delta SSIM `0.0000`, Delta LPIPS `-0.0003`
- vs `App-only-10k`: Delta PSNR `0.0268`, Delta SSIM `0.0000`, Delta LPIPS `-0.0005`

## Appendix Consistency

- Appendix Average matches summary_overall_average.csv: `yes`

## Paper TeX Consistency

- Main TeX source not found in the workspace. Abstract / Conclusion / Table 2 / Table 3 / Table 5 / Table 7 could not be auto-synced.



