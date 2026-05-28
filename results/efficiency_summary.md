# Efficiency Summary

## Method Averages
- 3DGS-30k: render FPS 75.45, GPU memory 7125.1 MB, model size 580.07 MB
- 3DGS-40k-cont: render FPS 77.26, GPU memory 7132.0 MB, model size 580.98 MB
- SH-only-10k: render FPS 76.02, GPU memory 7127.8 MB, model size 580.98 MB
- App-only-10k: render FPS 75.49, GPU memory 7174.5 MB, model size 655.95 MB
- GeoFAR-SH: render FPS 74.45, GPU memory 7174.5 MB, model size 655.95 MB
- GeoFAR-SH torch-precompute: render FPS 39.35, GPU memory 7491.4 MB, model size 655.95 MB

## GeoFAR-SH Overheads
- Versus 3DGS-30k: FPS -1.00, memory +49.4 MB, model size +75.88 MB
- Versus SH-only-10k: additional appearance overhead is +74.97 MB in model size and +46.6 MB in memory.

## Optional GeoFAR-SH Torch-Precompute
- GeoFAR-SH torch-precompute: render FPS 39.35, GPU memory 7491.4 MB, model size 655.95 MB



