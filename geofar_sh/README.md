# GeoFAR-SH Core Code Map

This directory contains the copied method implementation from the local project.

## Source Mapping

- `train.py` -> `geofar_sh/src/train.py`
- `render.py` -> `geofar_sh/src/render.py`
- `metrics.py` -> `geofar_sh/src/metrics.py`
- `full_eval.py` -> `geofar_sh/src/full_eval.py`
- `arguments/` -> `geofar_sh/src/arguments/`
- `scene/` -> `geofar_sh/src/scene/`
- `gaussian_renderer/` -> `geofar_sh/src/gaussian_renderer/`
- `utils/` -> `geofar_sh/src/utils/`

## Relevant Implementation Areas

- Gaussian model and appearance residual state:
  - `geofar_sh/src/scene/gaussian_model.py`
  - `geofar_sh/src/scene/appearance_residual.py`
- Training loop and Stage 2 activation:
  - `geofar_sh/src/train.py`
- Rendering and colour replacement before rasterization:
  - `geofar_sh/src/gaussian_renderer/__init__.py`
- Arguments/configuration:
  - `geofar_sh/src/arguments/__init__.py`

The code is copied rather than moved. The original local project remains unchanged.



