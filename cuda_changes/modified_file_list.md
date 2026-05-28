# Modified File List

The following files were copied from the real local project into `cuda_changes/modified_files/` for inspection.

## Python-side renderer and model integration

- `gaussian_renderer/__init__.py`
  - Purpose: selects SH colour computation mode, applies appearance residual colour precomputation when requested, and passes enhanced colours to the rasterizer.
- `scene/appearance_residual.py`
  - Purpose: defines appearance residual configuration, residual head computation, fusion logic, and rasterizer input preparation.
- `scene/gaussian_model.py`
  - Purpose: stores Gaussian parameters, appearance latent/residual state, optimizer groups, checkpoint save/load behavior, and Stage 2 parameter control.
- `arguments/__init__.py`
  - Purpose: adds flags for `use_appearance_residual`, latent dimension, residual scale, compute mode, freezing controls, mask/gate options, and Stage 2 behavior.

## CUDA rasterizer interface

- `submodules/diff-gaussian-rasterization/ext.cpp`
  - Purpose: Python/C++ extension binding.
- `submodules/diff-gaussian-rasterization/rasterize_points.cu`
  - Purpose: rasterization entry points and tensor argument plumbing.
- `submodules/diff-gaussian-rasterization/rasterize_points.h`
  - Purpose: rasterization interface declarations.
- `submodules/diff-gaussian-rasterization/setup.py`
  - Purpose: extension build configuration.
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
  - Purpose: Python rasterizer wrapper.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
  - Purpose: forward CUDA rasterization path.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h`
  - Purpose: forward CUDA declarations.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
  - Purpose: backward CUDA gradient path.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h`
  - Purpose: backward CUDA declarations.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`
  - Purpose: host-side rasterizer implementation.
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
  - Purpose: host-side rasterizer declarations.

`geofar_sh_cuda.patch` was attempted from the local git metadata. In this workspace the submodule diff could not be reconstructed cleanly, so the copied modified files are the authoritative release evidence.



