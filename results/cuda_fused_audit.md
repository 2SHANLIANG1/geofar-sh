# CUDA-Fused Audit Report

Project root: `<PROJECT_ROOT>`  
Audit date: `2026-05-13`

## 1. Final Verdict

**Full CUDA-fused**

Reason: in the default fused path of the current project, all required appearance-residual components are implemented inside the CUDA rasterizer path, and both forward/backward CUDA code exist for:

- appearance latent
- diffuse residual head
- specular residual head
- specular mask
- global gate
- residual fusion
- enhanced color application on top of SH color

Important boundary condition:

- This verdict is true for the fused runtime path: `appearance_compute_mode -> fused`, `convert_SHs_python == False`, and `colors_precomp == None`.
- The same codebase also contains a fallback/non-fused path: if `pipe.convert_SHs_python == True` or `appearance_compute_mode == "torch_precompute"`, color is computed in Python before rasterization. That path is **not** CUDA-fused, but it is not the default path.

## 2. Evidence Table

| Question | Finding | Evidence file | Line numbers | Interpretation |
| --- | --- | --- | --- | --- |
| appearance latent creation | created as `nn.Parameter` in Python model state | `scene/gaussian_model.py` | `130-151`, `562-568` | Parameter storage is in Python, expected for Torch extension inputs |
| appearance latent read for fused render | packed into `appearance_inputs` then passed into rasterizer | `scene/appearance_residual.py`, `gaussian_renderer/__init__.py` | `601-676`, `77-98`, `102-125` | Runtime data enters CUDA rasterizer |
| diffuse residual head location | CUDA device function in rasterizer | `cuda_rasterizer/forward.cu` | `101-124`, `205-223` | Diffuse residual is computed inside CUDA |
| specular residual head location | CUDA device function in rasterizer | `cuda_rasterizer/forward.cu` | `87-99`, `214-223` | Specular residual is computed inside CUDA |
| specular mask location | CUDA sigmoid in rasterizer | `cuda_rasterizer/forward.cu` | `208-223` | Mask is computed inside CUDA |
| global gate location | CUDA sigmoid in rasterizer | `cuda_rasterizer/forward.cu` | `199-203` | Gate is computed inside CUDA |
| residual fusion formula | `(1-mask)*diff + mask*spec`, then `rgb_sh + lambda * gate * residual` in CUDA | `cuda_rasterizer/forward.cu` | `221-237` | Residual fusion and enhanced color are CUDA-side |
| enhanced color timing | computed inside rasterizer only when `colors_precomp == nullptr` | `cuda_rasterizer/forward.cu` | `443-487` | Enhanced color is integrated into SH color path in CUDA |
| SH color location in fused path | `computeColorFromSH` in CUDA | `cuda_rasterizer/forward.cu` | `18-71`, `447-448` | SH -> RGB is CUDA-side in default fused path |
| SH color location in Python fallback | Python `eval_sh(...)` if `convert_SHs_python` or `torch_precompute` | `gaussian_renderer/__init__.py` | `81-89` | Non-fused fallback exists |
| diffuse/spec/mask/gate backward | CUDA backward implementation exists | `cuda_rasterizer/backward.cu` | `85-382` | Full analytic backward exists in CUDA |
| appearance latent backward | `dL_dappearance_latent` accumulated in CUDA | `cuda_rasterizer/backward.cu` | `123-141`, `365-380`, `856-915` | Latent gradients are CUDA-side |
| head parameter backward | gradients for `app_w_*`, `app_b_*`, second-layer params accumulated in CUDA | `cuda_rasterizer/backward.cu` | `248-301`, `323-361`, `856-915` | Head gradients are CUDA-side |
| Python residual head in hot render path | not used in default render path; Python helper exists for regularization/fallback only | `scene/appearance_residual.py`, `gaussian_renderer/__init__.py` | `275-276`, `471-598`, `601-676`, `79-98` | Python helper exists, but fused render path uses CUDA |
| rasterizer binding supports appearance params | binding forwards all appearance tensors/flags into `_C.rasterize_gaussians` and backward | `diff_gaussian_rasterization/__init__.py` | `21-104`, `152-208`, `259-379`, `412-512` | Appearance path is truly integrated into rasterizer API |

## 3. Call Chain

Default training/render fused chain:

`train.py` -> `gaussian_renderer.render(...)` -> `GaussianRasterizer.forward(...)` -> `diff_gaussian_rasterization._RasterizeGaussians.forward(...)` -> `_C.rasterize_gaussians(...)` -> `RasterizeGaussiansCUDA(...)` -> `CudaRasterizer::Rasterizer::forward(...)` -> `FORWARD::preprocess(...)` -> `preprocessCUDA(...)` -> `computeColorFromSH(...)` -> `applyAppearanceResidual(...)` -> `FORWARD::render(...)`

Backward chain:

`loss.backward()` -> `diff_gaussian_rasterization._RasterizeGaussians.backward(...)` -> `_C.rasterize_gaussians_backward(...)` -> `RasterizeGaussiansBackwardCUDA(...)` -> `CudaRasterizer::Rasterizer::backward(...)` -> `BACKWARD::render(...)` -> `BACKWARD::preprocess(...)` -> `preprocessCUDA(...)` -> `computeAppearanceResidualBackward(...)`

Where appearance residual happens:

- Python packs appearance tensors in `build_appearance_inputs_for_rasterizer(...)`: `scene/appearance_residual.py:601-676`
- Python decides fused vs precompute mode in `gaussian_renderer.render(...)`: `gaussian_renderer/__init__.py:77-98`
- Actual per-Gaussian appearance residual computation happens inside CUDA:
  - SH color: `cuda_rasterizer/forward.cu:20-71`
  - residual/gate/mask/fusion: `cuda_rasterizer/forward.cu:126-237`
  - invocation point in preprocess kernel: `cuda_rasterizer/forward.cu:443-487`

## 4. Direct Answers To The Audit Questions

### appearance latent 鏄湪鍝噷鍒涘缓鐨勶紵

- Created in Python as trainable tensor state:
  - `scene/gaussian_model.py:130-151`
  - `scene/gaussian_model.py:562-568`

### appearance latent 鏄湪鍝噷琚鍙栫殑锛?
- Packed in Python:
  - `scene/appearance_residual.py:601-676`
- Passed through Torch binding:
  - `diff_gaussian_rasterization/__init__.py:438-511`
- Read in CUDA forward:
  - `cuda_rasterizer/forward.cu:165-171`
- Read in CUDA backward:
  - `cuda_rasterizer/backward.cu:149-159`

### diffuse residual head 鏄?Python torch.nn.Module 杩樻槸 CUDA kernel锛?
- **CUDA kernel in the hot render path**
- Evidence:
  - `cuda_rasterizer/forward.cu:101-124`
  - `cuda_rasterizer/forward.cu:214-223`
- Note:
  - Python helper exists for analysis/regularization fallback, but default render path uses CUDA.

### specular residual head 鏄?Python torch.nn.Module 杩樻槸 CUDA kernel锛?
- **CUDA kernel in the hot render path**
- Evidence:
  - `cuda_rasterizer/forward.cu:87-99`
  - `cuda_rasterizer/forward.cu:218-223`

### specular mask 鏄?Python 璁＄畻杩樻槸 CUDA 璁＄畻锛?
- **CUDA computation in the fused path**
- Evidence:
  - `cuda_rasterizer/forward.cu:208-223`

### global gate 鏄?Python 璁＄畻杩樻槸 CUDA 璁＄畻锛?
- **CUDA computation in the fused path**
- Evidence:
  - `cuda_rasterizer/forward.cu:199-203`

### enhanced color 鏄湪璋冪敤 GaussianRasterizer 涔嬪墠灏辩畻濂戒簡锛岃繕鏄湪 CUDA rasterizer 鍐呴儴绠楃殑锛?
- **Default fused path**: inside CUDA rasterizer
  - `cuda_rasterizer/forward.cu:447-485`
- **Fallback path**: can be precomputed in Python if `appearance_compute_mode == "torch_precompute"`
  - `gaussian_renderer/__init__.py:81-89`
  - `scene/appearance_residual.py:592-598`

### SH color 鏄湪 Python 绔畻锛岃繕鏄湪 CUDA computeColorFromSH 鍐呯畻锛?
- **Default path**: CUDA `computeColorFromSH`
  - `cuda_rasterizer/forward.cu:20-71`
  - `cuda_rasterizer/forward.cu:447-448`
- **Fallback path**: Python `eval_sh(...)`
  - `gaussian_renderer/__init__.py:83-87`

### 鏈€缁堜紶鍏?rasterizer 鐨?`colors_precomp / features / shs` 鍒嗗埆鏄粈涔堬紵

- Default fused path:
  - `colors_precomp = None`
  - `shs = pc.get_features` or split `dc=pc.get_features_dc`, `shs=pc.get_features_rest`
  - `appearance_inputs = build_appearance_inputs_for_rasterizer(...)`
  - Evidence: `gaussian_renderer/__init__.py:81-98`, `102-125`
- Python fallback path:
  - `colors_precomp = clamp(eval_sh(...) + 0.5)` or enhanced Python precompute color
  - `shs = None`
  - CUDA appearance branch is skipped because `colors_precomp != nullptr`
  - Evidence: `gaussian_renderer/__init__.py:81-89`, `cuda_rasterizer/forward.cu:443-487`

## 5. CUDA Kernel Evidence

### forward.cu

Found:

- `computeColorFromSH(...)`: `cuda_rasterizer/forward.cu:20-71`
- `applyAppearanceResidual(...)`: `cuda_rasterizer/forward.cu:126-237`
- gate sigmoid: `cuda_rasterizer/forward.cu:199-203`
- specular mask sigmoid: `cuda_rasterizer/forward.cu:208-211`
- diffuse residual branch: `cuda_rasterizer/forward.cu:214-223`
- specular residual branch: `cuda_rasterizer/forward.cu:218-223`
- residual fusion: `cuda_rasterizer/forward.cu:223`
- enhanced color writeback: `cuda_rasterizer/forward.cu:237`
- invocation inside preprocess kernel: `cuda_rasterizer/forward.cu:447-485`

Not found:

- No separate external Python-only head invocation inside CUDA path. The computation is directly implemented as device functions.

### backward.cu

Found:

- `computeAppearanceResidualBackward(...)`: `cuda_rasterizer/backward.cu:85-382`
- gate backward: `cuda_rasterizer/backward.cu:337-361`
- mask backward: `cuda_rasterizer/backward.cu:293-313`
- diffuse/spec backward: `cuda_rasterizer/backward.cu:214-291`
- latent gradient accumulation: `cuda_rasterizer/backward.cu:323-328`, `365-380`
- mean gradient from appearance path: `cuda_rasterizer/backward.cu:372-380`
- preprocess backward invocation: `cuda_rasterizer/backward.cu:856-915`

Not found:

- No missing backward for latent/head/mask/gate in the fused path.

### rasterizer_impl / ext / setup evidence

- Binding exports forward/backward:
  - `submodules/diff-gaussian-rasterization/ext.cpp:15-18`
- setup builds forward/backward CUDA:
  - `submodules/diff-gaussian-rasterization/setup.py:25-26`
- C++ rasterizer forward wires appearance params into `FORWARD::preprocess(...)`:
  - `cuda_rasterizer/rasterizer_impl.cu:198-334`
- C++ rasterizer backward wires appearance params and grads into `BACKWARD::preprocess(...)`:
  - `cuda_rasterizer/rasterizer_impl.cu:403-608`

## 6. Paper Wording Recommendation

Because the default implementation path in the current repository is genuinely fused into the CUDA rasterizer color path and has CUDA backward support, the paper **can keep** the 鈥淐UDA-fused鈥?claim, with two caveats:

- The paper should explicitly state that this refers to the default fused runtime path, not the optional Python fallback path.
- The paper should disclose that `convert_SHs_python=True` or `appearance_compute_mode="torch_precompute"` switches to a non-fused fallback.

Recommended wording:

- Title / contribution / abstract:
  - 鈥渁 CUDA-fused appearance residual directly integrated into the per-Gaussian SH color path of the rasterizer鈥?- Method clarification sentence:
  - 鈥淚n the default fused path, SH color evaluation, appearance latent decoding, diffuse/specular residual heads, specular mask, global gate, residual fusion, and their backward gradients are implemented inside the CUDA rasterizer.鈥?- Reproducibility caveat:
  - 鈥淭he repository also retains a Python precompute fallback (`torch_precompute` / `convert_SHs_python`) for debugging and compatibility; this fallback is not the path used by the main method claim.鈥?
## 7. Required Code Evidence Appendix

### Key grep / search results

```powershell
rg -n -S "appearance|appearance_latent|use_appearance|residual|diffuse|specular|mask|gate|global_gate|residual_scale|enhanced_color|sh_refine|features_dc|features_rest|rasterizer|GaussianRasterizer|diff_gaussian_rasterization|computeColorFromSH|preprocessCUDA|renderCUDA|backward" <PROJECT_ROOT>
```

Key hits used in this audit:

- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `scene/appearance_residual.py`
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
- `submodules/diff-gaussian-rasterization/rasterize_points.cu`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`

### Key file fragments

#### Runtime mode selection

- `scene/appearance_residual.py:121-127`
  - `appearance_compute_mode` normalizes `"auto"` to `"fused"`
- `arguments/__init__.py:68-69`
  - `convert_SHs_python = False`
  - `compute_cov3D_python = False`

#### Python side only packs inputs for fused render

- `gaussian_renderer/__init__.py:77-98`
  - decides between Python precompute and CUDA fused path
- `scene/appearance_residual.py:601-676`
  - packages latent, head weights, mask/gate parameters, flags

#### CUDA fused forward

- `cuda_rasterizer/forward.cu:126-237`
  - full appearance residual forward
- `cuda_rasterizer/forward.cu:443-487`
  - directly inserted after `computeColorFromSH(...)`

#### CUDA fused backward

- `cuda_rasterizer/backward.cu:85-382`
  - full appearance residual backward
- `cuda_rasterizer/backward.cu:856-915`
  - invoked from preprocess backward kernel

### Audit Notes

- The Python function `compute_appearance_regularizers(...)` recomputes appearance branch outputs in Python for logging/regularization: `scene/appearance_residual.py:679-810`. This does **not** invalidate the CUDA-fused render claim, because the render hot path and its primary optimization gradients still run through the CUDA implementation.
- The Python fallback path is real and should be documented, but it is not the default path selected by the current repository configuration.



