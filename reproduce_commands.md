# Reproduction Command Templates

All commands use placeholders. Replace them locally before running:

- `<PROJECT_ROOT>`: path to this repository
- `<DATA_ROOT>`: path to datasets
- `<OUTPUT_ROOT>`: path for generated outputs
- `<SCENE>`: scene name
- `<CHECKPOINT_30K>`: path to the 30k 3DGS checkpoint

## Windows PowerShell

### 1. Single-scene Stage 2 refinement

```powershell
cd <PROJECT_ROOT>
conda activate geofar_sh
python geofar_sh/src/train.py -s <DATA_ROOT>/<SCENE> -m <OUTPUT_ROOT>/<SCENE>/geofar_sh_ours --start_checkpoint <CHECKPOINT_30K> --iterations 40000 --use_appearance_residual --appearance_residual_enable_step 30001 --appearance_latent_dim 8 --appearance_compute_mode fused --disable_stage2_densification
```

### 2. 13-scene fairness-control batch

```powershell
cd <PROJECT_ROOT>
python scripts/run_fairness_controls.py --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> --variants 3dgs_30k_baseline,3dgs_40k_cont,sh_only_10k,app_only_10k,geofar_sh_ours
```

### 3. Evaluation

```powershell
python scripts/evaluate_all.py --output-root <OUTPUT_ROOT> --data-root <DATA_ROOT>
```

### 4. Aggregate results

```powershell
python scripts/aggregate_results.py --input-root <OUTPUT_ROOT> --output-dir results
```

### 5. Make tables

```powershell
python scripts/make_tables.py --results-dir results --output-dir results
```

### 6. Make figures

```powershell
python scripts/make_figures.py --results-dir results --output-dir figures --render-root <OUTPUT_ROOT>
```

### 7. Verify geometry frozen

```powershell
python scripts/verify_geometry_frozen.py --output-root <OUTPUT_ROOT> --checkpoint-30k <CHECKPOINT_30K> --results-dir results
```

### 8. Measure efficiency

```powershell
python scripts/measure_efficiency.py --output-root <OUTPUT_ROOT> --data-root <DATA_ROOT> --results-dir results
```

## Linux Bash

### 1. Single-scene Stage 2 refinement

```bash
cd <PROJECT_ROOT>
conda activate geofar_sh
python geofar_sh/src/train.py -s <DATA_ROOT>/<SCENE> -m <OUTPUT_ROOT>/<SCENE>/geofar_sh_ours --start_checkpoint <CHECKPOINT_30K> --iterations 40000 --use_appearance_residual --appearance_residual_enable_step 30001 --appearance_latent_dim 8 --appearance_compute_mode fused --disable_stage2_densification
```

### 2. 13-scene fairness-control batch

```bash
cd <PROJECT_ROOT>
python scripts/run_fairness_controls.py --data-root <DATA_ROOT> --output-root <OUTPUT_ROOT> --variants 3dgs_30k_baseline,3dgs_40k_cont,sh_only_10k,app_only_10k,geofar_sh_ours
```

### 3. Evaluation

```bash
python scripts/evaluate_all.py --output-root <OUTPUT_ROOT> --data-root <DATA_ROOT>
```

### 4. Aggregate results

```bash
python scripts/aggregate_results.py --input-root <OUTPUT_ROOT> --output-dir results
```

### 5. Make tables

```bash
python scripts/make_tables.py --results-dir results --output-dir results
```

### 6. Make figures

```bash
python scripts/make_figures.py --results-dir results --output-dir figures --render-root <OUTPUT_ROOT>
```

### 7. Verify geometry frozen

```bash
python scripts/verify_geometry_frozen.py --output-root <OUTPUT_ROOT> --checkpoint-30k <CHECKPOINT_30K> --results-dir results
```

### 8. Measure efficiency

```bash
python scripts/measure_efficiency.py --output-root <OUTPUT_ROOT> --data-root <DATA_ROOT> --results-dir results
```



