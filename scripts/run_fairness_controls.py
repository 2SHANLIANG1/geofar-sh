# Release wrapper copied from original source: scripts/run_paper_fairness_controls.py
# Private local paths were sanitized for the GitHub release package.
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(r"python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

GAS_ROOT = Path(r"<OUTPUT_ROOT>")
GAS_OUTPUT = GAS_ROOT / "output"
GAS_OUTPUT_CKPT30K = GAS_ROOT / "output_ckpt30k"
GS_OUTPUT = REPO / "output"
OUTPUT_ROOT = GS_OUTPUT / "paper_fairness_controls"

DEFAULT_SCENES = [
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
    "train",
    "truck",
    "drjohnson",
    "playroom",
]
DEFAULT_VARIANTS = ["3dgs_40k_cont", "sh_only_10k", "app_only_10k", "geofar_sh_ours"]
BASELINE_VARIANT = "3dgs_30k_baseline"
TARGET_ITER = 40000

SCENE_DATA = {
    "bicycle": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\bicycle")),
    "bonsai": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\bonsai")),
    "counter": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\counter")),
    "flowers": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\flowers")),
    "garden": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\garden")),
    "kitchen": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\kitchen")),
    "room": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\room")),
    "stump": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\stump")),
    "treehill": ("Mip-NeRF360", Path(r"<DATA_ROOT>\\mipnerf360\\treehill")),
    "train": ("Tanks&Temples", Path(r"<DATA_ROOT>\\tandt\\train")),
    "truck": ("Tanks&Temples", Path(r"<DATA_ROOT>\\tandt\\truck")),
    "drjohnson": ("DeepBlending", Path(r"<DATA_ROOT>\\db\\drjohnson")),
    "playroom": ("DeepBlending", Path(r"<DATA_ROOT>\\db\\playroom")),
}

REQUIRED_ARTIFACTS = [
    "results.json",
    "metrics.json",
    "compare.csv",
    "train_stdout.log",
    "train_stderr.log",
    "render_stdout.log",
    "render_stderr.log",
    "metrics_stdout.log",
    "metrics_stderr.log",
    "time_memory.json",
    "geometry_sh_delta.json",
]


@dataclass
class BaselineResolution:
    metrics_path: Optional[Path]
    metrics_kind: str
    metrics_iteration: Optional[int]
    candidates: list[Path] = field(default_factory=list)
    searched_locations: list[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class CheckpointResolution:
    selected_path: Optional[Path]
    source: str
    candidates: list[Path] = field(default_factory=list)
    searched_locations: list[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class SceneMeta:
    scene: str
    dataset: str
    source_path: Path
    images: str
    white_background: bool
    sh_degree: int
    gas_output_dir: Path
    data_path_exists: bool
    baseline: BaselineResolution
    checkpoint: CheckpointResolution


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_signed(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}"


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def parse_cfg_args(path: Path) -> Dict[str, Any]:
    from argparse import Namespace

    return vars(eval(read_text(path).strip(), {"Namespace": Namespace}))


def find_all_files(root: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return unique_paths(found)


def extract_iteration_from_name(name: str) -> Optional[int]:
    digits = "".join(ch if ch.isdigit() else " " for ch in name)
    values = [int(part) for part in digits.split() if part.isdigit()]
    return max(values) if values else None


def parse_metric_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = read_json(path)
    except Exception:
        return None

    if isinstance(data, dict):
        if {"PSNR", "SSIM", "LPIPS"}.issubset(data.keys()):
            iteration = safe_float(data.get("iteration"))
            return {
                "PSNR": float(data["PSNR"]),
                "SSIM": float(data["SSIM"]),
                "LPIPS": float(data["LPIPS"]),
                "iteration": int(iteration) if iteration is not None else extract_iteration_from_name(path.as_posix()),
                "kind": "direct_metrics",
            }

        best_iter = -1
        best_item: Optional[dict[str, Any]] = None
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            if {"PSNR", "SSIM", "LPIPS"}.issubset(value.keys()):
                iteration = extract_iteration_from_name(key) or -1
                if iteration > best_iter:
                    best_iter = iteration
                    best_item = value
            else:
                for nested_key, nested_value in value.items():
                    if not isinstance(nested_value, dict):
                        continue
                    if not {"PSNR", "SSIM", "LPIPS"}.issubset(nested_value.keys()):
                        continue
                    iteration = extract_iteration_from_name(nested_key) or extract_iteration_from_name(key) or -1
                    if iteration > best_iter:
                        best_iter = iteration
                        best_item = nested_value
        if best_item is not None:
            return {
                "PSNR": float(best_item["PSNR"]),
                "SSIM": float(best_item["SSIM"]),
                "LPIPS": float(best_item["LPIPS"]),
                "iteration": best_iter if best_iter >= 0 else None,
                "kind": "results_dict",
            }
    return None


def score_baseline_candidate(path: Path) -> tuple[int, int, int]:
    parsed = parse_metric_file(path)
    iteration = int(parsed["iteration"]) if parsed and parsed.get("iteration") is not None else -1
    text = path.as_posix().lower()
    rank = 0
    if text.endswith("/results.json"):
        rank += 20
    if "/test/ours_30000/" in text:
        rank += 30
    if "ours_30000" in text or "30000" in text:
        rank += 40
    if iteration == 30000:
        rank += 100
    elif iteration > 0:
        rank -= abs(iteration - 30000)
    return (rank, iteration, int(path.stat().st_mtime))


def resolve_baseline_metrics(gas_dir: Path) -> BaselineResolution:
    direct_candidates = [
        gas_dir / "results.json",
        gas_dir / "metrics.json",
        gas_dir / "test" / "ours_30000" / "results.json",
        gas_dir / "test" / "ours_30000" / "metrics.json",
    ]
    searched_locations = [str(path) for path in direct_candidates]
    valid_direct = [path for path in direct_candidates if path.exists() and parse_metric_file(path) is not None]
    if valid_direct:
        selected = max(valid_direct, key=score_baseline_candidate)
        parsed = parse_metric_file(selected)
        return BaselineResolution(
            metrics_path=selected,
            metrics_kind=str(parsed["kind"]),
            metrics_iteration=parsed.get("iteration"),
            candidates=valid_direct,
            searched_locations=searched_locations,
        )

    recursive_candidates = [
        path
        for path in find_all_files(gas_dir, ["results.json", "metrics.json"])
        if parse_metric_file(path) is not None
    ]
    if recursive_candidates:
        selected = max(recursive_candidates, key=score_baseline_candidate)
        parsed = parse_metric_file(selected)
        searched_locations.append(f"recursive search under {gas_dir}")
        return BaselineResolution(
            metrics_path=selected,
            metrics_kind=str(parsed["kind"]),
            metrics_iteration=parsed.get("iteration"),
            candidates=recursive_candidates,
            searched_locations=searched_locations,
        )

    searched_locations.append(f"recursive search under {gas_dir}")
    return BaselineResolution(
        metrics_path=None,
        metrics_kind="missing",
        metrics_iteration=None,
        searched_locations=searched_locations,
        reason="no readable baseline results.json or metrics.json found",
    )


def checkpoint_sort_key(scene: str, path: Path) -> tuple[int, int, int, int]:
    scene_lower = scene.lower()
    text = path.as_posix().lower()
    name_iter = extract_iteration_from_name(path.name) or -1
    scene_rank = 1 if scene_lower in text else 0
    source_rank = 0
    if text == (GAS_OUTPUT / scene / "chkpnt30000.pth").as_posix().lower():
        source_rank = 50
    elif text == (GAS_OUTPUT_CKPT30K / scene / "chkpnt30000.pth").as_posix().lower():
        source_rank = 40
    elif text == (GAS_OUTPUT / scene / "checkpoints" / "chkpnt30000.pth").as_posix().lower():
        source_rank = 30
    elif text.startswith((GAS_OUTPUT / scene).as_posix().lower()):
        source_rank = 20
    elif text.startswith(GS_OUTPUT.as_posix().lower()):
        source_rank = 10
    return (source_rank, scene_rank, name_iter, int(path.stat().st_mtime))


def resolve_checkpoint(scene: str, gas_dir: Path) -> CheckpointResolution:
    searched_locations = [
        str(gas_dir / "chkpnt30000.pth"),
        str(GAS_OUTPUT_CKPT30K / scene / "chkpnt30000.pth"),
        str(gas_dir / "checkpoints" / "chkpnt30000.pth"),
        f"recursive *.pth search under {gas_dir} with filename containing 30000",
        f"recursive *.pth search under {GS_OUTPUT} with path containing {scene}",
    ]
    candidates: list[Path] = []

    priority_paths = [
        gas_dir / "chkpnt30000.pth",
        GAS_OUTPUT_CKPT30K / scene / "chkpnt30000.pth",
        gas_dir / "checkpoints" / "chkpnt30000.pth",
    ]
    for path in priority_paths:
        if path.exists():
            candidates.append(path)

    if gas_dir.exists():
        candidates.extend([path for path in gas_dir.rglob("*.pth") if "30000" in path.name])
        candidates.extend([path for path in gas_dir.rglob("*.pth.tar") if "30000" in path.name])
    if GS_OUTPUT.exists():
        candidates.extend(
            [
                path
                for path in GS_OUTPUT.rglob("*.pth")
                if "30000" in path.name and scene.lower() in path.as_posix().lower()
            ]
        )
        candidates.extend(
            [
                path
                for path in GS_OUTPUT.rglob("*.pth.tar")
                if "30000" in path.name and scene.lower() in path.as_posix().lower()
            ]
        )

    candidates = unique_paths(candidates)
    if not candidates:
        return CheckpointResolution(
            selected_path=None,
            source="missing",
            searched_locations=searched_locations,
            reason="no chkpnt30000 candidate found in configured search paths",
        )

    selected = max(candidates, key=lambda path: checkpoint_sort_key(scene, path))
    selected_text = selected.as_posix().lower()
    if selected_text == (gas_dir / "chkpnt30000.pth").as_posix().lower():
        source = "gas_output_root"
    elif selected_text == (GAS_OUTPUT_CKPT30K / scene / "chkpnt30000.pth").as_posix().lower():
        source = "gas_output_ckpt30k"
    elif selected_text == (gas_dir / "checkpoints" / "chkpnt30000.pth").as_posix().lower():
        source = "gas_output_checkpoints"
    elif selected_text.startswith(gas_dir.as_posix().lower()):
        source = "gas_output_recursive"
    else:
        source = "gs_output_recursive"
    return CheckpointResolution(
        selected_path=selected,
        source=source,
        candidates=candidates,
        searched_locations=searched_locations,
    )


def discover_scene_meta(scenes: Iterable[str]) -> tuple[list[SceneMeta], list[dict[str, Any]]]:
    metas: list[SceneMeta] = []
    issues: list[dict[str, Any]] = []

    for scene in scenes:
        scene_key = scene.lower()
        if scene_key not in SCENE_DATA:
            issues.append({"scene": scene, "reason": "unknown scene"})
            continue

        dataset, data_path = SCENE_DATA[scene_key]
        gas_dir = GAS_OUTPUT / scene
        cfg_path = gas_dir / "cfg_args"

        images = "images"
        white_background = dataset == "DeepBlending"
        sh_degree = 3
        if cfg_path.exists():
            try:
                cfg = parse_cfg_args(cfg_path)
                images = str(cfg.get("images", images))
                white_background = bool(cfg.get("white_background", white_background))
                sh_degree = int(cfg.get("sh_degree", sh_degree))
            except Exception as exc:
                issues.append({"scene": scene, "reason": f"failed to parse cfg_args: {exc}", "cfg_args": str(cfg_path)})

        baseline = resolve_baseline_metrics(gas_dir)
        checkpoint = resolve_checkpoint(scene, gas_dir)
        metas.append(
            SceneMeta(
                scene=scene,
                dataset=dataset,
                source_path=data_path,
                images=images,
                white_background=white_background,
                sh_degree=sh_degree,
                gas_output_dir=gas_dir,
                data_path_exists=data_path.exists(),
                baseline=baseline,
                checkpoint=checkpoint,
            )
        )
    return metas, issues


def variant_specs() -> Dict[str, Dict[str, Any]]:
    stage2_common = [
        "--appearance_stage2_start", "30001",
        "--appearance_stage2_iters", "10000",
        "--appearance_residual_enable_step", "30001",
        "--appearance_residual_warmup_steps", "1500",
        "--appearance_lambda_warmup_iters", "1500",
        "--stage2_geom_unfreeze_iter", "0",
        "--disable_densify_stage2", "true",
        "--disable_prune_stage2", "true",
        "--freeze_xyz", "true",
        "--freeze_scaling", "true",
        "--freeze_rotation", "true",
        "--freeze_opacity", "true",
        "--freeze_exposure", "true",
        "--stage2_lr_f_dc", "0.0001",
        "--stage2_lr_f_rest", "0.0001",
        "--lr_appearance_latent", "0.0005",
        "--lr_fastkan", "0.0001",
        "--lr_fastkan_gate", "0.00005",
        "--residual_scale", "0.10",
        "--lambda_residual_reg", "1e-05",
        "--lambda_gate_reg", "1e-06",
        "--lambda_diffuse_consistency", "5e-05",
        "--lambda_branch_diversity", "1e-06",
    ]
    app_common = [
        "--enable_appearance_residual", "true",
        "--use_appearance_residual",
        "--appearance_latent_dim", "8",
        "--enable_diffuse_residual", "true",
        "--enable_specular_residual", "true",
        "--enable_specular_mask", "true",
        "--enable_global_gate", "true",
        "--residual_mode", "full",
    ]

    return {
        "3dgs_40k_cont": {
            "formal_name": "3DGS-40k-cont",
            "geometry_frozen": False,
            "sh_refine": False,
            "appearance_residual": False,
            "total_iters": 40000,
            "args": [],
        },
        "sh_only_10k": {
            "formal_name": "SH-only-10k",
            "geometry_frozen": True,
            "sh_refine": True,
            "appearance_residual": False,
            "total_iters": 40000,
            "args": [
                *stage2_common,
                "--enable_appearance_residual", "false",
                "--enable_sh_refine", "true",
                "--stage2_refine_sh",
            ],
        },
        "app_only_10k": {
            "formal_name": "App-only-10k",
            "geometry_frozen": True,
            "sh_refine": False,
            "appearance_residual": True,
            "total_iters": 40000,
            "args": [
                *stage2_common,
                *app_common,
                "--enable_sh_refine", "false",
            ],
        },
        "geofar_sh_ours": {
            "formal_name": "GeoFAR-SH",
            "geometry_frozen": True,
            "sh_refine": True,
            "appearance_residual": True,
            "total_iters": 40000,
            "args": [
                *stage2_common,
                *app_common,
                "--enable_sh_refine", "true",
                "--stage2_refine_sh",
            ],
        },
    }


def variant_output_dir(meta: SceneMeta, variant: str) -> Path:
    return OUTPUT_ROOT / meta.scene / variant


def required_paths(run_dir: Path) -> list[Path]:
    return [run_dir / name for name in REQUIRED_ARTIFACTS]


def variant_complete(run_dir: Path) -> bool:
    return all(path.exists() for path in required_paths(run_dir))


def has_saved_iteration(run_dir: Path, iteration: int) -> bool:
    ply = run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return ply.exists()


def has_render_outputs(run_dir: Path, iteration: int) -> bool:
    renders_dir = run_dir / "test" / f"ours_{iteration}" / "renders"
    gt_dir = run_dir / "test" / f"ours_{iteration}" / "gt"
    return renders_dir.exists() and gt_dir.exists() and any(renders_dir.glob("*.png")) and any(gt_dir.glob("*.png"))


def build_train_command(meta: SceneMeta, run_dir: Path, start_checkpoint: Path, extra_args: list[str]) -> list[str]:
    command = [
        str(PYTHON),
        "train.py",
        "-s", str(meta.source_path),
        "-m", str(run_dir),
        "-i", str(meta.images),
        "--eval",
        "--sh_degree", str(meta.sh_degree),
        "--iterations", str(TARGET_ITER),
        "--test_iterations", str(TARGET_ITER),
        "--save_iterations", str(TARGET_ITER),
        "--checkpoint_iterations", "35000", "37500", str(TARGET_ITER),
        "--start_checkpoint", str(start_checkpoint),
        "--disable_viewer",
        *extra_args,
    ]
    if meta.white_background:
        command.append("--white_background")
    return command


def build_render_command(run_dir: Path, iteration: int) -> list[str]:
    return [str(PYTHON), "render.py", "-m", str(run_dir), "--iteration", str(iteration), "--skip_train"]


def build_metrics_command(run_dir: Path) -> list[str]:
    return [str(PYTHON), "metrics.py", "-m", str(run_dir)]


def run_logged(command: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    write_text(stdout_path, proc.stdout)
    write_text(stderr_path, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\nstdout={stdout_path}\nstderr={stderr_path}"
        )


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoints = unique_paths(list(run_dir.glob("chkpnt*.pth")) + list(run_dir.glob("chkpnt*.pth.tar")))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: extract_iteration_from_name(path.name) or -1)


def parse_run_metrics(results_path: Path) -> Dict[str, float]:
    parsed = parse_metric_file(results_path)
    if parsed is None:
        raise RuntimeError(f"Could not parse metrics from {results_path}")
    return {
        "iteration": int(parsed["iteration"]) if parsed.get("iteration") is not None else extract_iteration_from_name(results_path.name) or 0,
        "PSNR": float(parsed["PSNR"]),
        "SSIM": float(parsed["SSIM"]),
        "LPIPS": float(parsed["LPIPS"]),
    }


def load_checkpoint_capture(path: Path) -> tuple[list[Any], int]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], int):
        return list(obj[0]), int(obj[1])
    if isinstance(obj, tuple):
        return list(obj), -1
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def tensor_mean_max_abs(a: Any, b: Any) -> tuple[float, float]:
    import torch

    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return (0.0, 0.0)
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.ndim >= 1 and b.ndim >= 1 and a.shape[0] != b.shape[0]:
        n = min(int(a.shape[0]), int(b.shape[0]))
        a = a[:n]
        b = b[:n]
    if a.shape != b.shape:
        common_dims = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
        if not common_dims:
            return (0.0, 0.0)
        slices = tuple(slice(0, size) for size in common_dims)
        a = a[slices]
        b = b[slices]
    delta = (b - a).abs()
    return float(delta.mean().item()), float(delta.max().item())


def tensor_abs_stats(value: Any) -> tuple[float, float]:
    import torch

    if not torch.is_tensor(value):
        return (0.0, 0.0)
    delta = value.detach().float().cpu().abs()
    if delta.numel() == 0:
        return (0.0, 0.0)
    return float(delta.mean().item()), float(delta.max().item())


def compute_delta_stats(start_ckpt: Path, end_ckpt: Path) -> Dict[str, Any]:
    start_state, _ = load_checkpoint_capture(start_ckpt)
    end_state, _ = load_checkpoint_capture(end_ckpt)
    xyz_mean, xyz_max = tensor_mean_max_abs(start_state[1], end_state[1])
    fdc_mean, fdc_max = tensor_mean_max_abs(start_state[2], end_state[2])
    frest_mean, frest_max = tensor_mean_max_abs(start_state[3], end_state[3])
    scaling_mean, scaling_max = tensor_mean_max_abs(start_state[4], end_state[4])
    rotation_mean, rotation_max = tensor_mean_max_abs(start_state[5], end_state[5])
    opacity_mean, opacity_max = tensor_mean_max_abs(start_state[6], end_state[6])

    appearance_groups = {
        "appearance_latent": 13,
        "app_w_rgb": 14,
        "app_b_rgb": 15,
        "app_w_gate": 16,
        "app_b_gate": 17,
        "app_w_diff": 18,
        "app_b_diff": 19,
        "app_w_spec": 20,
        "app_b_spec": 21,
        "app_w_mask": 22,
        "app_b_mask": 23,
        "app_w2_gate": 24,
        "app_b2_gate": 25,
        "app_w2_diff": 26,
        "app_b2_diff": 27,
        "app_w2_spec": 28,
        "app_b2_spec": 29,
        "app_w2_mask": 30,
        "app_b2_mask": 31,
    }
    appearance_delta: Dict[str, Dict[str, float]] = {}
    appearance_means: list[float] = []
    appearance_maxes: list[float] = []
    for name, idx in appearance_groups.items():
        start_tensor = start_state[idx] if idx < len(start_state) else None
        end_tensor = end_state[idx] if idx < len(end_state) else None
        if start_tensor is None and end_tensor is None:
            continue
        if start_tensor is None:
            mean_abs, max_abs = tensor_abs_stats(end_tensor)
        elif end_tensor is None:
            mean_abs, max_abs = tensor_abs_stats(start_tensor)
        else:
            mean_abs, max_abs = tensor_mean_max_abs(start_tensor, end_tensor)
        appearance_delta[name] = {"mean_abs": mean_abs, "max_abs": max_abs}
        appearance_means.append(mean_abs)
        appearance_maxes.append(max_abs)

    return {
        "delta_xyz": xyz_mean,
        "delta_xyz_max": xyz_max,
        "delta_scale": scaling_mean,
        "delta_scale_max": scaling_max,
        "delta_rotation": rotation_mean,
        "delta_rotation_max": rotation_max,
        "delta_opacity": opacity_mean,
        "delta_opacity_max": opacity_max,
        "delta_sh_dc": fdc_mean,
        "delta_sh_dc_max": fdc_max,
        "delta_sh_rest": frest_mean,
        "delta_sh_rest_max": frest_max,
        "delta_sh": 0.5 * (fdc_mean + frest_mean),
        "delta_sh_max": max(fdc_max, frest_max),
        "gaussian_count_before": int(start_state[1].shape[0]),
        "gaussian_count_after": int(end_state[1].shape[0]),
        "appearance_delta": appearance_delta,
        "appearance_delta_mean": sum(appearance_means) / len(appearance_means) if appearance_means else 0.0,
        "appearance_delta_max": max(appearance_maxes) if appearance_maxes else 0.0,
    }


def read_timing_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_json = run_dir / "metrics.json"
    if not metrics_json.exists():
        return {}
    try:
        metrics_state = read_json(metrics_json)
    except Exception:
        return {}
    timing = metrics_state.get("timing") or {}
    memory = metrics_state.get("memory") or {}
    return {
        "stage2_time_sec": timing.get("stage2_time_sec"),
        "total_train_sec": timing.get("total_train_sec"),
        "peak_gpu_mem_mb": memory.get("peak_gpu_mem_mb", memory.get("peak_gpu_reserved_mb")),
        "peak_gpu_reserved_mb": memory.get("peak_gpu_reserved_mb"),
        "final_num_gaussians": metrics_state.get("final_num_gaussians"),
    }


def write_time_memory_json(run_dir: Path) -> Dict[str, Any]:
    timing = read_timing_metrics(run_dir)
    payload = {
        "stage2_time_sec": timing.get("stage2_time_sec"),
        "stage2_time_min": (safe_float(timing.get("stage2_time_sec")) or 0.0) / 60.0 if timing.get("stage2_time_sec") is not None else None,
        "total_train_sec": timing.get("total_train_sec"),
        "peak_gpu_mem_mb": timing.get("peak_gpu_mem_mb"),
        "peak_gpu_reserved_mb": timing.get("peak_gpu_reserved_mb"),
        "final_num_gaussians": timing.get("final_num_gaussians"),
    }
    write_json(run_dir / "time_memory.json", payload)
    return payload


def write_compare_csv(
    run_dir: Path,
    meta: SceneMeta,
    variant: str,
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    notes: str,
) -> None:
    headers = [
        "dataset",
        "scene",
        "variant",
        "baseline_psnr",
        "psnr",
        "delta_psnr",
        "baseline_ssim",
        "ssim",
        "delta_ssim",
        "baseline_lpips",
        "lpips",
        "delta_lpips",
        "notes",
    ]
    row = {
        "dataset": meta.dataset,
        "scene": meta.scene,
        "variant": variant,
        "baseline_psnr": baseline_metrics["PSNR"],
        "psnr": current_metrics["PSNR"],
        "delta_psnr": current_metrics["PSNR"] - baseline_metrics["PSNR"],
        "baseline_ssim": baseline_metrics["SSIM"],
        "ssim": current_metrics["SSIM"],
        "delta_ssim": current_metrics["SSIM"] - baseline_metrics["SSIM"],
        "baseline_lpips": baseline_metrics["LPIPS"],
        "lpips": current_metrics["LPIPS"],
        "delta_lpips": current_metrics["LPIPS"] - baseline_metrics["LPIPS"],
        "notes": notes,
    }
    with (run_dir / "compare.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def materialize_baseline(meta: SceneMeta) -> dict[str, Any]:
    run_dir = variant_output_dir(meta, BASELINE_VARIANT)
    run_dir.mkdir(parents=True, exist_ok=True)
    if meta.baseline.metrics_path is None:
        return {
            "scene": meta.scene,
            "variant": BASELINE_VARIANT,
            "status": "blocked",
            "stage": "baseline",
            "reason": "missing baseline metrics",
            "run_dir": str(run_dir),
        }

    baseline_metrics = parse_run_metrics(meta.baseline.metrics_path)
    shutil.copy2(meta.baseline.metrics_path, run_dir / "results.json")
    write_json(
        run_dir / "metrics.json",
        {
            "variant": BASELINE_VARIANT,
            "timing": {"stage2_time_sec": 0.0, "total_train_sec": 0.0},
            "memory": {"peak_gpu_mem_mb": None, "peak_gpu_reserved_mb": None},
            "final_num_gaussians": None,
            "baseline_metrics_source": str(meta.baseline.metrics_path),
        },
    )
    write_json(
        run_dir / "geometry_sh_delta.json",
        {
            "delta_xyz": 0.0,
            "delta_scale": 0.0,
            "delta_rotation": 0.0,
            "delta_opacity": 0.0,
            "delta_sh": 0.0,
            "appearance_delta": {},
            "appearance_delta_mean": 0.0,
            "appearance_delta_max": 0.0,
        },
    )
    write_time_memory_json(run_dir)
    write_text(run_dir / "train_stdout.log", "baseline 3DGS-30k reused from <OUTPUT_ROOT>; no training executed.\n")
    write_text(run_dir / "train_stderr.log", "")
    write_text(run_dir / "render_stdout.log", "baseline offline render reused from read-only baseline source.\n")
    write_text(run_dir / "render_stderr.log", "")
    write_text(run_dir / "metrics_stdout.log", "baseline metrics reused from read-only baseline source.\n")
    write_text(run_dir / "metrics_stderr.log", "")
    write_compare_csv(run_dir, meta, BASELINE_VARIANT, baseline_metrics, baseline_metrics, f"baseline_metrics_path={meta.baseline.metrics_path}")
    return {
        "scene": meta.scene,
        "variant": BASELINE_VARIANT,
        "status": "done",
        "stage": "baseline",
        "reason": "reused read-only baseline metrics",
        "run_dir": str(run_dir),
    }


def ensure_render_and_metrics(meta: SceneMeta, run_dir: Path) -> None:
    if not has_render_outputs(run_dir, TARGET_ITER):
        run_logged(
            build_render_command(run_dir, TARGET_ITER),
            REPO,
            run_dir / "render_stdout.log",
            run_dir / "render_stderr.log",
        )
    elif not (run_dir / "render_stdout.log").exists():
        write_text(run_dir / "render_stdout.log", "render outputs already existed; render step skipped.\n")
        write_text(run_dir / "render_stderr.log", "")

    if not (run_dir / "results.json").exists():
        run_logged(
            build_metrics_command(run_dir),
            REPO,
            run_dir / "metrics_stdout.log",
            run_dir / "metrics_stderr.log",
        )
    elif not (run_dir / "metrics_stdout.log").exists():
        write_text(run_dir / "metrics_stdout.log", "results.json already existed; metrics step skipped.\n")
        write_text(run_dir / "metrics_stderr.log", "")

    if not (run_dir / "metrics.json").exists():
        write_json(
            run_dir / "metrics.json",
            {
                "variant": run_dir.name,
                "timing": {},
                "memory": {},
                "final_num_gaussians": None,
                "notes": "placeholder metrics.json generated because training metrics were unavailable",
            },
        )


def run_variant(meta: SceneMeta, variant: str, clean: bool, skip_existing: bool) -> dict[str, Any]:
    spec = variant_specs()[variant]
    run_dir = variant_output_dir(meta, variant)
    current_baseline = resolve_baseline_metrics(meta.gas_output_dir)
    current_checkpoint = resolve_checkpoint(meta.scene, meta.gas_output_dir)

    if clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and variant_complete(run_dir):
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "skipped_existing",
            "stage": "skip",
            "reason": "all required artifacts already exist",
            "run_dir": str(run_dir),
        }
    if current_baseline.metrics_path is None:
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "blocked",
            "stage": "baseline",
            "reason": "missing baseline metrics",
            "run_dir": str(run_dir),
        }
    if current_checkpoint.selected_path is None:
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "blocked",
            "stage": "checkpoint",
            "reason": "found baseline result but no resumable 30k checkpoint",
            "run_dir": str(run_dir),
        }
    if not meta.data_path_exists:
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "blocked",
            "stage": "dataset",
            "reason": f"missing data path: {meta.source_path}",
            "run_dir": str(run_dir),
        }

    try:
        if not has_saved_iteration(run_dir, TARGET_ITER):
            start_checkpoint = latest_checkpoint(run_dir) or current_checkpoint.selected_path
            run_logged(
                build_train_command(meta, run_dir, start_checkpoint, list(spec["args"])),
                REPO,
                run_dir / "train_stdout.log",
                run_dir / "train_stderr.log",
            )
        elif not (run_dir / "train_stdout.log").exists():
            write_text(run_dir / "train_stdout.log", "iteration_40000 model already existed; training step skipped.\n")
            write_text(run_dir / "train_stderr.log", "")

        ensure_render_and_metrics(meta, run_dir)

        final_ckpt = latest_checkpoint(run_dir)
        delta_stats: Dict[str, Any] = {}
        if final_ckpt is not None:
            delta_stats = compute_delta_stats(current_checkpoint.selected_path, final_ckpt)
        write_json(run_dir / "geometry_sh_delta.json", delta_stats)
        write_time_memory_json(run_dir)

        baseline_metrics = parse_run_metrics(current_baseline.metrics_path)
        current_metrics = parse_run_metrics(run_dir / "results.json")
        write_compare_csv(
            run_dir,
            meta,
            variant,
            baseline_metrics,
            current_metrics,
            f"source_checkpoint={current_checkpoint.selected_path}",
        )
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "done",
            "stage": "complete",
            "reason": "",
            "run_dir": str(run_dir),
        }
    except Exception as exc:
        return {
            "scene": meta.scene,
            "variant": variant,
            "status": "failed",
            "stage": "runtime",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir),
            "train_stdout_log": str(run_dir / "train_stdout.log"),
            "train_stderr_log": str(run_dir / "train_stderr.log"),
            "render_stdout_log": str(run_dir / "render_stdout.log"),
            "render_stderr_log": str(run_dir / "render_stderr.log"),
            "metrics_stdout_log": str(run_dir / "metrics_stdout.log"),
            "metrics_stderr_log": str(run_dir / "metrics_stderr.log"),
        }


def write_missing_data_paths(root: Path, metas: list[SceneMeta]) -> None:
    missing = [meta for meta in metas if not meta.data_path_exists]
    lines = ["# Missing Data Paths", ""]
    if not missing:
        lines.append("No missing dataset paths.")
    else:
        for meta in missing:
            lines.append(f"- {meta.scene}: `{meta.source_path}`")
    write_text(root / "missing_data_paths.md", "\n".join(lines) + "\n")


def dry_run_plan(root: Path, metas: list[SceneMeta], variants: list[str], skip_existing: bool) -> None:
    specs = variant_specs()
    lines = ["# Dry Run Plan", ""]
    payload: dict[str, Any] = {"output_root": str(root), "scenes": []}

    for meta in metas:
        lines.append(f"## {meta.scene} ({meta.dataset})")
        lines.append("")
        lines.append(f"- dataset path: `{meta.source_path}`")
        lines.append(f"- dataset path exists: `{meta.data_path_exists}`")
        lines.append(f"- baseline metrics path: `{meta.baseline.metrics_path}`" if meta.baseline.metrics_path else "- baseline metrics path: missing")
        lines.append(f"- checkpoint path: `{meta.checkpoint.selected_path}`" if meta.checkpoint.selected_path else "- checkpoint path: missing")
        lines.append("- checkpoint search priority:")
        for item in meta.checkpoint.searched_locations:
            lines.append(f"  - `{item}`")
        lines.append("")

        scene_payload = {
            "scene": meta.scene,
            "dataset": meta.dataset,
            "dataset_path": str(meta.source_path),
            "data_path_exists": meta.data_path_exists,
            "baseline_metrics_path": str(meta.baseline.metrics_path) if meta.baseline.metrics_path else None,
            "resolved_checkpoint": str(meta.checkpoint.selected_path) if meta.checkpoint.selected_path else None,
            "checkpoint_source": meta.checkpoint.source,
            "variants": [],
        }

        for variant in variants:
            run_dir = variant_output_dir(meta, variant)
            complete = variant_complete(run_dir)
            if skip_existing and complete:
                status = "skip-existing"
                reason = "all required artifacts already exist"
            elif meta.baseline.metrics_path is None:
                status = "blocked"
                reason = "missing baseline metrics"
            elif not meta.data_path_exists:
                status = "blocked"
                reason = f"missing data path: {meta.source_path}"
            elif meta.checkpoint.selected_path is None:
                status = "blocked"
                reason = "missing resumable 30k checkpoint"
            else:
                status = "ready"
                reason = ""

            lines.append(f"### {variant}")
            lines.append("")
            lines.append(f"- output dir: `{run_dir}`")
            lines.append(f"- status: `{status}`")
            if reason:
                lines.append(f"- reason: {reason}")
            if status == "ready":
                train_cmd = build_train_command(meta, run_dir, meta.checkpoint.selected_path, list(specs[variant]["args"]))
                lines.append(f"- train: `{json.dumps(train_cmd, ensure_ascii=False)}`")
                lines.append(f"- render: `{json.dumps(build_render_command(run_dir, TARGET_ITER), ensure_ascii=False)}`")
                lines.append(f"- metrics: `{json.dumps(build_metrics_command(run_dir), ensure_ascii=False)}`")
            lines.append("")

            scene_payload["variants"].append(
                {
                    "variant": variant,
                    "output_dir": str(run_dir),
                    "status": status,
                    "reason": reason,
                    "complete": complete,
                }
            )
        payload["scenes"].append(scene_payload)

    write_text(root / "dry_run_plan.md", "\n".join(lines) + "\n")
    write_json(root / "dry_run_plan.json", payload)
    print(read_text(root / "dry_run_plan.md"))


def write_failed_jobs(root: Path, records: list[dict[str, Any]]) -> None:
    failed = [record for record in records if record["status"] in {"failed", "blocked"}]
    lines = ["# Failed Jobs", ""]
    if not failed:
        lines.append("No failed or blocked jobs.")
    else:
        for record in failed:
            lines.append(f"## {record['scene']} / {record['variant']}")
            lines.append("")
            lines.append(f"- status: `{record['status']}`")
            lines.append(f"- stage: `{record.get('stage', '')}`")
            lines.append(f"- reason: {record.get('reason', '')}")
            run_dir = record.get("run_dir")
            if run_dir:
                lines.append(f"- run dir: `{run_dir}`")
            for key in [
                "train_stdout_log",
                "train_stderr_log",
                "render_stdout_log",
                "render_stderr_log",
                "metrics_stdout_log",
                "metrics_stderr_log",
            ]:
                if key in record:
                    lines.append(f"- {key}: `{record[key]}`")
            lines.append("")
    write_text(root / "failed_jobs.md", "\n".join(lines) + "\n")


def write_completed_jobs(root: Path, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record["status"] in {"done", "skipped_existing"}]
    lines = ["# Completed Jobs", ""]
    if not completed:
        lines.append("No completed jobs.")
    else:
        for record in completed:
            lines.append(f"- {record['scene']} / {record['variant']} / {record['status']}")
    write_text(root / "completed_jobs.md", "\n".join(lines) + "\n")


def update_running_status(root: Path, records: list[dict[str, Any]], current: Optional[dict[str, Any]] = None) -> None:
    total = len(records)
    done = sum(1 for record in records if record["status"] == "done")
    skipped = sum(1 for record in records if record["status"] == "skipped_existing")
    failed = sum(1 for record in records if record["status"] == "failed")
    blocked = sum(1 for record in records if record["status"] == "blocked")
    lines = [
        "# Running Status",
        "",
        f"- total recorded jobs: {total}",
        f"- done: {done}",
        f"- skipped_existing: {skipped}",
        f"- failed: {failed}",
        f"- blocked: {blocked}",
    ]
    if current is not None:
        lines.extend(
            [
                "",
                "## Latest Update",
                "",
                f"- scene: `{current.get('scene', '')}`",
                f"- variant: `{current.get('variant', '')}`",
                f"- status: `{current.get('status', '')}`",
                f"- stage: `{current.get('stage', '')}`",
                f"- reason: {current.get('reason', '')}",
            ]
        )
    lines.extend(["", "## Recent Jobs", ""])
    for record in records[-15:]:
        lines.append(
            f"- {record['scene']} / {record['variant']} / {record['status']} / {record.get('stage', '')} / {record.get('reason', '')}"
        )
    write_text(root / "RUNNING_STATUS.md", "\n".join(lines) + "\n")


def scene_variant_outcome_map(records: list[dict[str, Any]]) -> Dict[tuple[str, str], dict[str, Any]]:
    mapping: Dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        mapping[(record["scene"], record["variant"])] = record
    return mapping


def summarize_scene_level(root: Path, metas: list[SceneMeta], requested_variants: list[str], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcome_map = scene_variant_outcome_map(records)
    specs = variant_specs()
    variants = [BASELINE_VARIANT, *requested_variants]

    for meta in metas:
        baseline_path_text = str(meta.baseline.metrics_path) if meta.baseline.metrics_path else ""
        for variant in variants:
            run_dir = variant_output_dir(meta, variant)
            result_path = run_dir / "results.json"
            result_metrics = parse_run_metrics(result_path) if result_path.exists() and parse_metric_file(result_path) is not None else None
            time_memory = read_json(run_dir / "time_memory.json") if (run_dir / "time_memory.json").exists() else {}
            delta_stats = read_json(run_dir / "geometry_sh_delta.json") if (run_dir / "geometry_sh_delta.json").exists() else {}
            outcome = outcome_map.get((meta.scene, variant), {})

            if variant == BASELINE_VARIANT:
                geometry_frozen = False
                sh_refine = False
                appearance_residual = False
                total_iters = 30000
                source_checkpoint = ""
            else:
                spec = specs[variant]
                geometry_frozen = spec["geometry_frozen"]
                sh_refine = spec["sh_refine"]
                appearance_residual = spec["appearance_residual"]
                total_iters = spec["total_iters"]
                source_checkpoint = str(meta.checkpoint.selected_path) if meta.checkpoint.selected_path else ""

            rows.append(
                {
                    "dataset": meta.dataset,
                    "scene": meta.scene,
                    "variant": variant,
                    "source_checkpoint": source_checkpoint,
                    "baseline_metrics_path": baseline_path_text,
                    "geometry_frozen": geometry_frozen,
                    "sh_refine": sh_refine,
                    "appearance_residual": appearance_residual,
                    "total_iters": total_iters,
                    "psnr": result_metrics["PSNR"] if result_metrics else None,
                    "ssim": result_metrics["SSIM"] if result_metrics else None,
                    "lpips": result_metrics["LPIPS"] if result_metrics else None,
                    "num_gaussians": time_memory.get("final_num_gaussians", delta_stats.get("gaussian_count_after")),
                    "stage2_time_min": time_memory.get("stage2_time_min"),
                    "peak_gpu_mem_mb": time_memory.get("peak_gpu_mem_mb"),
                    "delta_xyz": delta_stats.get("delta_xyz"),
                    "delta_scale": delta_stats.get("delta_scale"),
                    "delta_rotation": delta_stats.get("delta_rotation"),
                    "delta_opacity": delta_stats.get("delta_opacity"),
                    "delta_sh": delta_stats.get("delta_sh"),
                    "appearance_delta": delta_stats.get("appearance_delta_mean"),
                    "status": outcome.get("status", "unknown"),
                    "notes": outcome.get("reason", ""),
                }
            )

    headers = [
        "dataset",
        "scene",
        "variant",
        "source_checkpoint",
        "baseline_metrics_path",
        "geometry_frozen",
        "sh_refine",
        "appearance_residual",
        "total_iters",
        "psnr",
        "ssim",
        "lpips",
        "num_gaussians",
        "stage2_time_min",
        "peak_gpu_mem_mb",
        "delta_xyz",
        "delta_scale",
        "delta_rotation",
        "delta_opacity",
        "delta_sh",
        "appearance_delta",
        "status",
        "notes",
    ]
    with (root / "summary_scene_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def metric_rows_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["psnr"] is not None and row["ssim"] is not None and row["lpips"] is not None]


def average_rows(rows: list[dict[str, Any]]) -> Dict[str, float]:
    return {
        "PSNR": sum(float(row["psnr"]) for row in rows) / len(rows),
        "SSIM": sum(float(row["ssim"]) for row in rows) / len(rows),
        "LPIPS": sum(float(row["lpips"]) for row in rows) / len(rows),
    }


def append_delta_columns(target_rows: list[dict[str, Any]], lookup: Dict[Any, dict[str, Any]], key_builder) -> None:
    refs = [
        (BASELINE_VARIANT, "3DGS-30k"),
        ("3dgs_40k_cont", "3DGS-40k-cont"),
        ("sh_only_10k", "SH-only-10k"),
        ("app_only_10k", "App-only-10k"),
    ]
    for row in target_rows:
        key = key_builder(row)
        for ref_variant, ref_label in refs:
            ref_key = list(key)
            ref_key[-1] = ref_variant
            ref = lookup.get(tuple(ref_key))
            for metric in ["PSNR", "SSIM", "LPIPS"]:
                field = f"螖{metric} vs {ref_label}"
                row[field] = None if ref is None else float(row[metric]) - float(ref[metric])


def summarize_averages(root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = metric_rows_only(rows)
    dataset_groups: Dict[tuple[str, str], list[dict[str, Any]]] = {}
    overall_groups: Dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        dataset_groups.setdefault((row["dataset"], row["variant"]), []).append(row)
        overall_groups.setdefault(row["variant"], []).append(row)

    dataset_rows: list[dict[str, Any]] = []
    dataset_lookup: Dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, variant), group in sorted(dataset_groups.items()):
        avg = average_rows(group)
        row = {"dataset": dataset, "variant": variant, **avg}
        dataset_rows.append(row)
        dataset_lookup[(dataset, variant)] = row
    append_delta_columns(dataset_rows, dataset_lookup, lambda row: [row["dataset"], row["variant"]])

    dataset_headers = [
        "dataset",
        "variant",
        "PSNR",
        "SSIM",
        "LPIPS",
        "螖PSNR vs 3DGS-30k",
        "螖SSIM vs 3DGS-30k",
        "螖LPIPS vs 3DGS-30k",
        "螖PSNR vs 3DGS-40k-cont",
        "螖SSIM vs 3DGS-40k-cont",
        "螖LPIPS vs 3DGS-40k-cont",
        "螖PSNR vs SH-only-10k",
        "螖SSIM vs SH-only-10k",
        "螖LPIPS vs SH-only-10k",
        "螖PSNR vs App-only-10k",
        "螖SSIM vs App-only-10k",
        "螖LPIPS vs App-only-10k",
    ]
    with (root / "summary_dataset_average.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_headers)
        writer.writeheader()
        writer.writerows(dataset_rows)

    overall_rows: list[dict[str, Any]] = []
    overall_lookup: Dict[str, dict[str, Any]] = {}
    for variant, group in sorted(overall_groups.items()):
        avg = average_rows(group)
        row = {"variant": variant, **avg}
        overall_rows.append(row)
        overall_lookup[variant] = row
    append_delta_columns(overall_rows, overall_lookup, lambda row: [row["variant"]])

    overall_headers = [
        "variant",
        "PSNR",
        "SSIM",
        "LPIPS",
        "螖PSNR vs 3DGS-30k",
        "螖SSIM vs 3DGS-30k",
        "螖LPIPS vs 3DGS-30k",
        "螖PSNR vs 3DGS-40k-cont",
        "螖SSIM vs 3DGS-40k-cont",
        "螖LPIPS vs 3DGS-40k-cont",
        "螖PSNR vs SH-only-10k",
        "螖SSIM vs SH-only-10k",
        "螖LPIPS vs SH-only-10k",
        "螖PSNR vs App-only-10k",
        "螖SSIM vs App-only-10k",
        "螖LPIPS vs App-only-10k",
    ]
    with (root / "summary_overall_average.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=overall_headers)
        writer.writeheader()
        writer.writerows(overall_rows)
    return dataset_rows, overall_rows


def get_row(rows: list[dict[str, Any]], variant: str) -> Optional[dict[str, Any]]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return None


def build_variant_scene_lookup(rows: list[dict[str, Any]]) -> Dict[tuple[str, str], dict[str, Any]]:
    return {(row["scene"], row["variant"]): row for row in metric_rows_only(rows)}


def write_win_rate_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    lookup = build_variant_scene_lookup(rows)
    ours_rows = [row for row in metric_rows_only(rows) if row["variant"] == "geofar_sh_ours"]
    comparisons = [BASELINE_VARIANT, "3dgs_40k_cont", "sh_only_10k", "app_only_10k"]
    output_rows = []
    for ref_variant in comparisons:
        psnr_wins = 0
        ssim_wins = 0
        lpips_wins = 0
        total = 0
        for ours in ours_rows:
            other = lookup.get((ours["scene"], ref_variant))
            if other is None:
                continue
            total += 1
            if float(ours["psnr"]) > float(other["psnr"]):
                psnr_wins += 1
            if float(ours["ssim"]) > float(other["ssim"]):
                ssim_wins += 1
            if float(ours["lpips"]) < float(other["lpips"]):
                lpips_wins += 1
        output_rows.append(
            {
                "comparison": f"vs {ref_variant}",
                "PSNR win count": psnr_wins,
                "SSIM win count": ssim_wins,
                "LPIPS win count": lpips_wins,
                "PSNR win rate": None if total == 0 else psnr_wins / total,
                "SSIM win rate": None if total == 0 else ssim_wins / total,
                "LPIPS win rate": None if total == 0 else lpips_wins / total,
            }
        )
    headers = [
        "comparison",
        "PSNR win count",
        "SSIM win count",
        "LPIPS win count",
        "PSNR win rate",
        "SSIM win rate",
        "LPIPS win rate",
    ]
    with (root / "win_rate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            elif value is None:
                rendered.append("N/A")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_paper_tables(root: Path, scene_rows: list[dict[str, Any]], overall_rows: list[dict[str, Any]]) -> None:
    overall_lookup = {row["variant"]: row for row in overall_rows}

    main_headers = [
        "Method",
        "Geometry Frozen",
        "SH Refine",
        "Appearance Residual",
        "PSNR",
        "SSIM",
        "LPIPS",
        "螖PSNR vs 3DGS-30k",
        "螖PSNR vs 3DGS-40k-cont",
    ]
    main_rows = []
    for variant, label, geo, sh, app in [
        (BASELINE_VARIANT, "3DGS-30k", "No", "No", "No"),
        ("3dgs_40k_cont", "3DGS-40k-cont", "No", "No", "No"),
        ("sh_only_10k", "SH-only-10k", "Yes", "Yes", "No"),
        ("app_only_10k", "App-only-10k", "Yes", "No", "Yes"),
        ("geofar_sh_ours", "GeoFAR-SH", "Yes", "Yes", "Yes"),
    ]:
        row = overall_lookup.get(variant, {})
        main_rows.append(
            [
                label,
                geo,
                sh,
                app,
                row.get("PSNR"),
                row.get("SSIM"),
                row.get("LPIPS"),
                row.get("螖PSNR vs 3DGS-30k"),
                row.get("螖PSNR vs 3DGS-40k-cont"),
            ]
        )
    write_text(root / "paper_table_main.md", markdown_table(main_headers, main_rows) + "\n")

    ablation_headers = ["Method", "PSNR", "SSIM", "LPIPS"]
    ablation_rows = [
        ["3DGS-30k", *(overall_lookup.get(BASELINE_VARIANT, {}).get(key) for key in ["PSNR", "SSIM", "LPIPS"])],
        ["3DGS-40k-cont", *(overall_lookup.get("3dgs_40k_cont", {}).get(key) for key in ["PSNR", "SSIM", "LPIPS"])],
        ["SH-only-10k", *(overall_lookup.get("sh_only_10k", {}).get(key) for key in ["PSNR", "SSIM", "LPIPS"])],
        ["App-only-10k", *(overall_lookup.get("app_only_10k", {}).get(key) for key in ["PSNR", "SSIM", "LPIPS"])],
        ["GeoFAR-SH", *(overall_lookup.get("geofar_sh_ours", {}).get(key) for key in ["PSNR", "SSIM", "LPIPS"])],
    ]
    write_text(root / "paper_table_ablation.md", markdown_table(ablation_headers, ablation_rows) + "\n")

    scene_headers = ["Dataset", "Scene", "3DGS-30k", "3DGS-40k-cont", "SH-only-10k", "App-only-10k", "GeoFAR-SH"]
    metric_rows = metric_rows_only(scene_rows)
    scenes = sorted({(row["dataset"], row["scene"]) for row in metric_rows})
    lookup = {(row["dataset"], row["scene"], row["variant"]): row for row in metric_rows}
    scene_table_rows = []
    for dataset, scene in scenes:
        scene_table_rows.append(
            [
                dataset,
                scene,
                lookup.get((dataset, scene, BASELINE_VARIANT), {}).get("psnr"),
                lookup.get((dataset, scene, "3dgs_40k_cont"), {}).get("psnr"),
                lookup.get((dataset, scene, "sh_only_10k"), {}).get("psnr"),
                lookup.get((dataset, scene, "app_only_10k"), {}).get("psnr"),
                lookup.get((dataset, scene, "geofar_sh_ours"), {}).get("psnr"),
            ]
        )
    write_text(root / "paper_table_scene_level.md", markdown_table(scene_headers, scene_table_rows) + "\n")


def write_fairness_analysis(root: Path, scene_rows: list[dict[str, Any]], overall_rows: list[dict[str, Any]]) -> None:
    overall_lookup = {row["variant"]: row for row in overall_rows}
    lookup = build_variant_scene_lookup(scene_rows)
    ours = overall_lookup.get("geofar_sh_ours")

    def overall_delta(ref_variant: str, metric: str) -> Optional[float]:
        ours_row = overall_lookup.get("geofar_sh_ours")
        ref_row = overall_lookup.get(ref_variant)
        if ours_row is None or ref_row is None:
            return None
        return float(ours_row[metric]) - float(ref_row[metric])

    worse_than_cont: list[str] = []
    worse_than_sh: list[str] = []
    worse_than_app: list[str] = []
    for (scene, variant), ours_row in sorted(lookup.items()):
        if variant != "geofar_sh_ours":
            continue
        cont = lookup.get((scene, "3dgs_40k_cont"))
        sh_only = lookup.get((scene, "sh_only_10k"))
        app_only = lookup.get((scene, "app_only_10k"))
        if cont and float(ours_row["psnr"]) < float(cont["psnr"]):
            worse_than_cont.append(scene)
        if sh_only and float(ours_row["psnr"]) < float(sh_only["psnr"]):
            worse_than_sh.append(scene)
        if app_only and float(ours_row["psnr"]) < float(app_only["psnr"]):
            worse_than_app.append(scene)

    delta_vs_cont = overall_delta("3dgs_40k_cont", "PSNR")
    delta_vs_sh = overall_delta("sh_only_10k", "PSNR")
    delta_vs_app = overall_delta("app_only_10k", "PSNR")
    close_to_cont = delta_vs_cont is not None and delta_vs_cont <= 0.05
    sh_explains_most = delta_vs_sh is not None and delta_vs_sh <= 0.05
    residual_has_extra = delta_vs_sh is not None and delta_vs_app is not None and delta_vs_sh > 0.05 and delta_vs_app > 0.05

    if delta_vs_cont is not None and delta_vs_cont <= 0.0:
        contribution_wording = "璁烘枃涓昏础鐚簲鏀逛负 low-intrusion, geometry-stable refinement锛屼笉鑳藉啓鎴愭樉钁楄川閲忔彁鍗囥€?
    elif sh_explains_most:
        contribution_wording = "璁烘枃搴斿己璋?geometry-frozen SH and appearance joint refinement锛岃€屼笉鏄崟鐙己璋?residual head銆?
    else:
        contribution_wording = "鍙互寮鸿皟 appearance residual 鍦?geometry-stable SH refinement 涔嬩笂浠嶆湁鐙珛璐＄尞銆?

    cautious = sorted(set(worse_than_cont + worse_than_sh + worse_than_app))
    lines = [
        "# Fairness Analysis",
        "",
        f"- Ours 鐩告瘮 3DGS-30k 鐨勫钩鍧囨彁鍗? 螖PSNR {format_signed(overall_delta(BASELINE_VARIANT, 'PSNR'))}, 螖SSIM {format_signed(overall_delta(BASELINE_VARIANT, 'SSIM'))}, 螖LPIPS {format_signed(overall_delta(BASELINE_VARIANT, 'LPIPS'))}",
        f"- Ours 鐩告瘮 3DGS-40k-cont 鐨勫钩鍧囨彁鍗? 螖PSNR {format_signed(overall_delta('3dgs_40k_cont', 'PSNR'))}, 螖SSIM {format_signed(overall_delta('3dgs_40k_cont', 'SSIM'))}, 螖LPIPS {format_signed(overall_delta('3dgs_40k_cont', 'LPIPS'))}",
        f"- Ours 鐩告瘮 SH-only-10k 鐨勫钩鍧囨彁鍗? 螖PSNR {format_signed(overall_delta('sh_only_10k', 'PSNR'))}, 螖SSIM {format_signed(overall_delta('sh_only_10k', 'SSIM'))}, 螖LPIPS {format_signed(overall_delta('sh_only_10k', 'LPIPS'))}",
        f"- Ours 鐩告瘮 App-only-10k 鐨勫钩鍧囨彁鍗? 螖PSNR {format_signed(overall_delta('app_only_10k', 'PSNR'))}, 螖SSIM {format_signed(overall_delta('app_only_10k', 'SSIM'))}, 螖LPIPS {format_signed(overall_delta('app_only_10k', 'LPIPS'))}",
        "",
        f"- 3DGS-40k-cont {'宸茬粡鎺ヨ繎鎴栬秴杩?Ours' if close_to_cont else '娌℃湁鎺ヨ繎 Ours锛岄澶栬缁?10k 涓嶆槸涓昏瑙ｉ噴'}銆?,
        f"- SH-only {'宸茬粡瑙ｉ噴浜嗗ぇ閮ㄥ垎鏀剁泭' if sh_explains_most else '涓嶈冻浠ヨВ閲婂ぇ閮ㄥ垎鏀剁泭'}銆?,
        f"- appearance residual {'浠嶆湁棰濆璐＄尞' if residual_has_extra else '棰濆璐＄尞鏈夐檺锛岄渶瑕佽皑鎱庤〃杩?}銆?,
        f"- 鍝簺鍦烘櫙 Ours 涓嶅 3DGS-40k-cont: {', '.join(worse_than_cont) if worse_than_cont else 'none'}",
        f"- 鍝簺鍦烘櫙 Ours 涓嶅 SH-only: {', '.join(worse_than_sh) if worse_than_sh else 'none'}",
        f"- 鍝簺鍦烘櫙 Ours 涓嶅 App-only: {', '.join(worse_than_app) if worse_than_app else 'none'}",
        f"- 鍝簺鍦烘櫙搴旇鍦ㄨ鏂囦腑璋ㄦ厧璁ㄨ: {', '.join(cautious) if cautious else 'none'}",
        "",
        f"- 璁烘枃涓昏础鐚〃杩板缓璁? {contribution_wording}",
    ]
    write_text(root / "fairness_analysis.md", "\n".join(lines) + "\n")


def write_paper_conclusion_suggestion(root: Path, overall_rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]]) -> None:
    overall_lookup = {row["variant"]: row for row in overall_rows}
    ours = overall_lookup.get("geofar_sh_ours")
    baseline = overall_lookup.get(BASELINE_VARIANT)
    cont = overall_lookup.get("3dgs_40k_cont")
    sh_only = overall_lookup.get("sh_only_10k")

    def delta_psnr(a: Optional[dict[str, Any]], b: Optional[dict[str, Any]]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a["PSNR"]) - float(b["PSNR"])

    d_base = delta_psnr(ours, baseline)
    d_cont = delta_psnr(ours, cont)
    d_sh = delta_psnr(ours, sh_only)

    stable_improvement = d_base is not None and d_base > 0.1 and d_cont is not None and d_cont > 0.05
    conservative = d_base is not None and d_base > 0.0 and (d_cont is None or d_cont <= 0.05)
    weaken_cuda = d_sh is not None and d_sh <= 0.05

    lines = [
        "# Paper Conclusion Suggestion",
        "",
        "## Chapter 4",
        "",
        f"- {'鍙互鍐?stable improvement over 3DGS' if stable_improvement else '涓嶅缓璁啓 stable improvement over 3DGS'}銆?,
        f"- {'寤鸿鏀规垚 conservative but consistent improvement' if conservative else '涓嶅繀寮鸿皟 conservative but consistent improvement'}銆?,
        f"- {'闇€瑕佸急鍖?CUDA-fused 鎴?residual contribution' if weaken_cuda else '鍙互淇濈暀 residual contribution锛屼絾瑕佷互鍏钩瀵圭収涓哄墠鎻?}銆?,
        "- 搴旂獊鍑?geometry delta = 0 鐨勫綊鍥犱紭鍔匡紝灏ゅ叾鏄湪 SH-only銆丄pp-only銆丱urs 涓夎€呭姣斾腑銆?,
        "",
        "## Conclusion",
        "",
        "- 濡傛灉 3DGS-40k-cont 鎺ヨ繎 Ours锛岀粨璁哄簲寮鸿皟 low-intrusion, geometry-stable refinement銆?,
        "- 濡傛灉 SH-only 涓?Ours 宸窛寰堝皬锛岀粨璁哄簲寮鸿皟 joint refinement 鑰屼笉鏄?residual head 鍗曠嫭璐＄尞銆?,
        "- 濡傛灉 Ours 浠嶆槑鏄句紭浜?App-only锛屽垯鍙互淇濈暀 appearance branch 鐨勫繀瑕佹€ц杩般€?,
    ]
    write_text(root / "paper_conclusion_suggestion.md", "\n".join(lines) + "\n")


def print_final_summary(scene_rows: list[dict[str, Any]], overall_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    overall_lookup = {row["variant"]: row for row in overall_rows}

    def improvement(ref_variant: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        ours = overall_lookup.get("geofar_sh_ours")
        ref = overall_lookup.get(ref_variant)
        if ours is None or ref is None:
            return (None, None, None)
        return (
            float(ours["PSNR"]) - float(ref["PSNR"]),
            float(ours["SSIM"]) - float(ref["SSIM"]),
            float(ours["LPIPS"]) - float(ref["LPIPS"]),
        )

    valid_scenes = len({row["scene"] for row in metric_rows_only(scene_rows) if row["variant"] == BASELINE_VARIANT})
    completed_variants = len([record for record in records if record["variant"] != BASELINE_VARIANT and record["status"] in {"done", "skipped_existing"}])
    failed_jobs = len([record for record in records if record["status"] in {"failed", "blocked"}])

    for ref_variant in [BASELINE_VARIANT, "3dgs_40k_cont", "sh_only_10k", "app_only_10k"]:
        d_psnr, d_ssim, d_lpips = improvement(ref_variant)
        label = {
            BASELINE_VARIANT: "3DGS-30k",
            "3dgs_40k_cont": "3DGS-40k-cont",
            "sh_only_10k": "SH-only-10k",
            "app_only_10k": "App-only-10k",
        }[ref_variant]
        print(f"Ours vs {label}: 螖PSNR={format_signed(d_psnr)} 螖SSIM={format_signed(d_ssim)} 螖LPIPS={format_signed(d_lpips)}")

    print(f"Valid scenes: {valid_scenes}")
    print(f"Completed variant jobs: {completed_variants}")
    print(f"Failed jobs: {failed_jobs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metas, issues = discover_scene_meta(args.scenes)
    write_missing_data_paths(OUTPUT_ROOT, metas)
    if issues:
        write_json(OUTPUT_ROOT / "scene_discovery_issues.json", issues)

    if args.dry_run:
        dry_run_plan(OUTPUT_ROOT, metas, args.variants, args.skip_existing)
        update_running_status(OUTPUT_ROOT, [], {"scene": "-", "variant": "-", "status": "dry-run", "stage": "planning", "reason": "dry-run completed"})
        return

    records: list[dict[str, Any]] = []
    for meta in metas:
        baseline_record = materialize_baseline(meta)
        records.append(baseline_record)
        update_running_status(OUTPUT_ROOT, records, baseline_record)
        for variant in args.variants:
            result = run_variant(meta, variant, clean=args.clean, skip_existing=args.skip_existing)
            records.append(result)
            update_running_status(OUTPUT_ROOT, records, result)

    write_failed_jobs(OUTPUT_ROOT, records)
    write_completed_jobs(OUTPUT_ROOT, records)

    scene_rows = summarize_scene_level(OUTPUT_ROOT, metas, args.variants, records)
    _, overall_rows = summarize_averages(OUTPUT_ROOT, scene_rows)
    write_win_rate_summary(OUTPUT_ROOT, scene_rows)
    write_fairness_analysis(OUTPUT_ROOT, scene_rows, overall_rows)
    write_paper_tables(OUTPUT_ROOT, scene_rows, overall_rows)
    write_paper_conclusion_suggestion(OUTPUT_ROOT, overall_rows, scene_rows)
    update_running_status(OUTPUT_ROOT, records, {"scene": "-", "variant": "-", "status": "finished", "stage": "summary", "reason": "all summaries generated"})
    print_final_summary(scene_rows, overall_rows, records)


if __name__ == "__main__":
    main()






