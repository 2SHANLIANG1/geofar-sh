# Release wrapper copied from original source: scripts/collect_efficiency_evidence.py
# Private local paths were sanitized for the GitHub release package.
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

REPO_FALLBACK = Path(__file__).resolve().parents[1]
if str(REPO_FALLBACK) not in sys.path:
    sys.path.insert(0, str(REPO_FALLBACK))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


FORMAL_METHODS = [
    "3dgs_30k_baseline",
    "3dgs_40k_cont",
    "sh_only_10k",
    "app_only_10k",
    "geofar_sh_ours",
]
OPTIONAL_METHODS = ["geofar_sh_ours_torch_precompute"]
ALL_METHODS = FORMAL_METHODS + OPTIONAL_METHODS
SCENES = [
    ("Mip-NeRF360", "bicycle"),
    ("Mip-NeRF360", "bonsai"),
    ("Mip-NeRF360", "counter"),
    ("Mip-NeRF360", "flowers"),
    ("Mip-NeRF360", "garden"),
    ("Mip-NeRF360", "kitchen"),
    ("Mip-NeRF360", "room"),
    ("Mip-NeRF360", "stump"),
    ("Mip-NeRF360", "treehill"),
    ("Tanks&Temples", "train"),
    ("Tanks&Temples", "truck"),
    ("DeepBlending", "drjohnson"),
    ("DeepBlending", "playroom"),
]
METHOD_LABELS = {
    "3dgs_30k_baseline": "3DGS-30k",
    "3dgs_40k_cont": "3DGS-40k-cont",
    "sh_only_10k": "SH-only-10k",
    "app_only_10k": "App-only-10k",
    "geofar_sh_ours": "GeoFAR-SH",
    "geofar_sh_ours_torch_precompute": "GeoFAR-SH torch-precompute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CUDA-fused efficiency evidence for GeoFAR-SH.")
    parser.add_argument("--root", type=Path, default=Path(r"<PROJECT_ROOT>"))
    parser.add_argument("--fairness_dir", type=Path, default=Path(r"<PROJECT_ROOT>\output\paper_fairness_controls"))
    parser.add_argument("--baseline_ckpt_dir", type=Path, default=Path(r"<OUTPUT_ROOT>\output_ckpt30k"))
    parser.add_argument("--output_dir", type=Path, default=Path(r"<PROJECT_ROOT>\output\paper_efficiency_evidence"))
    parser.add_argument("--baseline_output_dir", type=Path, default=Path(r"<OUTPUT_ROOT>\output"))
    parser.add_argument("--benchmark_repeats", type=int, default=1)
    parser.add_argument("--benchmark_max_views", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        if path.exists():
            path.unlink()

    def log(self, message: str) -> None:
        print(message)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def load_namespace_cfg(path: Path) -> Namespace:
    parser = argparse.ArgumentParser()
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)
    defaults = parser.parse_args([])
    cfg_ns = eval(path.read_text(encoding="utf-8").strip(), {"Namespace": Namespace})
    merged = vars(defaults).copy()
    merged.update(vars(cfg_ns))
    return Namespace(**merged)


def torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def parse_checkpoint_tensors(path: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    payload = torch_load(path)
    iteration = None
    model_args = payload
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], tuple):
        model_args = payload[0]
        iteration = int(payload[1]) if isinstance(payload[1], int) else None
    if not isinstance(model_args, tuple):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    tensors: dict[str, torch.Tensor] = {}
    slots = {
        "_xyz": 1,
        "_features_dc": 2,
        "_features_rest": 3,
        "_scaling": 4,
        "_rotation": 5,
        "_opacity": 6,
    }
    for name, idx in slots.items():
        if len(model_args) > idx and torch.is_tensor(model_args[idx]):
            tensors[name] = model_args[idx].detach().cpu()
    if len(model_args) >= 32:
        extra = {
            "_appearance_latent": 13,
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
        for name, idx in extra.items():
            if len(model_args) > idx and torch.is_tensor(model_args[idx]):
                tensors[name] = model_args[idx].detach().cpu()
    return tensors, iteration


def count_params(tensors: dict[str, torch.Tensor], method: str) -> tuple[int, int, int]:
    total = 0
    appearance = 0
    trainable_stage2 = 0
    for name, tensor in tensors.items():
        count = int(tensor.numel())
        total += count
        lowered = name.lower()
        is_appearance = any(tok in lowered for tok in ("appearance", "latent", "app_", "diff", "spec", "mask", "gate", "residual", "head", "mlp", "fastkan"))
        if is_appearance:
            appearance += count
        if method == "3dgs_30k_baseline":
            trainable = False
        elif method == "3dgs_40k_cont":
            trainable = name in {"_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"}
        elif method == "sh_only_10k":
            trainable = name in {"_features_dc", "_features_rest"}
        elif method == "app_only_10k":
            trainable = is_appearance
        elif method in {"geofar_sh_ours", "geofar_sh_ours_torch_precompute"}:
            trainable = name in {"_features_dc", "_features_rest"} or is_appearance
        else:
            trainable = False
        if trainable:
            trainable_stage2 += count
    return total, appearance, trainable_stage2


def file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024.0 * 1024.0)


def dir_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024.0 * 1024.0)


def recursive_first(root: Path, patterns: list[str]) -> Path | None:
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def find_baseline_run(scene: str, baseline_output_dir: Path, baseline_ckpt_dir: Path) -> dict[str, Path | None]:
    run_dir = baseline_output_dir / scene
    point_dir = run_dir / "point_cloud" / "iteration_30000"
    return {
        "run_dir": run_dir if run_dir.exists() else None,
        "checkpoint": (baseline_ckpt_dir / scene / "chkpnt30000.pth") if (baseline_ckpt_dir / scene / "chkpnt30000.pth").exists() else recursive_first(run_dir, ["*30000*.pth"]),
        "point_cloud_dir": point_dir if point_dir.exists() else recursive_first(run_dir, ["point_cloud/iteration_30000"]),
        "results": run_dir / "results.json" if (run_dir / "results.json").exists() else recursive_first(run_dir, ["results.json"]),
        "cfg_args": run_dir / "cfg_args" if (run_dir / "cfg_args").exists() else None,
        "train_log": run_dir / "train.log" if (run_dir / "train.log").exists() else None,
        "iteration": Path("30000"),
    }


def find_variant_run(scene: str, method: str, fairness_dir: Path) -> dict[str, Path | None]:
    run_method = "geofar_sh_ours" if method == "geofar_sh_ours_torch_precompute" else method
    run_dir = fairness_dir / scene / run_method
    point_dir = run_dir / "point_cloud" / "iteration_40000"
    return {
        "run_dir": run_dir if run_dir.exists() else None,
        "checkpoint": (run_dir / "chkpnt40000.pth") if (run_dir / "chkpnt40000.pth").exists() else recursive_first(run_dir, ["*40000*.pth", "*final_checkpoint*.pth"]),
        "point_cloud_dir": point_dir if point_dir.exists() else recursive_first(run_dir, ["point_cloud/iteration_40000"]),
        "results": run_dir / "results.json" if (run_dir / "results.json").exists() else recursive_first(run_dir, ["results.json"]),
        "cfg_args": run_dir / "cfg_args" if (run_dir / "cfg_args").exists() else None,
        "train_log": run_dir / "train.log" if (run_dir / "train.log").exists() else recursive_first(run_dir, ["*train*.log"]),
        "time_memory": run_dir / "time_memory.json" if (run_dir / "time_memory.json").exists() else None,
        "iteration": Path("40000"),
    }


def parse_train_time(info: dict[str, Path | None], method: str, baseline_ckpt: Path | None) -> tuple[float | None, str]:
    if method == "3dgs_30k_baseline":
        return None, "unavailable"
    tm_path = info.get("time_memory")
    if tm_path is not None and Path(tm_path).exists():
        data = json.loads(Path(tm_path).read_text(encoding="utf-8"))
        stage2 = safe_float(data.get("stage2_time_sec"))
        total = safe_float(data.get("total_train_sec"))
        if stage2 is not None and stage2 > 0:
            return stage2, "parsed_log"
        if total is not None and total > 0:
            return total, "parsed_log"
    ckpt = info.get("checkpoint")
    if ckpt is not None and baseline_ckpt is not None and Path(ckpt).exists() and baseline_ckpt.exists():
        sec = Path(ckpt).stat().st_mtime - baseline_ckpt.stat().st_mtime
        if sec > 0:
            return sec, "timestamp_diff"
    return None, "unavailable"


def variant_mode(method: str) -> str:
    if method == "geofar_sh_ours_torch_precompute":
        return "torch_precompute"
    if method in {"geofar_sh_ours", "app_only_10k"}:
        return "cuda_fused"
    return "original_3dgs"


def build_args(cfg_path: Path, run_dir: Path, method: str) -> Namespace:
    args = load_namespace_cfg(cfg_path)
    args.model_path = str(run_dir)
    args.eval = True
    if method == "geofar_sh_ours_torch_precompute":
        args.appearance_compute_mode = "torch_precompute"
        args.convert_SHs_python = True
    return args


def load_test_views(dataset_args: Namespace) -> list[Any]:
    source_path = dataset_args.source_path
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            source_path,
            dataset_args.images,
            dataset_args.depths,
            dataset_args.eval,
            dataset_args.train_test_exp,
        )
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](
            source_path,
            dataset_args.white_background,
            dataset_args.depths,
            dataset_args.eval,
        )
    else:
        raise RuntimeError(f"Could not recognize scene type for {source_path}")
    return cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset_args, scene_info.is_nerf_synthetic, True)


def load_gaussians_for_benchmark(args: Namespace, point_cloud_dir: Path) -> tuple[GaussianModel, str]:
    gaussians = GaussianModel(args.sh_degree)
    gaussians.configure_appearance_residual(args)
    gaussians.load_ply(str(point_cloud_dir / "point_cloud.ply"), args.train_test_exp)
    load_status = "baseline_mode"
    appearance_path = point_cloud_dir / "appearance_residual.pth"
    if gaussians.appearance_residual_enabled:
        if appearance_path.exists():
            state = torch.load(appearance_path, map_location="cuda")
            for name, source in state.items():
                if source is None or not hasattr(gaussians, name):
                    continue
                target = getattr(gaussians, name)
                if target is None or tuple(target.shape) != tuple(source.shape):
                    continue
                target.data.copy_(source)
            gaussians.mark_appearance_residual_loaded(True)
            load_status = "appearance_residual_loaded"
        else:
            gaussians.mark_appearance_residual_loaded(False)
            load_status = "appearance_residual_requested_but_missing_weights"
    return gaussians, load_status


def benchmark_render(
    cfg_path: Path,
    run_dir: Path,
    point_cloud_dir: Path,
    method: str,
    repeats: int,
    max_views: int,
    logger: Logger,
    cached_views: list[Any],
) -> dict[str, Any]:
    args = build_args(cfg_path, run_dir, method)
    parser = argparse.ArgumentParser()
    model_group = ModelParams(parser, sentinel=True)
    pipe_group = PipelineParams(parser)
    dataset = model_group.extract(args)
    pipeline = pipe_group.extract(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_times: list[float] = []
    per_view_times: list[float] = []
    peak_allocs: list[float] = []
    peak_reserved: list[float] = []
    num_views = 0
    resolution = ""
    loaded_status = "unknown"

    for repeat_idx in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gaussians, loaded_status = load_gaussians_for_benchmark(args, point_cloud_dir)
        views = cached_views
        if not views:
            raise RuntimeError(f"No test cameras found for {run_dir}")
        num_views = min(len(views), max_views)
        bench_views = views[:num_views]
        warmup_views = bench_views[: min(5, len(bench_views))]
        if bench_views:
            resolution = f"{int(bench_views[0].image_width)}x{int(bench_views[0].image_height)}"
        with torch.no_grad():
            for view in warmup_views:
                _ = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp)["render"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for view in bench_views:
                _ = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp)["render"]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        render_times.append(elapsed)
        per_view_times.append(elapsed / float(num_views))
        peak_allocs.append(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        peak_reserved.append(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
        logger.log(
            f"[BENCH] {method} {run_dir.name} repeat={repeat_idx + 1}/{repeats} mode={variant_mode(method)} "
            f"views={num_views} time={elapsed:.4f}s fps={num_views / elapsed:.4f} peak_alloc={peak_allocs[-1]:.2f}MB peak_reserved={peak_reserved[-1]:.2f}MB load_status={loaded_status}"
        )
        del gaussians
        torch.cuda.empty_cache()
        gc.collect()

    mean_total = sum(render_times) / len(render_times)
    fps_values = [num_views / value for value in render_times]
    return {
        "num_benchmark_views": num_views,
        "render_time_total_sec": mean_total,
        "render_time_per_view_sec": sum(per_view_times) / len(per_view_times),
        "render_fps": sum(fps_values) / len(fps_values),
        "render_fps_std": statistics.pstdev(fps_values) if len(fps_values) > 1 else 0.0,
        "benchmark_resolution": resolution,
        "benchmark_mode": variant_mode(method),
        "peak_gpu_memory_mb": sum(peak_reserved) / len(peak_reserved),
        "peak_torch_allocated_mb": sum(peak_allocs) / len(peak_allocs),
        "peak_torch_reserved_mb": sum(peak_reserved) / len(peak_reserved),
        "memory_measure_source": "direct_benchmark",
        "appearance_load_status": loaded_status,
    }


def summary_stats(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [row for row in rows if row["method"] == method and row["status"] == "success"]
    values = lambda key: [safe_float(row[key]) for row in subset if safe_float(row[key]) is not None]
    fps_vals = values("render_fps")
    mem_vals = values("peak_gpu_memory_mb")
    size_vals = values("total_model_size_mb")
    gauss_vals = values("num_gaussians")
    param_vals = values("total_parameter_count")
    over_vals = values("parameter_overhead_vs_3dgs30k")
    time_vals = values("stage2_train_time_hour")
    return {
        "method": method,
        "scene_count": len(subset),
        "mean_stage2_train_time_hour": "" if not time_vals else f"{sum(time_vals) / len(time_vals):.12g}",
        "mean_render_fps": "" if not fps_vals else f"{sum(fps_vals) / len(fps_vals):.12g}",
        "std_render_fps": "" if not fps_vals else f"{statistics.pstdev(fps_vals):.12g}",
        "mean_peak_gpu_memory_mb": "" if not mem_vals else f"{sum(mem_vals) / len(mem_vals):.12g}",
        "mean_total_model_size_mb": "" if not size_vals else f"{sum(size_vals) / len(size_vals):.12g}",
        "mean_num_gaussians": "" if not gauss_vals else f"{sum(gauss_vals) / len(gauss_vals):.12g}",
        "mean_parameter_count": "" if not param_vals else f"{sum(param_vals) / len(param_vals):.12g}",
        "mean_parameter_overhead_vs_3dgs30k": "" if not over_vals else f"{sum(over_vals) / len(over_vals):.12g}",
        "status": "success" if subset else "failed",
    }


def make_table(summary_map: dict[str, dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Efficiency comparison of different Stage-2 refinement variants. Training time is measured for the second-stage optimization from 30k to 40k iterations, and rendering speed is averaged over the test views of the 13 valid scenes.}",
        r"\label{tab:efficiency_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & Stage-2 time (h) & Render FPS & GPU memory (MB) & Model size (MB) \\",
        r"\hline",
    ]
    for method in FORMAL_METHODS:
        row = summary_map[method]
        stage2 = "--" if method == "3dgs_30k_baseline" else fmt(safe_float(row["mean_stage2_train_time_hour"]), 2)
        lines.append(
            f"{METHOD_LABELS[method]} & {stage2} & {fmt(safe_float(row['mean_render_fps']), 2)} & {fmt(safe_float(row['mean_peak_gpu_memory_mb']), 1)} & {fmt(safe_float(row['mean_total_model_size_mb']), 2)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def make_text(summary_map: dict[str, dict[str, Any]]) -> str:
    geo = summary_map["geofar_sh_ours"]
    base = summary_map["3dgs_30k_baseline"]
    sh = summary_map["sh_only_10k"]
    geo_fps = safe_float(geo["mean_render_fps"]) or 0.0
    geo_mem = safe_float(geo["mean_peak_gpu_memory_mb"]) or 0.0
    geo_size = safe_float(geo["mean_total_model_size_mb"]) or 0.0
    base_fps = safe_float(base["mean_render_fps"]) or 0.0
    base_mem = safe_float(base["mean_peak_gpu_memory_mb"]) or 0.0
    base_size = safe_float(base["mean_total_model_size_mb"]) or 0.0
    sh_mem = safe_float(sh["mean_peak_gpu_memory_mb"]) or 0.0
    sh_size = safe_float(sh["mean_total_model_size_mb"]) or 0.0
    fps_delta = geo_fps - base_fps
    mem_delta = geo_mem - base_mem
    size_delta = geo_size - base_size
    sh_mem_delta = geo_mem - sh_mem
    sh_size_delta = geo_size - sh_size
    rel = abs(fps_delta) / max(base_fps, 1e-6)
    if fps_delta >= 0 and rel < 0.05:
        last_sentence = "These results show that the proposed refinement maintains comparable rendering efficiency while preserving the low-intrusion rendering structure of 3DGS."
    elif fps_delta < 0 and rel < 0.15:
        last_sentence = "These results show that the proposed refinement introduces a limited computational overhead while preserving the low-intrusion rendering structure of 3DGS."
    else:
        last_sentence = "These results indicate that GeoFAR-SH trades a moderate computational overhead for geometry-stable post-convergence appearance refinement."
    text = f"""
\\subsubsection{{Efficiency Analysis}}

Since GeoFAR-SH integrates the residual-enhanced color computation into the CUDA rasterizer, we further evaluate its computational efficiency. All methods are benchmarked using the same test views and rendering resolution, with CUDA synchronization and warm-up iterations before timing. Table~\\ref{{tab:efficiency_comparison}} reports the average Stage-2 training time, rendering speed, peak GPU memory usage, and model size over the 13 valid scenes.

GeoFAR-SH achieves an average rendering speed of {geo_fps:.2f} FPS with {geo_mem:.1f} MB peak GPU memory usage. Compared with the 3DGS-30k baseline, GeoFAR-SH changes the rendering speed by {fps_delta:+.2f} FPS and increases the model size by {size_delta:+.2f} MB, with a memory change of {mem_delta:+.1f} MB. Compared with SH-only-10k, the additional appearance residual branch introduces {sh_size_delta:+.2f} MB model-size overhead and {sh_mem_delta:+.1f} MB memory overhead. {last_sentence}
""".strip()
    return text + "\n"


def markdown_summary(summary_map: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Efficiency Summary",
        "",
        "## Method Averages",
    ]
    for method in FORMAL_METHODS + [m for m in OPTIONAL_METHODS if m in summary_map]:
        row = summary_map[method]
        lines.append(
            f"- {METHOD_LABELS[method]}: render FPS {fmt(safe_float(row['mean_render_fps']), 2)}, GPU memory {fmt(safe_float(row['mean_peak_gpu_memory_mb']), 1)} MB, model size {fmt(safe_float(row['mean_total_model_size_mb']), 2)} MB"
        )
    geo = summary_map["geofar_sh_ours"]
    base = summary_map["3dgs_30k_baseline"]
    sh = summary_map["sh_only_10k"]
    lines.extend(
        [
            "",
            "## GeoFAR-SH Overheads",
            f"- Versus 3DGS-30k: FPS {((safe_float(geo['mean_render_fps']) or 0.0) - (safe_float(base['mean_render_fps']) or 0.0)):+.2f}, memory {((safe_float(geo['mean_peak_gpu_memory_mb']) or 0.0) - (safe_float(base['mean_peak_gpu_memory_mb']) or 0.0)):+.1f} MB, model size {((safe_float(geo['mean_total_model_size_mb']) or 0.0) - (safe_float(base['mean_total_model_size_mb']) or 0.0)):+.2f} MB",
            f"- Versus SH-only-10k: additional appearance overhead is {((safe_float(geo['mean_total_model_size_mb']) or 0.0) - (safe_float(sh['mean_total_model_size_mb']) or 0.0)):+.2f} MB in model size and {((safe_float(geo['mean_peak_gpu_memory_mb']) or 0.0) - (safe_float(sh['mean_peak_gpu_memory_mb']) or 0.0)):+.1f} MB in memory.",
        ]
    )
    if "geofar_sh_ours_torch_precompute" in summary_map:
        row = summary_map["geofar_sh_ours_torch_precompute"]
        lines.extend(
            [
                "",
                "## Optional GeoFAR-SH Torch-Precompute",
                f"- GeoFAR-SH torch-precompute: render FPS {fmt(safe_float(row['mean_render_fps']), 2)}, GPU memory {fmt(safe_float(row['mean_peak_gpu_memory_mb']), 1)} MB, model size {fmt(safe_float(row['mean_total_model_size_mb']), 2)} MB",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_dir(args.output_dir)
    logger = Logger(args.output_dir / "collect_efficiency_evidence.log")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    scene_baselines: dict[str, dict[str, Any]] = {}

    baseline_infos = {scene: find_baseline_run(scene, args.baseline_output_dir, args.baseline_ckpt_dir) for _, scene in SCENES}
    cached_scene_views: dict[str, list[Any]] = {}

    for dataset, scene in SCENES:
        base_cfg = baseline_infos[scene].get("cfg_args")
        cfg_source = base_cfg if isinstance(base_cfg, Path) and base_cfg.exists() else None
        if cfg_source is None:
            for probe in FORMAL_METHODS[1:]:
                probe_info = find_variant_run(scene, probe, args.fairness_dir)
                probe_cfg = probe_info.get("cfg_args")
                if isinstance(probe_cfg, Path) and probe_cfg.exists():
                    cfg_source = probe_cfg
                    break
        if cfg_source is None:
            raise RuntimeError(f"No cfg_args found to load shared test cameras for scene {scene}")
        shared_args = build_args(cfg_source, cfg_source.parent, "3dgs_30k_baseline")
        parser = argparse.ArgumentParser()
        model_group = ModelParams(parser, sentinel=True)
        shared_dataset = model_group.extract(shared_args)
        cached_scene_views[scene] = load_test_views(shared_dataset)
        logger.log(f"[CACHE] scene={scene} loaded {len(cached_scene_views[scene])} shared test views")
        for method in ALL_METHODS:
            logger.log(f"[START] dataset={dataset} scene={scene} method={method}")
            if method == "3dgs_30k_baseline":
                info = baseline_infos[scene]
                run_dir = info["run_dir"]
                ckpt = info["checkpoint"]
                point_dir = info["point_cloud_dir"]
                results_path = info["results"]
                cfg_args = info["cfg_args"]
                train_log = info["train_log"]
                iteration = 30000
            else:
                info = find_variant_run(scene, method, args.fairness_dir)
                run_dir = info["run_dir"]
                ckpt = info["checkpoint"]
                point_dir = info["point_cloud_dir"]
                results_path = info["results"]
                cfg_args = info["cfg_args"]
                train_log = info["train_log"]
                iteration = 40000

            if run_dir is None or ckpt is None or point_dir is None or cfg_args is None:
                reason = f"missing path(s): run_dir={run_dir}, checkpoint={ckpt}, point_cloud_dir={point_dir}, cfg_args={cfg_args}"
                failed.append({"scene": scene, "method": method, "reason": reason})
                logger.log(f"[FAIL] {scene}/{method}: {reason}")
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "method": method,
                        "checkpoint_path": str(ckpt) if ckpt else "",
                        "point_cloud_path": str(point_dir) if point_dir else "",
                        "result_path": str(results_path) if results_path else "",
                        "log_path": str(train_log) if train_log else "",
                        "status": "failed",
                    }
                )
                continue

            try:
                tensors, _ = parse_checkpoint_tensors(ckpt)
                total_param_count, appearance_param_count, trainable_stage2 = count_params(tensors, method)
                num_gaussians = int(tensors["_xyz"].shape[0]) if "_xyz" in tensors else None
                ckpt_size = file_size_mb(ckpt)
                point_size = dir_size_mb(point_dir)
                total_model_size = point_size
                baseline_ckpt = baseline_infos[scene]["checkpoint"]
                stage2_time_sec, train_time_source = parse_train_time(info, method, baseline_ckpt if isinstance(baseline_ckpt, Path) else None)
                bench = benchmark_render(cfg_args, run_dir, point_dir, method, args.benchmark_repeats, args.benchmark_max_views, logger, cached_scene_views[scene])
                row = {
                    "dataset": dataset,
                    "scene": scene,
                    "method": method,
                    "checkpoint_path": str(ckpt),
                    "point_cloud_path": str(point_dir),
                    "result_path": str(results_path) if results_path else "",
                    "log_path": str(train_log) if train_log else "",
                    "status": "success",
                    "stage2_train_time_sec": "" if stage2_time_sec is None else f"{stage2_time_sec:.12g}",
                    "stage2_train_time_min": "" if stage2_time_sec is None else f"{stage2_time_sec / 60.0:.12g}",
                    "stage2_train_time_hour": "" if stage2_time_sec is None else f"{stage2_time_sec / 3600.0:.12g}",
                    "train_time_source": train_time_source,
                    "num_benchmark_views": bench["num_benchmark_views"],
                    "render_time_total_sec": f"{bench['render_time_total_sec']:.12g}",
                    "render_time_per_view_sec": f"{bench['render_time_per_view_sec']:.12g}",
                    "render_fps": f"{bench['render_fps']:.12g}",
                    "render_fps_std": f"{bench['render_fps_std']:.12g}",
                    "benchmark_resolution": bench["benchmark_resolution"],
                    "benchmark_mode": bench["benchmark_mode"],
                    "peak_gpu_memory_mb": f"{bench['peak_gpu_memory_mb']:.12g}",
                    "peak_torch_allocated_mb": f"{bench['peak_torch_allocated_mb']:.12g}",
                    "peak_torch_reserved_mb": f"{bench['peak_torch_reserved_mb']:.12g}",
                    "memory_measure_source": bench["memory_measure_source"],
                    "checkpoint_size_mb": "" if ckpt_size is None else f"{ckpt_size:.12g}",
                    "point_cloud_size_mb": "" if point_size is None else f"{point_size:.12g}",
                    "total_model_size_mb": "" if total_model_size is None else f"{total_model_size:.12g}",
                    "num_gaussians": "" if num_gaussians is None else str(num_gaussians),
                    "total_parameter_count": str(total_param_count),
                    "appearance_parameter_count": str(appearance_param_count),
                    "trainable_parameter_count_stage2": str(trainable_stage2),
                    "parameter_overhead_vs_3dgs30k": "",
                    "model_size_overhead_vs_3dgs30k_mb": "",
                }
                rows.append(row)
                if method == "3dgs_30k_baseline":
                    scene_baselines[scene] = row
                logger.log(
                    f"[DONE] {scene}/{method}: fps={bench['render_fps']:.2f} mem={bench['peak_gpu_memory_mb']:.1f}MB model={total_model_size:.2f}MB "
                    f"params={total_param_count} train_time_source={train_time_source}"
                )
            except Exception as exc:
                failed.append({"scene": scene, "method": method, "reason": str(exc)})
                logger.log(f"[FAIL] {scene}/{method}: {exc}")
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene,
                        "method": method,
                        "checkpoint_path": str(ckpt),
                        "point_cloud_path": str(point_dir),
                        "result_path": str(results_path) if results_path else "",
                        "log_path": str(train_log) if train_log else "",
                        "status": "failed",
                    }
                )

    for row in rows:
        base = scene_baselines.get(row["scene"])
        if row["status"] != "success" or base is None or base["status"] != "success":
            continue
        total_params = safe_float(row.get("total_parameter_count"))
        base_params = safe_float(base.get("total_parameter_count"))
        total_size = safe_float(row.get("total_model_size_mb"))
        base_size = safe_float(base.get("total_model_size_mb"))
        if total_params is not None and base_params is not None:
            row["parameter_overhead_vs_3dgs30k"] = f"{(total_params - base_params):.12g}"
        if total_size is not None and base_size is not None:
            row["model_size_overhead_vs_3dgs30k_mb"] = f"{(total_size - base_size):.12g}"

    headers = [
        "dataset", "scene", "method", "checkpoint_path", "point_cloud_path", "result_path", "log_path", "status",
        "stage2_train_time_sec", "stage2_train_time_min", "stage2_train_time_hour", "train_time_source",
        "num_benchmark_views", "render_time_total_sec", "render_time_per_view_sec", "render_fps", "render_fps_std",
        "benchmark_resolution", "benchmark_mode", "peak_gpu_memory_mb", "peak_torch_allocated_mb", "peak_torch_reserved_mb",
        "memory_measure_source", "checkpoint_size_mb", "point_cloud_size_mb", "total_model_size_mb", "num_gaussians",
        "total_parameter_count", "appearance_parameter_count", "trainable_parameter_count_stage2",
        "parameter_overhead_vs_3dgs30k", "model_size_overhead_vs_3dgs30k_mb",
    ]
    write_csv(args.output_dir / "efficiency_per_scene.csv", rows, headers)

    summary_rows = [summary_stats(rows, method) for method in FORMAL_METHODS]
    extra_summaries = [summary_stats(rows, method) for method in OPTIONAL_METHODS if any(row["method"] == method and row["status"] == "success" for row in rows)]
    all_summary_rows = summary_rows + extra_summaries
    summary_map = {row["method"]: row for row in all_summary_rows}
    write_csv(
        args.output_dir / "efficiency_summary.csv",
        all_summary_rows,
        ["method", "scene_count", "mean_stage2_train_time_hour", "mean_render_fps", "std_render_fps", "mean_peak_gpu_memory_mb", "mean_total_model_size_mb", "mean_num_gaussians", "mean_parameter_count", "mean_parameter_overhead_vs_3dgs30k", "status"],
    )
    write_text(args.output_dir / "efficiency_summary.md", markdown_summary(summary_map))
    write_text(args.output_dir / "efficiency_latex_table.tex", make_table(summary_map))
    write_text(args.output_dir / "efficiency_latex_text.tex", make_text(summary_map))
    if failed:
        lines = [
            "# Failed Jobs",
            "",
            "| scene | method | reason |",
            "| --- | --- | --- |",
        ]
        for item in failed:
            lines.append(f"| {item['scene']} | {item['method']} | {item['reason']} |")
        write_text(args.output_dir / "failed_jobs.md", "\n".join(lines) + "\n")
    else:
        write_text(args.output_dir / "failed_jobs.md", "No failed jobs.\n")

    logger.log(f"[SUMMARY] output_dir={args.output_dir}")
    logger.log(f"[SUMMARY] failed_jobs={len(failed)}")
    if "geofar_sh_ours" in summary_map:
        geo = summary_map["geofar_sh_ours"]
        base = summary_map["3dgs_30k_baseline"]
        sh = summary_map["sh_only_10k"]
        geo_fps = safe_float(geo["mean_render_fps"]) or 0.0
        geo_mem = safe_float(geo["mean_peak_gpu_memory_mb"]) or 0.0
        geo_size = safe_float(geo["mean_total_model_size_mb"]) or 0.0
        base_fps = safe_float(base["mean_render_fps"]) or 0.0
        base_mem = safe_float(base["mean_peak_gpu_memory_mb"]) or 0.0
        base_size = safe_float(base["mean_total_model_size_mb"]) or 0.0
        logger.log(f"[SUMMARY] GeoFAR-SH average Render FPS = {geo_fps:.2f}")
        logger.log(f"[SUMMARY] GeoFAR-SH average GPU memory = {geo_mem:.1f} MB")
        logger.log(f"[SUMMARY] GeoFAR-SH average model size = {geo_size:.2f} MB")
        logger.log(f"[SUMMARY] GeoFAR-SH vs 3DGS-30k: FPS {geo_fps - base_fps:+.2f}, memory {geo_mem - base_mem:+.1f} MB, size {geo_size - base_size:+.2f} MB")
        logger.log(f"[SUMMARY] GeoFAR-SH vs SH-only-10k: extra memory {(geo_mem - (safe_float(sh['mean_peak_gpu_memory_mb']) or 0.0)):+.1f} MB, extra size {(geo_size - (safe_float(sh['mean_total_model_size_mb']) or 0.0)):+.2f} MB")
    logger.log(f"[SUMMARY] LaTeX table path = {args.output_dir / 'efficiency_latex_table.tex'}")
    logger.log(f"[SUMMARY] LaTeX text path = {args.output_dir / 'efficiency_latex_text.tex'}")


if __name__ == "__main__":
    main()




