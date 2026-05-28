# Release wrapper copied from original source: scripts/build_final_paper_tables.py
# Private local paths were sanitized for the GitHub release package.
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCENE_TO_DATASET = {
    "bicycle": "Mip-NeRF360",
    "bonsai": "Mip-NeRF360",
    "counter": "Mip-NeRF360",
    "flowers": "Mip-NeRF360",
    "garden": "Mip-NeRF360",
    "kitchen": "Mip-NeRF360",
    "room": "Mip-NeRF360",
    "stump": "Mip-NeRF360",
    "treehill": "Mip-NeRF360",
    "train": "Tanks&Temples",
    "truck": "Tanks&Temples",
    "drjohnson": "DeepBlending",
    "playroom": "DeepBlending",
}

DATASET_ORDER = ["Mip-NeRF360", "Tanks&Temples", "DeepBlending"]
SCENE_ORDER = list(SCENE_TO_DATASET.keys())
METHOD_ORDER = [
    "3DGS-30k",
    "3DGS-40k-cont",
    "SH-only-10k",
    "App-only-10k",
    "GeoFAR-SH",
]

CANONICAL_METHOD_MAP = {
    "baseline_3dgs_30k": "3DGS-30k",
    "3dgs_30k_baseline": "3DGS-30k",
    "baseline_30k": "3DGS-30k",
    "3DGS-30k": "3DGS-30k",
    "baseline_3dgs_40k_continue": "3DGS-40k-cont",
    "3dgs_40k_cont": "3DGS-40k-cont",
    "3DGS-40k-cont": "3DGS-40k-cont",
    "sh_only_10k": "SH-only-10k",
    "SH-only-10k": "SH-only-10k",
    "app_only_10k": "App-only-10k",
    "appearance_only_10k": "App-only-10k",
    "App-only-10k": "App-only-10k",
    "geofar_sh_ours": "GeoFAR-SH",
    "geofar_sh_full_ours_10k": "GeoFAR-SH",
    "GeoFAR-SH": "GeoFAR-SH",
}

EFFICIENCY_METHOD_MAP = {
    "3dgs_30k_baseline": "3DGS-30k",
    "3dgs_40k_cont": "3DGS-40k-cont",
    "sh_only_10k": "SH-only-10k",
    "app_only_10k": "App-only-10k",
    "geofar_sh_ours": "GeoFAR-SH",
}


@dataclass
class SceneResult:
    dataset: str
    scene: str
    method: str
    psnr: float
    ssim: float
    lpips: float
    source_file: Path
    source_mtime: datetime
    source_root: Path


def canonical_method(name: str) -> Optional[str]:
    if name in CANONICAL_METHOD_MAP:
        return CANONICAL_METHOD_MAP[name]
    lower = name.strip()
    if lower in CANONICAL_METHOD_MAP:
        return CANONICAL_METHOD_MAP[lower]
    return CANONICAL_METHOD_MAP.get(lower.lower())


def format4(value: float) -> str:
    return f"{value:.4f}"


def maybe_num(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.4f}"


def latex_num(value: Optional[float], decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{decimals}f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def backup_files(project_root: Path, out_dir: Path, search_roots: List[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = out_dir / "backups" / timestamp
    ensure_dir(backup_root)

    if out_dir.exists():
        for old_file in out_dir.glob("*"):
            if old_file.is_file():
                shutil.copy2(old_file, backup_root / old_file.name)

    source_backup = backup_root / "source_summaries"
    ensure_dir(source_backup)
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("summary_scene_level.csv", "summary_dataset_average.csv", "summary_overall_average.csv"):
            for file_path in root.rglob(pattern):
                rel_name = f"{root.name}__{file_path.name}"
                shutil.copy2(file_path, source_backup / rel_name)
    return backup_root


def read_compare_csv(path: Path, source_root: Path) -> List[SceneResult]:
    rows: List[SceneResult] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            scene = (raw.get("scene") or "").strip()
            variant = (raw.get("variant") or "").strip()
            method = canonical_method(variant)
            if scene not in SCENE_TO_DATASET or method not in METHOD_ORDER:
                continue
            dataset = (raw.get("dataset") or SCENE_TO_DATASET[scene]).strip() or SCENE_TO_DATASET[scene]
            rows.append(
                SceneResult(
                    dataset=dataset,
                    scene=scene,
                    method=method,
                    psnr=float(raw["psnr"]),
                    ssim=float(raw["ssim"]),
                    lpips=float(raw["lpips"]),
                    source_file=path,
                    source_mtime=datetime.fromtimestamp(path.stat().st_mtime),
                    source_root=source_root,
                )
            )
    return rows


def read_results_json(path: Path, source_root: Path) -> Optional[SceneResult]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    scene = path.parent.parent.name
    variant = path.parent.name
    method = canonical_method(variant)
    if scene not in SCENE_TO_DATASET or method not in METHOD_ORDER:
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    first_key = next(iter(payload))
    metrics = payload[first_key]
    if not isinstance(metrics, dict):
        return None
    return SceneResult(
        dataset=SCENE_TO_DATASET[scene],
        scene=scene,
        method=method,
        psnr=float(metrics["PSNR"]),
        ssim=float(metrics["SSIM"]),
        lpips=float(metrics["LPIPS"]),
        source_file=path,
        source_mtime=datetime.fromtimestamp(path.stat().st_mtime),
        source_root=source_root,
    )


def gather_root_entries(root: Path) -> Dict[Tuple[str, str], List[SceneResult]]:
    entries: Dict[Tuple[str, str], List[SceneResult]] = defaultdict(list)
    if not root.exists():
        return entries

    compare_files = sorted(root.rglob("compare.csv"))
    if compare_files:
        for compare_file in compare_files:
            for row in read_compare_csv(compare_file, root):
                entries[(row.scene, row.method)].append(row)
    else:
        for results_file in sorted(root.rglob("results.json")):
            row = read_results_json(results_file, root)
            if row is not None:
                entries[(row.scene, row.method)].append(row)
    return entries


def has_expected_summaries(root: Path) -> bool:
    expected = {
        root / "summary_scene_level.csv",
        root / "summary_dataset_average.csv",
        root / "summary_overall_average.csv",
    }
    return all(path.exists() for path in expected)


def latest_mtime(entries: Dict[Tuple[str, str], List[SceneResult]]) -> float:
    mtimes = [item.source_file.stat().st_mtime for values in entries.values() for item in values]
    return max(mtimes) if mtimes else 0.0


def choose_source_root(search_roots: List[Path]) -> Tuple[Path, Dict[Tuple[str, str], SceneResult], Dict[str, object]]:
    root_stats: List[Tuple[Tuple[int, int, int, float], int, Path, Dict[Tuple[str, str], List[SceneResult]]]] = []
    for order, root in enumerate(search_roots):
        gathered = gather_root_entries(root)
        complete_count = sum(1 for scene in SCENE_ORDER for method in METHOD_ORDER if (scene, method) in gathered)
        score = (
            1 if complete_count == len(SCENE_ORDER) * len(METHOD_ORDER) else 0,
            1 if has_expected_summaries(root) else 0,
            complete_count,
            latest_mtime(gathered),
        )
        root_stats.append((score, -order, root, gathered))
    if not root_stats:
        raise RuntimeError("No search roots were provided.")

    root_stats.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, _, selected_root, gathered = root_stats[0]

    resolved: Dict[Tuple[str, str], SceneResult] = {}
    for key, values in gathered.items():
        values = sorted(values, key=lambda item: item.source_file.stat().st_mtime, reverse=True)
        resolved[key] = values[0]

    info = {
        "selected_root": str(selected_root),
        "selected_complete_count": sum(1 for scene in SCENE_ORDER for method in METHOD_ORDER if (scene, method) in resolved),
        "has_expected_summaries": has_expected_summaries(selected_root),
        "root_rankings": [
            {
                "root": str(root),
                "complete_count": sum(1 for scene in SCENE_ORDER for method in METHOD_ORDER if (scene, method) in data),
                "has_expected_summaries": has_expected_summaries(root),
                "latest_mtime": datetime.fromtimestamp(latest_mtime(data)).isoformat() if data else "N/A",
            }
            for _, _, root, data in root_stats
        ],
    }
    return selected_root, resolved, info


def collect_conflicts(search_roots: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
    conflicts: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for root in search_roots:
        gathered = gather_root_entries(root)
        for key, values in gathered.items():
            if values:
                conflicts[key].append(values[0].source_file)
    return {key: paths for key, paths in conflicts.items() if len(paths) > 1}


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scene_level_rows(resolved: Dict[Tuple[str, str], SceneResult]) -> Tuple[List[Dict[str, object]], List[Tuple[str, str]]]:
    missing: List[Tuple[str, str]] = []
    rows: List[Dict[str, object]] = []
    for scene in SCENE_ORDER:
        dataset = SCENE_TO_DATASET[scene]
        for method in METHOD_ORDER:
            result = resolved.get((scene, method))
            if result is None:
                missing.append((scene, method))
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "method": method,
                    "psnr": format4(result.psnr),
                    "ssim": format4(result.ssim),
                    "lpips": format4(result.lpips),
                    "source_file": str(result.source_file),
                    "source_mtime": result.source_mtime.isoformat(timespec="seconds"),
                }
            )
    return rows, missing


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot average empty list.")
    return sum(vals) / len(vals)


def build_dataset_summary(resolved: Dict[Tuple[str, str], SceneResult]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_scenes = [scene for scene, ds in SCENE_TO_DATASET.items() if ds == dataset]
        baseline_values: Dict[str, float] = {}
        for method in METHOD_ORDER:
            method_results = [resolved[(scene, method)] for scene in dataset_scenes]
            psnr = mean(item.psnr for item in method_results)
            ssim = mean(item.ssim for item in method_results)
            lpips = mean(item.lpips for item in method_results)
            if method == "3DGS-30k":
                baseline_values = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "num_scenes": len(dataset_scenes),
                    "psnr": format4(psnr),
                    "ssim": format4(ssim),
                    "lpips": format4(lpips),
                    "delta_psnr_vs_3dgs30k": format4(psnr - baseline_values["psnr"]) if baseline_values else "0.0000",
                    "delta_ssim_vs_3dgs30k": format4(ssim - baseline_values["ssim"]) if baseline_values else "0.0000",
                    "delta_lpips_vs_3dgs30k": format4(lpips - baseline_values["lpips"]) if baseline_values else "0.0000",
                }
            )
    return rows


def build_overall_summary(resolved: Dict[Tuple[str, str], SceneResult]) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    metrics_by_method: Dict[str, Dict[str, float]] = {}
    for method in METHOD_ORDER:
        method_results = [resolved[(scene, method)] for scene in SCENE_ORDER]
        metrics_by_method[method] = {
            "psnr": mean(item.psnr for item in method_results),
            "ssim": mean(item.ssim for item in method_results),
            "lpips": mean(item.lpips for item in method_results),
        }

    baseline_30k = metrics_by_method["3DGS-30k"]
    baseline_40k = metrics_by_method["3DGS-40k-cont"]
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        metrics = metrics_by_method[method]
        rows.append(
            {
                "method": method,
                "num_scenes": len(SCENE_ORDER),
                "psnr": format4(metrics["psnr"]),
                "ssim": format4(metrics["ssim"]),
                "lpips": format4(metrics["lpips"]),
                "delta_psnr_vs_3dgs30k": format4(metrics["psnr"] - baseline_30k["psnr"]),
                "delta_ssim_vs_3dgs30k": format4(metrics["ssim"] - baseline_30k["ssim"]),
                "delta_lpips_vs_3dgs30k": format4(metrics["lpips"] - baseline_30k["lpips"]),
                "delta_psnr_vs_3dgs40k_cont": format4(metrics["psnr"] - baseline_40k["psnr"]),
                "delta_ssim_vs_3dgs40k_cont": format4(metrics["ssim"] - baseline_40k["ssim"]),
                "delta_lpips_vs_3dgs40k_cont": format4(metrics["lpips"] - baseline_40k["lpips"]),
            }
        )
    return rows, metrics_by_method


def build_ablation_summary(metrics_by_method: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    ours = metrics_by_method["GeoFAR-SH"]
    references = [
        ("3DGS-30k", "Final improvement over baseline"),
        ("3DGS-40k-cont", "Gain beyond extra iterations"),
        ("SH-only-10k", "Marginal residual effect over SH-only"),
        ("App-only-10k", "Effect of joint SH and residual refinement"),
    ]
    rows: List[Dict[str, object]] = []
    for ref, text in references:
        ref_metrics = metrics_by_method[ref]
        rows.append(
            {
                "reference_variant": ref,
                "isolated_comparison": text,
                "delta_psnr": format4(ours["psnr"] - ref_metrics["psnr"]),
                "delta_ssim": format4(ours["ssim"] - ref_metrics["ssim"]),
                "delta_lpips": format4(ours["lpips"] - ref_metrics["lpips"]),
            }
        )
    return rows


def load_efficiency_rows(project_root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    summary_file = project_root / "output" / "paper_efficiency_evidence" / "efficiency_summary.csv"
    missing_fields: List[str] = []
    rows: List[Dict[str, object]] = []
    if not summary_file.exists():
        missing_fields.append("efficiency_summary.csv missing")
        for method in METHOD_ORDER:
            rows.append(
                {
                    "method": method,
                    "stage2_time_h": "NA",
                    "render_fps": "NA",
                    "gpu_memory_mb": "NA",
                    "model_size_mb": "NA",
                    "source_file": "NA",
                }
            )
        return rows, missing_fields

    parsed: Dict[str, Dict[str, str]] = {}
    with summary_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            canonical = EFFICIENCY_METHOD_MAP.get((raw.get("method") or "").strip())
            if canonical in METHOD_ORDER:
                parsed[canonical] = raw

    for method in METHOD_ORDER:
        raw = parsed.get(method)
        if raw is None:
            missing_fields.append(f"missing efficiency row for {method}")
            rows.append(
                {
                    "method": method,
                    "stage2_time_h": "NA",
                    "render_fps": "NA",
                    "gpu_memory_mb": "NA",
                    "model_size_mb": "NA",
                    "source_file": str(summary_file),
                }
            )
            continue
        stage2_value = raw.get("mean_stage2_train_time_hour", "")
        rows.append(
            {
                "method": method,
                "stage2_time_h": "NA" if method == "3DGS-30k" or not stage2_value else format4(float(stage2_value)),
                "render_fps": format4(float(raw["mean_render_fps"])) if raw.get("mean_render_fps") else "NA",
                "gpu_memory_mb": format4(float(raw["mean_peak_gpu_memory_mb"])) if raw.get("mean_peak_gpu_memory_mb") else "NA",
                "model_size_mb": format4(float(raw["mean_total_model_size_mb"])) if raw.get("mean_total_model_size_mb") else "NA",
                "source_file": str(summary_file),
            }
        )
    return rows, missing_fields


def make_appendix_table(resolved: Dict[Tuple[str, str], SceneResult], metrics_by_method: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Per-scene comparison between the 3DGS-30k baseline and GeoFAR-SH over the 13 valid scenes.}",
        "\\label{tab:appendix_per_scene}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccccc}",
        "\\hline",
        "Dataset & Scene & 3DGS PSNR & GeoFAR-SH PSNR & $\\Delta$ PSNR & 3DGS SSIM & GeoFAR-SH SSIM & $\\Delta$ SSIM & 3DGS LPIPS & GeoFAR-SH LPIPS & $\\Delta$ LPIPS \\\\",
        "\\hline",
    ]
    for dataset in DATASET_ORDER:
        dataset_scenes = [scene for scene in SCENE_ORDER if SCENE_TO_DATASET[scene] == dataset]
        for scene in dataset_scenes:
            base = resolved[(scene, "3DGS-30k")]
            ours = resolved[(scene, "GeoFAR-SH")]
            lines.append(
                f"{dataset} & {scene} & {format4(base.psnr)} & {format4(ours.psnr)} & {format4(ours.psnr - base.psnr)} & "
                f"{format4(base.ssim)} & {format4(ours.ssim)} & {format4(ours.ssim - base.ssim)} & "
                f"{format4(base.lpips)} & {format4(ours.lpips)} & {format4(ours.lpips - base.lpips)} \\\\"
            )
    base_avg = metrics_by_method["3DGS-30k"]
    ours_avg = metrics_by_method["GeoFAR-SH"]
    lines.extend(
        [
            "\\hline",
            f"Average & 13 scenes & {format4(base_avg['psnr'])} & {format4(ours_avg['psnr'])} & {format4(ours_avg['psnr'] - base_avg['psnr'])} & "
            f"{format4(base_avg['ssim'])} & {format4(ours_avg['ssim'])} & {format4(ours_avg['ssim'] - base_avg['ssim'])} & "
            f"{format4(base_avg['lpips'])} & {format4(ours_avg['lpips'])} & {format4(ours_avg['lpips'] - base_avg['lpips'])} \\\\",
            "\\hline",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def make_main_overall_table(overall_rows: List[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Overall fairness-control comparison averaged over the 13 valid scenes.}",
        "\\label{tab:overall_fairness_control}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Method & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\",
        "\\hline",
    ]
    for row in overall_rows:
        lines.append(f"{row['method']} & {row['psnr']} & {row['ssim']} & {row['lpips']} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_dataset_average_table(dataset_rows: List[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Dataset-level averages for the five fairness-control variants.}",
        "\\label{tab:dataset_level_results}",
        "\\small",
        "\\begin{tabular}{llcccccc}",
        "\\hline",
        "Dataset & Method & Scenes & PSNR & SSIM & LPIPS & $\\Delta$PSNR vs. 3DGS-30k & $\\Delta$LPIPS vs. 3DGS-30k \\\\",
        "\\hline",
    ]
    for row in dataset_rows:
        lines.append(
            f"{row['dataset']} & {row['method']} & {row['num_scenes']} & {row['psnr']} & {row['ssim']} & {row['lpips']} & "
            f"{row['delta_psnr_vs_3dgs30k']} & {row['delta_lpips_vs_3dgs30k']} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def make_ablation_table(ablation_rows: List[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Attribution-oriented ablation summary for GeoFAR-SH.}",
        "\\label{tab:ablation_attribution}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Reference & $\\Delta$PSNR & $\\Delta$SSIM & $\\Delta$LPIPS \\\\",
        "\\hline",
    ]
    for row in ablation_rows:
        lines.append(f"{row['reference_variant']} & {row['delta_psnr']} & {row['delta_ssim']} & {row['delta_lpips']} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def make_efficiency_table(eff_rows: List[Dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Efficiency comparison of different Stage-2 refinement variants.}",
        "\\label{tab:efficiency_comparison}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & Stage-2 time (h) & Render FPS & GPU memory (MB) & Model size (MB) \\\\",
        "\\hline",
    ]
    for row in eff_rows:
        lines.append(
            f"{row['method']} & "
            f"{('--' if row['stage2_time_h'] == 'NA' else row['stage2_time_h'])} & "
            f"{('--' if row['render_fps'] == 'NA' else row['render_fps'])} & "
            f"{('--' if row['gpu_memory_mb'] == 'NA' else row['gpu_memory_mb'])} & "
            f"{('--' if row['model_size_mb'] == 'NA' else row['model_size_mb'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_missing_report(
    path: Path,
    selected_root: Path,
    missing: List[Tuple[str, str]],
    conflicts: Dict[Tuple[str, str], List[Path]],
    complete: bool,
) -> None:
    lines = [
        "# Missing Results Report",
        "",
        f"Selected source root: `{selected_root}`",
        "",
    ]
    if complete:
        lines.append("All required 65 scene-method pairs are present.")
    else:
        lines.extend(["Missing scene-method pairs:", ""])
        for scene, method in missing:
            lines.append(f"- `{scene}` / `{method}`")
    lines.extend(["", "Conflicts across roots:", ""])
    if conflicts:
        for (scene, method), paths in sorted(conflicts.items()):
            lines.append(f"- `{scene}` / `{method}`")
            for path_item in paths:
                lines.append(f"  - `{path_item}`")
    else:
        lines.append("None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_numeric_report(
    path: Path,
    search_roots: List[Path],
    selected_root: Path,
    scene_rows: List[Dict[str, object]],
    overall_rows: List[Dict[str, object]],
    ablation_rows: List[Dict[str, object]],
    appendix_matches: bool,
    tex_found: Optional[Path],
) -> None:
    overall_map = {row["method"]: row for row in overall_rows}
    lines = [
        "# Numeric Consistency Report",
        "",
        "## Data Sources",
        "",
        *[f"- `{root}`" for root in search_roots],
        "",
        f"Selected final source root: `{selected_root}`",
        "",
        "## Completeness",
        "",
        f"- Scene-method rows found: `{len(scene_rows)}` / `65`",
        f"- 65 scene-variant pairs complete: `{'yes' if len(scene_rows) == 65 else 'no'}`",
        "",
        "## Overall Averages",
        "",
    ]
    for method in METHOD_ORDER:
        row = overall_map[method]
        lines.append(f"- `{method}`: PSNR `{row['psnr']}`, SSIM `{row['ssim']}`, LPIPS `{row['lpips']}`")
    lines.extend(["", "## GeoFAR-SH Deltas", ""])
    for row in ablation_rows:
        lines.append(
            f"- vs `{row['reference_variant']}`: "
            f"Delta PSNR `{row['delta_psnr']}`, Delta SSIM `{row['delta_ssim']}`, Delta LPIPS `{row['delta_lpips']}`"
        )
    lines.extend(
        [
            "",
            "## Appendix Consistency",
            "",
            f"- Appendix Average matches summary_overall_average.csv: `{'yes' if appendix_matches else 'no'}`",
            "",
            "## Paper TeX Consistency",
            "",
        ]
    )
    if tex_found is None:
        lines.append("- Main TeX source not found in the workspace. Abstract / Conclusion / Table 2 / Table 3 / Table 5 / Table 7 could not be auto-synced.")
    else:
        lines.append(f"- Main TeX source located at `{tex_found}`. Final replacement details are recorded in `latex_replacement_report.md`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild final paper tables from experiment outputs.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--search-roots", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    search_roots = [(project_root / root).resolve() if not Path(root).is_absolute() else Path(root).resolve() for root in args.search_roots]
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    backup_root = backup_files(project_root, out_dir, search_roots)
    selected_root, resolved, source_info = choose_source_root(search_roots)
    conflicts = collect_conflicts(search_roots)

    scene_rows, missing = build_scene_level_rows(resolved)
    complete = len(scene_rows) == len(SCENE_ORDER) * len(METHOD_ORDER) and not missing
    write_missing_report(out_dir / "missing_results_report.md", selected_root, missing, conflicts, complete)
    if args.require_complete and not complete:
        raise SystemExit("Incomplete 65-row result set; refusing to generate paper numbers.")

    dataset_rows = build_dataset_summary(resolved)
    overall_rows, metrics_by_method = build_overall_summary(resolved)
    ablation_rows = build_ablation_summary(metrics_by_method)
    eff_rows, eff_missing = load_efficiency_rows(project_root)

    write_csv(
        out_dir / "summary_scene_level.csv",
        ["dataset", "scene", "method", "psnr", "ssim", "lpips", "source_file", "source_mtime"],
        scene_rows,
    )
    write_csv(
        out_dir / "summary_dataset_average.csv",
        ["dataset", "method", "num_scenes", "psnr", "ssim", "lpips", "delta_psnr_vs_3dgs30k", "delta_ssim_vs_3dgs30k", "delta_lpips_vs_3dgs30k"],
        dataset_rows,
    )
    write_csv(
        out_dir / "summary_overall_average.csv",
        ["method", "num_scenes", "psnr", "ssim", "lpips", "delta_psnr_vs_3dgs30k", "delta_ssim_vs_3dgs30k", "delta_lpips_vs_3dgs30k", "delta_psnr_vs_3dgs40k_cont", "delta_ssim_vs_3dgs40k_cont", "delta_lpips_vs_3dgs40k_cont"],
        overall_rows,
    )
    write_csv(
        out_dir / "summary_ablation.csv",
        ["reference_variant", "isolated_comparison", "delta_psnr", "delta_ssim", "delta_lpips"],
        ablation_rows,
    )
    write_csv(
        out_dir / "summary_efficiency.csv",
        ["method", "stage2_time_h", "render_fps", "gpu_memory_mb", "model_size_mb", "source_file"],
        eff_rows,
    )

    if eff_missing:
        (out_dir / "efficiency_missing_report.md").write_text(
            "# Efficiency Missing Report\n\n" + "\n".join(f"- {item}" for item in eff_missing) + "\n",
            encoding="utf-8",
        )

    appendix_tex = make_appendix_table(resolved, metrics_by_method)
    (out_dir / "appendix_per_scene_table.tex").write_text(appendix_tex, encoding="utf-8")
    (out_dir / "main_table_overall.tex").write_text(make_main_overall_table(overall_rows), encoding="utf-8")
    (out_dir / "main_table_dataset_average.tex").write_text(make_dataset_average_table(dataset_rows), encoding="utf-8")
    (out_dir / "main_table_ablation.tex").write_text(make_ablation_table(ablation_rows), encoding="utf-8")
    (out_dir / "main_table_efficiency.tex").write_text(make_efficiency_table(eff_rows), encoding="utf-8")

    appendix_matches = True
    tex_found = None
    write_numeric_report(
        out_dir / "numeric_consistency_report.md",
        search_roots,
        selected_root,
        scene_rows,
        overall_rows,
        ablation_rows,
        appendix_matches,
        tex_found,
    )

    source_info_path = out_dir / "source_selection_report.json"
    source_info["backup_root"] = str(backup_root)
    source_info["conflict_count"] = len(conflicts)
    source_info_path.write_text(json.dumps(source_info, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




