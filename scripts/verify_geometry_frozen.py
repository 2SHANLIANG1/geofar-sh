# Release wrapper copied from original source: scripts/verify_geometry_frozen_stage2.py
# Private local paths were sanitized for the GitHub release package.
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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

REPO = Path(__file__).resolve().parents[1]
GAS_OUTPUT = Path(r"<OUTPUT_ROOT>\output")
GAS_OUTPUT_CKPT30K = Path(r"<OUTPUT_ROOT>\output_ckpt30k")
FAIRNESS_ROOT = REPO / "output" / "paper_fairness_controls"
OUT_ROOT = REPO / "output" / "paper_geometry_frozen_verification"
LOG_PATH = OUT_ROOT / "verify_geometry_frozen.log"
FAILED_MD = OUT_ROOT / "failed_jobs.md"
PER_SCENE_CSV = OUT_ROOT / "geometry_frozen_per_scene.csv"
SUMMARY_CSV = OUT_ROOT / "geometry_frozen_summary.csv"
SUMMARY_MD = OUT_ROOT / "geometry_frozen_summary.md"
LATEX_TEX = OUT_ROOT / "geometry_frozen_latex_table.tex"

GEOMETRY_TENSORS = {
    "_xyz": ("Position", False),
    "_scaling": ("Scale", False),
    "_rotation": ("Rotation", False),
    "_opacity": ("Opacity", False),
}
SH_TENSORS = {
    "_features_dc": ("SH coefficients", True),
    "_features_rest": ("SH coefficients", True),
}
APPEARANCE_LATENT_KEYWORDS = ("appearance", "latent", "app_latent")
RESIDUAL_HEAD_KEYWORDS = ("diffuse", "specular", "spec", "diff", "mask", "gate", "residual", "head", "mlp", "fastkan", "app_")


@dataclass
class CheckpointInfo:
    path: Path
    iteration: int | None
    format_name: str
    tensors: dict[str, torch.Tensor]
    inner_len: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(message)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


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


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def tensor_stats(tensor: torch.Tensor) -> tuple[float, float]:
    data = tensor.detach().to(torch.float64).cpu()
    return float(data.abs().mean().item()), float(data.abs().max().item())


def delta_stats(init_tensor: torch.Tensor, final_tensor: torch.Tensor) -> tuple[float, float, float]:
    delta = (final_tensor.detach().to(torch.float64).cpu() - init_tensor.detach().to(torch.float64).cpu()).abs()
    mean_abs = float(delta.mean().item())
    max_abs = float(delta.max().item())
    l2 = float(torch.linalg.vector_norm(delta.reshape(-1), ord=2).item())
    return mean_abs, max_abs, l2


def shape_str(tensor: torch.Tensor | int | None) -> str:
    if tensor is None:
        return ""
    if isinstance(tensor, int):
        return str(tensor)
    return "x".join(str(v) for v in tensor.shape)


def format_sci(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value == 0:
        return "0"
    if 0 < abs(value) < 1e-8:
        return r"$<10^{-8}$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.2f}\times10^{{{exponent}}}$"


def parse_checkpoint(path: Path) -> CheckpointInfo:
    payload = torch.load(path, map_location="cpu")
    iteration = None
    model_args = payload
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], tuple):
        model_args = payload[0]
        iteration = int(payload[1]) if isinstance(payload[1], int) else None
    if not isinstance(model_args, tuple):
        raise RuntimeError(f"Unsupported checkpoint format at {path}")

    tensors: dict[str, torch.Tensor] = {}
    inner_len = len(model_args)
    base_slots = {
        "_xyz": 1,
        "_features_dc": 2,
        "_features_rest": 3,
        "_scaling": 4,
        "_rotation": 5,
        "_opacity": 6,
    }
    for name, idx in base_slots.items():
        if inner_len > idx and torch.is_tensor(model_args[idx]):
            tensors[name] = model_args[idx].detach().cpu()

    if inner_len >= 32:
        appearance_slots = {
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
        for name, idx in appearance_slots.items():
            if inner_len > idx and torch.is_tensor(model_args[idx]):
                tensors[name] = model_args[idx].detach().cpu()

    format_name = f"tuple_len_{inner_len}"
    return CheckpointInfo(path=path, iteration=iteration, format_name=format_name, tensors=tensors, inner_len=inner_len)


def recursive_candidates(root: Path, patterns: list[str]) -> list[Path]:
    results: list[Path] = []
    if not root.exists():
        return results
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    deduped: list[Path] = []
    seen: set[str] = set()
    for item in results:
        key = str(item).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def find_init_checkpoint(scene: str) -> Path | None:
    candidates = [
        GAS_OUTPUT / scene / "chkpnt30000.pth",
        GAS_OUTPUT_CKPT30K / scene / "chkpnt30000.pth",
        GAS_OUTPUT / scene / "checkpoints" / "chkpnt30000.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in recursive_candidates(GAS_OUTPUT / scene, ["*30000*.pth"]):
        if "chkpnt30000" in path.name.lower():
            return path
    for path in recursive_candidates(REPO / "output", [f"*{scene}*30000*.pth"]):
        return path
    return None


def find_final_checkpoint(scene: str) -> Path | None:
    scene_root = FAIRNESS_ROOT / scene / "geofar_sh_ours"
    candidates = [
        scene_root / "chkpnt40000.pth",
        scene_root / "final_checkpoint.pth",
        scene_root / "best_checkpoint.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    direct_patterns = ["*40000*.pth", "*final_checkpoint*.pth", "*best_checkpoint*.pth"]
    for path in recursive_candidates(scene_root, direct_patterns):
        return path
    for path in recursive_candidates(REPO / "output", [f"*{scene}*chkpnt40000*.pth", f"*{scene}*final_checkpoint*.pth"]):
        lowered = str(path).lower()
        if "geofar_sh_ours" in lowered or "geofar-sh" in lowered or "geofar" in lowered:
            return path
    return None


def find_stage2_init_checkpoint(scene: str) -> Path | None:
    scene_root = FAIRNESS_ROOT / scene / "geofar_sh_ours"
    for path in recursive_candidates(scene_root, ["*30000*.pth", "*stage2*init*.pth"]):
        return path
    return None


def classify_tensor(name: str) -> tuple[str | None, bool]:
    if name in GEOMETRY_TENSORS:
        return GEOMETRY_TENSORS[name]
    if name in SH_TENSORS:
        return SH_TENSORS[name]
    lowered = name.lower()
    if any(token in lowered for token in APPEARANCE_LATENT_KEYWORDS) and "app_w" not in lowered and "app_b" not in lowered:
        return ("Appearance latent", True)
    if any(token in lowered for token in RESIDUAL_HEAD_KEYWORDS):
        return ("Residual heads", True)
    return (None, False)


def expected_status(group: str) -> str:
    if group in {"Number of Gaussians", "Position", "Scale", "Rotation", "Opacity"}:
        return "frozen_no_change"
    if group == "SH coefficients":
        return "trainable_should_change"
    if group in {"Appearance latent", "Residual heads"}:
        return "introduced_or_trainable"
    return "unknown"


def build_row(
    dataset: str,
    scene: str,
    group: str,
    tensor_name: str,
    trainable: bool,
    init_tensor: torch.Tensor | None,
    final_tensor: torch.Tensor | None,
    init_ckpt: Path,
    final_ckpt: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "scene": scene,
        "parameter_group": group,
        "tensor_name": tensor_name,
        "trainable_stage2": "Yes" if trainable else "No",
        "shape_init": shape_str(init_tensor),
        "shape_final": shape_str(final_tensor),
        "mean_abs_delta": "",
        "max_abs_delta": "",
        "l2_delta": "",
        "init_mean_abs": "",
        "final_mean_abs": "",
        "status": "missing",
        "init_checkpoint": str(init_ckpt),
        "final_checkpoint": str(final_ckpt),
    }

    if group == "Number of Gaussians":
        init_count = int(init_tensor) if init_tensor is not None else -1
        final_count = int(final_tensor) if final_tensor is not None else -1
        delta = abs(final_count - init_count)
        row["shape_init"] = str(init_count)
        row["shape_final"] = str(final_count)
        row["mean_abs_delta"] = f"{float(delta):.12g}"
        row["max_abs_delta"] = f"{float(delta):.12g}"
        row["l2_delta"] = f"{float(delta):.12g}"
        row["init_mean_abs"] = f"{float(init_count):.12g}"
        row["final_mean_abs"] = f"{float(final_count):.12g}"
        row["status"] = "PASS" if delta == 0 else "FAIL"
        return row

    if final_tensor is None:
        row["status"] = "FAIL" if not trainable else "missing"
        return row

    final_mean_abs, _ = tensor_stats(final_tensor)
    row["final_mean_abs"] = f"{final_mean_abs:.12g}"

    if init_tensor is None:
        row["status"] = "new_and_trained"
        return row

    if tuple(init_tensor.shape) != tuple(final_tensor.shape):
        init_mean_abs, _ = tensor_stats(init_tensor)
        row["init_mean_abs"] = f"{init_mean_abs:.12g}"
        row["status"] = "FAIL" if not trainable else "shape_mismatch"
        return row

    init_mean_abs, _ = tensor_stats(init_tensor)
    mean_abs_delta, max_abs_delta, l2_delta = delta_stats(init_tensor, final_tensor)
    row["init_mean_abs"] = f"{init_mean_abs:.12g}"
    row["mean_abs_delta"] = f"{mean_abs_delta:.12g}"
    row["max_abs_delta"] = f"{max_abs_delta:.12g}"
    row["l2_delta"] = f"{l2_delta:.12g}"

    if group in {"Position", "Scale", "Rotation", "Opacity"}:
        row["status"] = "PASS" if max_abs_delta <= 1e-8 else "FAIL"
    elif group == "SH coefficients":
        row["status"] = "WARNING" if mean_abs_delta == 0.0 and max_abs_delta == 0.0 else "PASS"
    else:
        row["status"] = "WARNING" if mean_abs_delta == 0.0 and max_abs_delta == 0.0 else "PASS"
    return row


def aggregate_group(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    group_rows = [row for row in rows if row["parameter_group"] == group]
    scene_count = len({row["scene"] for row in group_rows})
    fail_count = sum(1 for row in group_rows if row["status"] == "FAIL")
    mean_values = [safe_float(row["mean_abs_delta"]) for row in group_rows]
    mean_values = [value for value in mean_values if value is not None]
    max_values = [safe_float(row["max_abs_delta"]) for row in group_rows]
    max_values = [value for value in max_values if value is not None]
    final_values = [safe_float(row["final_mean_abs"]) for row in group_rows]
    final_values = [value for value in final_values if value is not None]
    return {
        "parameter_group": group,
        "scene_count": scene_count,
        "fail_count": fail_count,
        "mean_of_mean_abs_delta": "" if not mean_values else f"{sum(mean_values) / len(mean_values):.12g}",
        "max_of_max_abs_delta": "" if not max_values else f"{max(max_values):.12g}",
        "mean_final_abs": "" if not final_values else f"{sum(final_values) / len(final_values):.12g}",
        "expected_status": expected_status(group),
    }


def make_latex_table(summary_rows: list[dict[str, Any]]) -> str:
    by_group = {row["parameter_group"]: row for row in summary_rows}

    def latex_row(group: str, label: str, trainable: str, allow_new: bool = False) -> str:
        row = by_group.get(group, {})
        mean_delta = safe_float(row.get("mean_of_mean_abs_delta"))
        max_delta = safe_float(row.get("max_of_max_abs_delta"))
        if allow_new and mean_delta is None:
            mean_text = "new"
        else:
            mean_text = format_sci(mean_delta)
        max_text = format_sci(max_delta)
        if allow_new and safe_float(row.get("mean_final_abs")) is not None and mean_delta is None:
            max_text = format_sci(safe_float(row.get("mean_final_abs")))
        return f"{label} & {trainable} & {mean_text} & {max_text} \\\\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Verification of geometry-frozen optimization in Stage 2. The parameter differences are computed between the 30k initialization checkpoint and the final GeoFAR-SH checkpoint.}",
        r"\label{tab:geometry_frozen_verification}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Parameter group & Trainable in Stage 2 & Mean absolute change & Max absolute change \\",
        r"\hline",
        latex_row("Number of Gaussians", "Number of Gaussians", "No"),
        latex_row("Position", r"Position $\mathbf{x}$", "No"),
        latex_row("Scale", r"Scale $\mathbf{s}$", "No"),
        latex_row("Rotation", r"Rotation $\mathbf{q}$", "No"),
        latex_row("Opacity", r"Opacity $\alpha$", "No"),
        latex_row("SH coefficients", r"SH coefficients $f^{\mathrm{SH}}$", "Yes"),
        latex_row("Appearance latent", r"Appearance latent $\mathbf{z}$", "Yes", allow_new=True),
        latex_row("Residual heads", r"Residual heads $\theta^{\mathrm{app}}$", "Yes", allow_new=True),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "% Note: appearance latent and residual head parameters are newly introduced in Stage 2 when the 30k baseline checkpoint does not contain them.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(OUT_ROOT)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("Starting geometry-frozen Stage 2 verification.")
    failed_jobs: list[dict[str, str]] = []
    per_scene_rows: list[dict[str, Any]] = []
    scene_status_lines: list[str] = []

    for dataset, scene in SCENES:
        init_path = find_init_checkpoint(scene)
        final_path = find_final_checkpoint(scene)
        stage2_init_path = find_stage2_init_checkpoint(scene)

        if init_path is None or final_path is None:
            reason = []
            if init_path is None:
                reason.append("missing init checkpoint")
            if final_path is None:
                reason.append("missing final checkpoint")
            failed_jobs.append(
                {
                    "scene": scene,
                    "variant": "geofar_sh_ours",
                    "stage": "checkpoint_search",
                    "error": ", ".join(reason),
                    "log_path": str(LOG_PATH),
                }
            )
            log(f"[FAIL] {scene}: {'; '.join(reason)}")
            continue

        log(f"[INFO] {scene}: init checkpoint -> {init_path}")
        log(f"[INFO] {scene}: final checkpoint -> {final_path}")
        if stage2_init_path is not None:
            log(f"[INFO] {scene}: stage2 appearance init checkpoint -> {stage2_init_path}")
        init_ckpt = parse_checkpoint(init_path)
        final_ckpt = parse_checkpoint(final_path)
        stage2_init_ckpt = parse_checkpoint(stage2_init_path) if stage2_init_path is not None else None

        rows_for_scene: list[dict[str, Any]] = []
        rows_for_scene.append(
            build_row(
                dataset,
                scene,
                "Number of Gaussians",
                "__num_gaussians__",
                False,
                int(init_ckpt.tensors["_xyz"].shape[0]) if "_xyz" in init_ckpt.tensors else None,
                int(final_ckpt.tensors["_xyz"].shape[0]) if "_xyz" in final_ckpt.tensors else None,
                init_path,
                final_path,
            )
        )

        handled_names: set[str] = set()
        for tensor_name, (group, trainable) in {**GEOMETRY_TENSORS, **SH_TENSORS}.items():
            init_tensor = init_ckpt.tensors.get(tensor_name)
            final_tensor = final_ckpt.tensors.get(tensor_name)
            rows_for_scene.append(build_row(dataset, scene, group, tensor_name, trainable, init_tensor, final_tensor, init_path, final_path))
            handled_names.add(tensor_name)

        all_names = sorted(set(init_ckpt.tensors.keys()) | set(final_ckpt.tensors.keys()))
        for tensor_name in all_names:
            if tensor_name in handled_names:
                continue
            group, trainable = classify_tensor(tensor_name)
            if group is None:
                continue
            init_tensor = None
            if stage2_init_ckpt is not None and tensor_name in stage2_init_ckpt.tensors:
                init_tensor = stage2_init_ckpt.tensors[tensor_name]
            elif tensor_name in init_ckpt.tensors:
                init_tensor = init_ckpt.tensors[tensor_name]
            final_tensor = final_ckpt.tensors.get(tensor_name)
            rows_for_scene.append(build_row(dataset, scene, group, tensor_name, trainable, init_tensor, final_tensor, init_path, final_path))

        per_scene_rows.extend(rows_for_scene)
        geometry_rows = [row for row in rows_for_scene if row["parameter_group"] in {"Number of Gaussians", "Position", "Scale", "Rotation", "Opacity"}]
        sh_rows = [row for row in rows_for_scene if row["parameter_group"] == "SH coefficients"]
        app_rows = [row for row in rows_for_scene if row["parameter_group"] == "Appearance latent"]
        residual_rows = [row for row in rows_for_scene if row["parameter_group"] == "Residual heads"]

        geometry_ok = all(row["status"] == "PASS" for row in geometry_rows)
        sh_changed = any((safe_float(row["max_abs_delta"]) or 0.0) > 0.0 for row in sh_rows)
        appearance_present = bool(app_rows)
        residual_present = bool(residual_rows)
        scene_status = "PASS" if geometry_ok else "FAIL"
        scene_status_lines.append(
            f"- {scene}: {scene_status}; geometry_frozen={'yes' if geometry_ok else 'no'}; sh_changed={'yes' if sh_changed else 'no'}; appearance_latent={'yes' if appearance_present else 'no'}; residual_heads={'yes' if residual_present else 'no'}"
        )
        log(scene_status_lines[-1])

    headers = [
        "dataset",
        "scene",
        "parameter_group",
        "tensor_name",
        "trainable_stage2",
        "shape_init",
        "shape_final",
        "mean_abs_delta",
        "max_abs_delta",
        "l2_delta",
        "init_mean_abs",
        "final_mean_abs",
        "status",
        "init_checkpoint",
        "final_checkpoint",
    ]
    write_csv(PER_SCENE_CSV, per_scene_rows, headers)

    summary_order = [
        "Number of Gaussians",
        "Position",
        "Scale",
        "Rotation",
        "Opacity",
        "SH coefficients",
        "Appearance latent",
        "Residual heads",
    ]
    summary_rows = [aggregate_group(per_scene_rows, group) for group in summary_order]
    write_csv(
        SUMMARY_CSV,
        summary_rows,
        ["parameter_group", "scene_count", "fail_count", "mean_of_mean_abs_delta", "max_of_max_abs_delta", "mean_final_abs", "expected_status"],
    )
    write_text(LATEX_TEX, make_latex_table(summary_rows))

    if failed_jobs:
        failed_lines = [
            "# Failed Jobs",
            "",
            "| scene | variant | stage | error | log path |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in failed_jobs:
            failed_lines.append(f"| {item['scene']} | {item['variant']} | {item['stage']} | {item['error']} | {item['log_path']} |")
        write_text(FAILED_MD, "\n".join(failed_lines) + "\n")
    else:
        write_text(FAILED_MD, "# Failed Jobs\n\nNo failed scenes.\n")

    geometry_failures = [row for row in per_scene_rows if row["parameter_group"] in {"Number of Gaussians", "Position", "Scale", "Rotation", "Opacity"} and row["status"] == "FAIL"]
    sh_warnings = [row for row in per_scene_rows if row["parameter_group"] == "SH coefficients" and row["status"] in {"WARNING", "PASS"}]
    sh_any_changed = any((safe_float(row["max_abs_delta"]) or 0.0) > 0.0 for row in sh_warnings)
    app_present = any(row["parameter_group"] == "Appearance latent" for row in per_scene_rows)
    residual_present = any(row["parameter_group"] == "Residual heads" for row in per_scene_rows)

    summary_lines = [
        "# Geometry-Frozen Verification Summary",
        "",
        "## Scene Status",
        *scene_status_lines,
        "",
        "## Overall",
        f"- Geometry parameters all frozen successfully: {'yes' if not geometry_failures and len(scene_status_lines) == len(SCENES) else 'no'}",
        f"- SH coefficients changed: {'yes' if sh_any_changed else 'no'}",
        f"- Appearance latent exists: {'yes' if app_present else 'no'}",
        f"- Residual heads exist: {'yes' if residual_present else 'no'}",
        f"- LaTeX table path: `{LATEX_TEX}`",
    ]
    write_text(SUMMARY_MD, "\n".join(summary_lines) + "\n")

    log("")
    for line in scene_status_lines:
        log(line)
    log(f"Geometry parameters all frozen successfully: {'yes' if not geometry_failures and len(scene_status_lines) == len(SCENES) else 'no'}")
    log(f"SH coefficients changed: {'yes' if sh_any_changed else 'no'}")
    log(f"Appearance latent exists: {'yes' if app_present else 'no'}")
    log(f"Residual heads exist: {'yes' if residual_present else 'no'}")
    log(f"LaTeX table path: {LATEX_TEX}")


if __name__ == "__main__":
    main()




