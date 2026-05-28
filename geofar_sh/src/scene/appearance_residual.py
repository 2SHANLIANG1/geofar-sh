import json
import json
import os
from dataclasses import asdict, dataclass

import torch


@dataclass
class AppearanceResidualConfig:
    mode: str = "none"
    enable_stage2: bool = True
    residual_scale: float = 0.05
    enable_step: int = 7000
    warmup_steps: int = 5000
    schedule_type: str = "linear"
    lambda_residual_reg: float = 1e-4
    lambda_gate_reg: float = 0.0
    lambda_smooth_reg: float = 0.0
    lr_main: float = 1e-4
    lr_gate: float = 5e-5
    lr_latent: float = 5e-4
    gate_enabled: bool = True
    smooth_epsilon: float = 0.03
    latent_dim: int = 8
    gate_floor: float = 0.02
    stage2_start: int = 15000
    stage2_iters: int = 7000
    freeze_geometry_in_stage2: bool = True
    disable_densify_in_stage2: bool = True
    detach_xyz_grad: bool = True
    detach_shape_grad: bool = True
    lambda_max: float = 1.0
    lambda_warmup_iters: int = 5000
    use_local_aniso_encoding: bool = False
    stage2_joint_refine: bool = False
    stage2_refine_sh: bool = False
    stage2_lr_xyz: float = 1.6e-6
    stage2_lr_f_dc: float = 2.5e-5
    stage2_lr_f_rest: float = 1.25e-6
    stage2_lr_opacity: float = 1e-5
    stage2_lr_scale: float = 5e-6
    stage2_lr_rotation: float = 5e-6
    stage2_anchor_lambda_xyz: float = 0.01
    stage2_anchor_lambda_sh: float = 0.001
    stage2_anchor_lambda_opacity: float = 0.001
    stage2_anchor_lambda_scale: float = 0.001
    stage2_anchor_lambda_rotation: float = 0.001
    stage2_enable_residual_densify: bool = False
    stage2_densify_from_iter: int = 1000
    stage2_densify_until_iter: int = 9000
    stage2_prune_from_iter: int = 1000
    stage2_prune_until_iter: int = 11000
    stage2_densify_mode: str = "residual_guided"
    # Paper main method default: DDSR only. Disable explicitly for ablations.
    use_decoupled_residual: bool = True
    disable_decoupled_residual: bool = False
    lambda_spec_mask_reg: float = 0.0
    spec_mask_entropy_reg: float = 0.0
    spec_mask_temperature: float = 2.0
    disable_global_gate: bool = False
    stage2_refine_opacity: bool = True
    stage2_refine_scale: bool = True
    stage2_refine_rotation: bool = True
    stage2_opacity_lr: float = 1e-5
    stage2_scale_lr: float = 5e-6
    stage2_rotation_lr: float = 5e-6
    stage2_scale_delta_clip: float = 0.03
    stage2_opacity_delta_clip: float = 0.05
    stage2_rotation_delta_clip: float = 0.03
    stage2_geom_grad_clip: float = 0.1
    stage2_geom_unfreeze_iter: int = 1500
    use_two_layer_ddsr_heads: bool = False
    ddsr_head_hidden_dim: int = 8
    ddsr_head_activation: str = "silu"
    appearance_head_hidden_dim: int = 8
    appearance_head_num_layers: int = 1
    appearance_head_activation: str = "silu"
    enable_image_embedding: bool = False
    image_embedding_dim: int = 0
    enable_exposure_embedding: bool = False
    exposure_embedding_dim: int = 0
    enable_extra_geo_features: bool = False
    lambda_diff_consistency: float = 1e-4
    lambda_branch_diversity: float = 5e-6
    stage2_refine_xyz: bool = False
    disable_prune_in_stage2: bool = True
    freeze_exposure_in_stage2: bool = False
    enable_diffuse_residual: bool = True
    enable_specular_residual: bool = True
    enable_specular_mask: bool = True
    enable_global_gate: bool = True
    residual_mode: str = "full"
    appearance_compute_mode: str = "fused"


def _parse_optional_bool(value, default=None):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "none"}:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_residual_mode(value, default="full"):
    text = str(value).strip().lower() if value is not None else default
    if text in {"", "auto", "none"}:
        return default
    if text not in {"full", "single", "diffuse_only", "specular_only"}:
        return default
    return text


def _normalize_compute_mode(value, default="fused"):
    text = str(value).strip().lower() if value is not None else default
    if text in {"", "auto", "none"}:
        return default
    if text not in {"fused", "torch_precompute", "disabled"}:
        return default
    return text


def resolve_appearance_mode(args):
    explicit_enable = _parse_optional_bool(getattr(args, "enable_appearance_residual", None), None)
    if explicit_enable is not None:
        return "cuda_latent" if explicit_enable else "none"
    if getattr(args, "use_appearance_residual", False):
        return "cuda_latent"
    if getattr(args, "use_fastkan_residual", False) or getattr(args, "use_mlp_residual", False):
        return "cuda_latent"
    return "none"


def build_appearance_config(args):
    enable_sh_refine = _parse_optional_bool(getattr(args, "enable_sh_refine", None), bool(getattr(args, "stage2_refine_sh", False)))
    freeze_xyz = _parse_optional_bool(getattr(args, "freeze_xyz", None), None)
    freeze_scaling = _parse_optional_bool(getattr(args, "freeze_scaling", None), None)
    freeze_rotation = _parse_optional_bool(getattr(args, "freeze_rotation", None), None)
    freeze_opacity = _parse_optional_bool(getattr(args, "freeze_opacity", None), None)
    freeze_exposure = _parse_optional_bool(getattr(args, "freeze_exposure", None), None)
    disable_densify_stage2 = _parse_optional_bool(
        getattr(args, "disable_densify_stage2", None),
        bool(getattr(args, "appearance_disable_densify_in_stage2", True)),
    )
    disable_prune_stage2 = _parse_optional_bool(getattr(args, "disable_prune_stage2", None), disable_densify_stage2)
    stage2_refine_xyz = not freeze_xyz if freeze_xyz is not None else bool(getattr(args, "stage2_joint_refine", False))
    stage2_refine_scale = not freeze_scaling if freeze_scaling is not None else (
        bool(getattr(args, "stage2_refine_scale", True))
        and not bool(getattr(args, "disable_stage2_refine_scale", False))
    )
    stage2_refine_rotation = not freeze_rotation if freeze_rotation is not None else (
        bool(getattr(args, "stage2_refine_rotation", True))
        and not bool(getattr(args, "disable_stage2_refine_rotation", False))
    )
    stage2_refine_opacity = not freeze_opacity if freeze_opacity is not None else (
        bool(getattr(args, "stage2_refine_opacity", True))
        and not bool(getattr(args, "disable_stage2_refine_opacity", False))
    )
    residual_mode = _normalize_residual_mode(
        getattr(args, "residual_mode", None),
        "full" if bool(getattr(args, "use_decoupled_residual", True)) and not bool(getattr(args, "disable_decoupled_residual", False)) else "single",
    )
    use_decoupled_residual = (
        residual_mode != "single"
        and bool(getattr(args, "use_decoupled_residual", True))
        and not bool(getattr(args, "disable_decoupled_residual", False))
    )
    enable_diffuse = _parse_optional_bool(getattr(args, "enable_diffuse_residual", None), True)
    enable_specular = _parse_optional_bool(getattr(args, "enable_specular_residual", None), True)
    enable_mask = _parse_optional_bool(getattr(args, "enable_specular_mask", None), use_decoupled_residual and residual_mode == "full")
    enable_global_gate = _parse_optional_bool(getattr(args, "enable_global_gate", None), not bool(getattr(args, "disable_global_gate", False)))
    lambda_mask_reg = float(getattr(args, "lambda_mask_reg", -1.0))
    lambda_mask_entropy = float(getattr(args, "lambda_mask_entropy", -1.0))
    lambda_diffuse_consistency = float(getattr(args, "lambda_diffuse_consistency", -1.0))
    appearance_compute_mode = _normalize_compute_mode(getattr(args, "appearance_compute_mode", None), "fused")
    return AppearanceResidualConfig(
        mode=resolve_appearance_mode(args),
        enable_stage2=not bool(getattr(args, "disable_stage2", False)),
        residual_scale=float(getattr(args, "residual_scale", 0.05)),
        enable_step=int(getattr(args, "appearance_residual_enable_step", 7000)),
        warmup_steps=int(getattr(args, "appearance_residual_warmup_steps", 5000)),
        schedule_type=str(getattr(args, "appearance_residual_schedule", "linear")),
        lambda_residual_reg=float(getattr(args, "lambda_residual_reg", 1e-4)),
        lambda_gate_reg=float(getattr(args, "lambda_gate_reg", 0.0)),
        lambda_smooth_reg=float(getattr(args, "lambda_smooth_reg", 0.0)),
        lr_main=float(getattr(args, "lr_fastkan", 1e-4)),
        lr_gate=float(getattr(args, "lr_fastkan_gate", 5e-5)),
        lr_latent=float(getattr(args, "lr_appearance_latent", 5e-4)),
        gate_enabled=not bool(getattr(args, "disable_residual_gate", False)),
        smooth_epsilon=float(getattr(args, "appearance_smooth_epsilon", 0.03)),
        latent_dim=int(getattr(args, "appearance_latent_dim", 8)),
        gate_floor=float(getattr(args, "appearance_gate_floor", 0.02)),
        stage2_start=int(getattr(args, "appearance_stage2_start", 15000)),
        stage2_iters=int(getattr(args, "stage2_iters", getattr(args, "appearance_stage2_iters", 7000))),
        freeze_geometry_in_stage2=bool(getattr(args, "appearance_freeze_geometry_in_stage2", True)),
        disable_densify_in_stage2=bool(disable_densify_stage2),
        detach_xyz_grad=bool(getattr(args, "appearance_detach_xyz_grad", True)),
        detach_shape_grad=bool(getattr(args, "appearance_detach_shape_grad", True)),
        lambda_max=float(getattr(args, "appearance_lambda_max", 1.0)),
        lambda_warmup_iters=int(getattr(args, "appearance_lambda_warmup_iters", 5000)),
        use_local_aniso_encoding=bool(getattr(args, "use_local_aniso_encoding", False))
        or bool(getattr(args, "enable_extra_geo_features", False)),
        stage2_joint_refine=bool(getattr(args, "stage2_joint_refine", False)),
        stage2_refine_sh=bool(enable_sh_refine),
        stage2_lr_xyz=float(getattr(args, "stage2_lr_xyz", 1.6e-6)),
        stage2_lr_f_dc=float(getattr(args, "stage2_lr_f_dc", 2.5e-5)),
        stage2_lr_f_rest=float(getattr(args, "stage2_lr_f_rest", 1.25e-6)),
        stage2_lr_opacity=float(getattr(args, "stage2_lr_opacity", getattr(args, "stage2_opacity_lr", 1e-5))),
        stage2_lr_scale=float(getattr(args, "stage2_lr_scale", getattr(args, "stage2_scale_lr", 5e-6))),
        stage2_lr_rotation=float(getattr(args, "stage2_lr_rotation", getattr(args, "stage2_rotation_lr", 5e-6))),
        stage2_anchor_lambda_xyz=float(getattr(args, "stage2_anchor_lambda_xyz", 0.01)),
        stage2_anchor_lambda_sh=float(getattr(args, "stage2_anchor_lambda_sh", 0.001)),
        stage2_anchor_lambda_opacity=float(getattr(args, "stage2_anchor_lambda_opacity", 0.001)),
        stage2_anchor_lambda_scale=float(getattr(args, "stage2_anchor_lambda_scale", 0.001)),
        stage2_anchor_lambda_rotation=float(getattr(args, "stage2_anchor_lambda_rotation", 0.001)),
        stage2_enable_residual_densify=bool(getattr(args, "stage2_enable_residual_densify", False)),
        stage2_densify_from_iter=int(getattr(args, "stage2_densify_from_iter", 1000)),
        stage2_densify_until_iter=int(getattr(args, "stage2_densify_until_iter", 9000)),
        stage2_prune_from_iter=int(getattr(args, "stage2_prune_from_iter", 1000)),
        stage2_prune_until_iter=int(getattr(args, "stage2_prune_until_iter", 11000)),
        stage2_densify_mode=str(getattr(args, "stage2_densify_mode", "residual_guided")),
        use_decoupled_residual=bool(use_decoupled_residual),
        disable_decoupled_residual=bool(getattr(args, "disable_decoupled_residual", False)),
        lambda_spec_mask_reg=lambda_mask_reg if lambda_mask_reg >= 0.0 else float(getattr(args, "lambda_spec_mask_reg", 0.0)),
        spec_mask_entropy_reg=lambda_mask_entropy if lambda_mask_entropy >= 0.0 else float(getattr(args, "spec_mask_entropy_reg", 0.0)),
        spec_mask_temperature=float(getattr(args, "spec_mask_temperature", 2.0)),
        disable_global_gate=not bool(enable_global_gate),
        stage2_refine_opacity=bool(stage2_refine_opacity),
        stage2_refine_scale=bool(stage2_refine_scale),
        stage2_refine_rotation=bool(stage2_refine_rotation),
        stage2_opacity_lr=float(getattr(args, "stage2_opacity_lr", 1e-5)),
        stage2_scale_lr=float(getattr(args, "stage2_scale_lr", 5e-6)),
        stage2_rotation_lr=float(getattr(args, "stage2_rotation_lr", 5e-6)),
        stage2_scale_delta_clip=float(getattr(args, "stage2_scale_delta_clip", 0.03)),
        stage2_opacity_delta_clip=float(getattr(args, "stage2_opacity_delta_clip", 0.05)),
        stage2_rotation_delta_clip=float(getattr(args, "stage2_rotation_delta_clip", 0.03)),
        stage2_geom_grad_clip=float(getattr(args, "stage2_geom_grad_clip", 0.1)),
        stage2_geom_unfreeze_iter=int(getattr(args, "stage2_geom_unfreeze_iter", 1500)),
        use_two_layer_ddsr_heads=(
            bool(getattr(args, "use_two_layer_ddsr_heads", False))
            or int(getattr(args, "appearance_head_num_layers", 1)) >= 2
        )
        and not bool(getattr(args, "disable_two_layer_ddsr_heads", False)),
        ddsr_head_hidden_dim=int(getattr(args, "appearance_head_hidden_dim", getattr(args, "ddsr_head_hidden_dim", 8))),
        ddsr_head_activation=str(getattr(args, "appearance_head_activation", getattr(args, "ddsr_head_activation", "silu"))),
        appearance_head_hidden_dim=int(getattr(args, "appearance_head_hidden_dim", getattr(args, "ddsr_head_hidden_dim", 8))),
        appearance_head_num_layers=int(getattr(args, "appearance_head_num_layers", 1)),
        appearance_head_activation=str(getattr(args, "appearance_head_activation", getattr(args, "ddsr_head_activation", "silu"))),
        enable_image_embedding=bool(getattr(args, "enable_image_embedding", False)),
        image_embedding_dim=int(getattr(args, "image_embedding_dim", 0)),
        enable_exposure_embedding=bool(getattr(args, "enable_exposure_embedding", False)),
        exposure_embedding_dim=int(getattr(args, "exposure_embedding_dim", 0)),
        enable_extra_geo_features=bool(getattr(args, "enable_extra_geo_features", False)),
        lambda_diff_consistency=lambda_diffuse_consistency if lambda_diffuse_consistency >= 0.0 else float(getattr(args, "lambda_diff_consistency", 1e-4)),
        lambda_branch_diversity=float(getattr(args, "lambda_branch_diversity", 5e-6)),
        stage2_refine_xyz=bool(stage2_refine_xyz),
        disable_prune_in_stage2=bool(disable_prune_stage2),
        freeze_exposure_in_stage2=bool(freeze_exposure) if freeze_exposure is not None else False,
        enable_diffuse_residual=bool(enable_diffuse),
        enable_specular_residual=bool(enable_specular),
        enable_specular_mask=bool(enable_mask),
        enable_global_gate=bool(enable_global_gate),
        residual_mode=residual_mode,
        appearance_compute_mode=appearance_compute_mode,
    )


def build_appearance_head(config):
    # Deprecated: the legacy PyTorch residual head is no longer used in the hot path.
    return None


def build_appearance_meta_dict(config):
    return asdict(config)


def load_appearance_meta(meta_path):
    if not os.path.exists(meta_path):
        return None, f"[WARN] appearance residual metadata is missing at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle), ""


def compare_appearance_meta(expected_meta, loaded_meta):
    if loaded_meta is None:
        return ""
    mismatches = []
    for key, expected_value in expected_meta.items():
        if loaded_meta.get(key) != expected_value:
            mismatches.append(f"{key}: expected={expected_value} loaded={loaded_meta.get(key)}")
    if not mismatches:
        return ""
    return "[WARN] appearance residual metadata mismatch: {}".format("; ".join(mismatches))


def inspect_appearance_weight_file(enabled, weight_path):
    if not enabled:
        return "baseline_mode", ""
    if os.path.exists(weight_path):
        return "appearance_residual_loaded", ""
    return (
        "appearance_residual_requested_but_missing_weights",
        f"[WARN] appearance residual weights are missing at {weight_path}",
    )


def summarize_appearance_render_status(enabled, loaded):
    if not enabled:
        return "baseline mode"
    if loaded:
        return "appearance residual loaded"
    return "appearance residual requested but missing weights"


def appearance_head_parameter_norm(pc):
    if pc is None:
        return 0.0
    total = 0.0
    for tensor in (
        pc.app_w_rgb, pc.app_b_rgb, pc.app_w_gate, pc.app_b_gate,
        getattr(pc, "app_w_diff", None), getattr(pc, "app_b_diff", None),
        getattr(pc, "app_w_spec", None), getattr(pc, "app_b_spec", None),
        getattr(pc, "app_w_mask", None), getattr(pc, "app_b_mask", None),
        getattr(pc, "app_w2_gate", None), getattr(pc, "app_b2_gate", None),
        getattr(pc, "app_w2_diff", None), getattr(pc, "app_b2_diff", None),
        getattr(pc, "app_w2_spec", None), getattr(pc, "app_b2_spec", None),
        getattr(pc, "app_w2_mask", None), getattr(pc, "app_b2_mask", None),
    ):
        if tensor is None:
            continue
        total += float(tensor.detach().pow(2).sum().item())
    return total ** 0.5


def compute_appearance_schedule(iteration, enable_step, warmup_steps, schedule_type="linear"):
    if iteration is None:
        return 1.0
    if iteration < enable_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    t = min(max((iteration - enable_step) / float(warmup_steps), 0.0), 1.0)
    if schedule_type == "sigmoid":
        return float(torch.sigmoid(torch.tensor((t - 0.5) * 10.0)).item())
    return float(t)


def compute_stage2_lambda(iteration, config):
    if not getattr(config, "enable_stage2", True):
        return float(config.lambda_max)
    if iteration is None:
        return float(config.lambda_max)
    if iteration < config.stage2_start:
        return 0.0
    if config.lambda_warmup_iters <= 0:
        return float(config.lambda_max)
    t = min(max((iteration - config.stage2_start) / float(config.lambda_warmup_iters), 0.0), 1.0)
    return float(config.lambda_max * t)


def is_stage2_active(iteration, config):
    if config.mode == "none":
        return False
    if not getattr(config, "enable_stage2", True):
        return True
    if iteration is None:
        return True
    if iteration < config.stage2_start:
        return False
    if config.stage2_iters <= 0:
        return True
    return iteration < (config.stage2_start + config.stage2_iters)


def build_appearance_forward_state(pc, iteration):
    if not getattr(pc, "appearance_training_active", False):
        return {
            "enabled": False,
            "lambda_t": 0.0,
            "enable_stage2": bool(pc.appearance_config.enable_stage2),
            "gate_floor": float(pc.appearance_config.gate_floor),
            "detach_xyz_grad": bool(pc.appearance_config.detach_xyz_grad),
            "detach_shape_grad": bool(pc.appearance_config.detach_shape_grad),
            "use_local_aniso_encoding": bool(pc.appearance_config.use_local_aniso_encoding),
            "use_decoupled_residual": bool(pc.appearance_config.use_decoupled_residual),
            "disable_global_gate": bool(pc.appearance_config.disable_global_gate),
            "spec_mask_temperature": float(pc.appearance_config.spec_mask_temperature),
            "use_two_layer_ddsr_heads": bool(pc.appearance_config.use_two_layer_ddsr_heads),
            "ddsr_head_hidden_dim": int(pc.appearance_config.ddsr_head_hidden_dim),
        }
    active_window = is_stage2_active(iteration, pc.appearance_config)
    lambda_schedule = compute_appearance_schedule(
        iteration=iteration,
        enable_step=pc.appearance_config.enable_step,
        warmup_steps=pc.appearance_config.warmup_steps,
        schedule_type=pc.appearance_config.schedule_type,
    )
    lambda_stage2 = compute_stage2_lambda(iteration, pc.appearance_config)
    lambda_t = float(lambda_schedule * lambda_stage2)
    return {
        "enabled": bool(pc.appearance_residual_enabled and active_window and lambda_t > 0.0),
        "lambda_t": lambda_t,
        "enable_stage2": bool(pc.appearance_config.enable_stage2),
        "gate_floor": float(pc.appearance_config.gate_floor),
        "detach_xyz_grad": bool(pc.appearance_config.detach_xyz_grad),
        "detach_shape_grad": bool(pc.appearance_config.detach_shape_grad),
        "use_local_aniso_encoding": bool(pc.appearance_config.use_local_aniso_encoding),
        "use_decoupled_residual": bool(pc.appearance_config.use_decoupled_residual),
        "disable_global_gate": bool(pc.appearance_config.disable_global_gate),
        "spec_mask_temperature": float(pc.appearance_config.spec_mask_temperature),
        "use_two_layer_ddsr_heads": bool(pc.appearance_config.use_two_layer_ddsr_heads),
        "ddsr_head_hidden_dim": int(pc.appearance_config.ddsr_head_hidden_dim),
    }


def appearance_feature_dim(config):
    return int(config.latent_dim) + 4 + (5 if config.use_local_aniso_encoding else 0)


def appearance_base_feature_dim(config):
    return int(config.latent_dim) + 4


def _ddsr_activation(x, config):
    name = str(config.ddsr_head_activation).lower()
    if name == "relu":
        return torch.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    # SiLU is smooth and stable for tiny CUDA-fused heads.
    return torch.nn.functional.silu(x)


def _linear_or_two_layer(h, w1, b1, w2, b2, config):
    if config.use_two_layer_ddsr_heads and w2 is not None and b2 is not None:
        return torch.matmul(_ddsr_activation(torch.matmul(h, w1.t()) + b1, config), w2.t()) + b2
    return torch.matmul(h, w1.t()) + b1


def _quat_to_local_dir(rot, view_dir):
    q = rot / rot.norm(dim=1, keepdim=True).clamp_min(1e-6)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R matches the CUDA covariance convention; local direction is R^T * world direction.
    m00 = 1.0 - 2.0 * (y * y + z * z)
    m01 = 2.0 * (x * y - r * z)
    m02 = 2.0 * (x * z + r * y)
    m10 = 2.0 * (x * y + r * z)
    m11 = 1.0 - 2.0 * (x * x + z * z)
    m12 = 2.0 * (y * z - r * x)
    m20 = 2.0 * (x * z - r * y)
    m21 = 2.0 * (y * z + r * x)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    vx, vy, vz = view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]
    return torch.stack(
        [
            m00 * vx + m10 * vy + m20 * vz,
            m01 * vx + m11 * vy + m21 * vz,
            m02 * vx + m12 * vy + m22 * vz,
        ],
        dim=1,
    )


def build_appearance_features_python(pc, viewpoint_camera):
    xyz = pc.get_xyz.detach() if pc.appearance_config.detach_xyz_grad else pc.get_xyz
    view_vec = xyz - viewpoint_camera.camera_center.to(xyz.device).unsqueeze(0)
    view_dist = view_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
    view_dir = view_vec / view_dist
    parts = [pc.get_appearance_latent, view_dir, view_dist]
    aux = {}
    if pc.appearance_config.use_local_aniso_encoding:
        rot = pc.get_rotation.detach() if pc.appearance_config.detach_shape_grad else pc.get_rotation
        scales = pc.get_scaling.detach() if pc.appearance_config.detach_shape_grad else pc.get_scaling
        local_dir = _quat_to_local_dir(rot, view_dir)
        sorted_scales = torch.sort(scales.clamp_min(1e-6), dim=1).values
        min_scale = sorted_scales[:, 0:1]
        mid_scale = sorted_scales[:, 1:2]
        max_scale = sorted_scales[:, 2:3]
        aniso = (max_scale / min_scale).clamp(max=10.0)
        mid_min = (mid_scale / min_scale).clamp(max=10.0)
        parts.extend([local_dir, aniso, mid_min])
        aux["local_dir_abs_mean"] = local_dir.abs().mean().detach()
        aux["anisotropy_ratio_mean"] = aniso.mean().detach()
    return torch.cat(parts, dim=1), aux


def _compute_branch_outputs_python(pc, h):
    zero = h.new_tensor(0.0)
    base_dim = appearance_base_feature_dim(pc.appearance_config)
    latent_dim = int(pc.appearance_config.latent_dim)
    h_base = h[:, :base_dim]
    h_diff = torch.cat([h[:, :latent_dim], h[:, latent_dim + 3:latent_dim + 4]], dim=1)

    raw_gate = _linear_or_two_layer(
        h_base,
        pc.app_w_gate[:, :base_dim],
        pc.app_b_gate,
        getattr(pc, "app_w2_gate", None),
        getattr(pc, "app_b2_gate", None),
        pc.appearance_config,
    )
    gate_sigmoid = torch.sigmoid(raw_gate)
    gate = pc.appearance_config.gate_floor + (1.0 - pc.appearance_config.gate_floor) * gate_sigmoid
    if pc.appearance_config.disable_global_gate or not pc.appearance_config.enable_global_gate:
        gate = torch.ones_like(gate)

    if pc.appearance_config.residual_mode == "single" or not pc.appearance_config.use_decoupled_residual:
        delta = torch.tanh(torch.matmul(h, pc.app_w_rgb.t()) + pc.app_b_rgb)
        return {
            "delta": delta,
            "delta_diff": zero.expand_as(delta),
            "delta_spec": zero.expand_as(delta),
            "spec_mask": zero.expand((h.shape[0], 1)),
            "gate": gate,
            "mask_learned": False,
            "decoupled": False,
        }

    diff_w = torch.cat(
        [
            pc.app_w_diff[:, :latent_dim],
            pc.app_w_diff[:, latent_dim + 3:latent_dim + 4],
        ],
        dim=1,
    )
    diff_w2 = getattr(pc, "app_w2_diff", None)
    mask_w2 = getattr(pc, "app_w2_mask", None)
    delta_diff = torch.tanh(
        _linear_or_two_layer(
            h_diff,
            diff_w,
            pc.app_b_diff,
            diff_w2,
            getattr(pc, "app_b2_diff", None),
            pc.appearance_config,
        )
    )
    delta_spec = torch.tanh(
        _linear_or_two_layer(
            h,
            pc.app_w_spec,
            pc.app_b_spec,
            getattr(pc, "app_w2_spec", None),
            getattr(pc, "app_b2_spec", None),
            pc.appearance_config,
        )
    )
    if not pc.appearance_config.enable_diffuse_residual:
        delta_diff = torch.zeros_like(delta_diff)
    if not pc.appearance_config.enable_specular_residual:
        delta_spec = torch.zeros_like(delta_spec)

    if pc.appearance_config.residual_mode == "diffuse_only":
        spec_mask = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)
        mask_learned = False
    elif pc.appearance_config.residual_mode == "specular_only":
        spec_mask = torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype)
        mask_learned = False
    elif not pc.appearance_config.enable_specular_mask:
        spec_mask = torch.full((h.shape[0], 1), 0.5, device=h.device, dtype=h.dtype)
        mask_learned = False
    else:
        mask_logits = _linear_or_two_layer(
            h_base,
            pc.app_w_mask[:, :base_dim],
            pc.app_b_mask,
            mask_w2,
            getattr(pc, "app_b2_mask", None),
            pc.appearance_config,
        )
        spec_mask = torch.sigmoid(float(pc.appearance_config.spec_mask_temperature) * mask_logits)
        mask_learned = True
    delta = (1.0 - spec_mask) * delta_diff + spec_mask * delta_spec
    return {
        "delta": delta,
        "delta_diff": delta_diff,
        "delta_spec": delta_spec,
        "spec_mask": spec_mask,
        "gate": gate,
        "mask_learned": mask_learned,
        "decoupled": True,
    }


def compute_appearance_colors_precomp(pc, viewpoint_camera, base_rgb, appearance_state):
    if not appearance_state["enabled"]:
        return base_rgb
    h, _ = build_appearance_features_python(pc, viewpoint_camera)
    outputs = _compute_branch_outputs_python(pc, h)
    enhanced = base_rgb + float(appearance_state["lambda_t"]) * outputs["gate"] * outputs["delta"]
    return torch.clamp(enhanced, 0.0, 1.0)


def build_appearance_inputs_for_rasterizer(pc, appearance_state):
    if not appearance_state["enabled"]:
        return None
    appearance_inputs = {
        "appearance_latent": pc.get_appearance_latent,
        "app_w_rgb": pc.app_w_rgb,
        "app_b_rgb": pc.app_b_rgb,
        "app_w_gate": pc.app_w_gate,
        "app_b_gate": pc.app_b_gate,
        "app_w_diff": pc.app_w_diff,
        "app_b_diff": pc.app_b_diff,
        "app_w_spec": pc.app_w_spec,
        "app_b_spec": pc.app_b_spec,
        "app_w_mask": pc.app_w_mask,
        "app_b_mask": pc.app_b_mask,
        "app_w2_gate": pc.app_w2_gate,
        "app_b2_gate": pc.app_b2_gate,
        "app_w2_diff": pc.app_w2_diff,
        "app_b2_diff": pc.app_b2_diff,
        "app_w2_spec": pc.app_w2_spec,
        "app_b2_spec": pc.app_b2_spec,
        "app_w2_mask": pc.app_w2_mask,
        "app_b2_mask": pc.app_b2_mask,
        "latent_dim": int(pc.appearance_config.latent_dim),
        "lambda_t": float(appearance_state["lambda_t"]),
        "gate_floor": float(appearance_state["gate_floor"]),
        "spec_mask_temperature": float(appearance_state["spec_mask_temperature"]),
        "enabled": True,
        "detach_xyz_grad": bool(appearance_state["detach_xyz_grad"]),
        "detach_shape_grad": bool(appearance_state["detach_shape_grad"]),
        "use_local_aniso_encoding": bool(appearance_state["use_local_aniso_encoding"]),
        "use_decoupled_residual": bool(appearance_state["use_decoupled_residual"]),
        "disable_global_gate": bool(appearance_state["disable_global_gate"]),
        "use_two_layer_ddsr_heads": bool(appearance_state["use_two_layer_ddsr_heads"]),
        "ddsr_head_hidden_dim": int(appearance_state["ddsr_head_hidden_dim"]),
    }
    if pc.appearance_config.residual_mode == "single" or not pc.appearance_config.use_decoupled_residual:
        appearance_inputs["use_decoupled_residual"] = False
        return appearance_inputs

    if not pc.appearance_config.enable_global_gate:
        appearance_inputs["disable_global_gate"] = True

    if not pc.appearance_config.enable_diffuse_residual and pc.app_w_diff is not None:
        appearance_inputs["app_w_diff"] = torch.zeros_like(pc.app_w_diff)
        appearance_inputs["app_b_diff"] = torch.zeros_like(pc.app_b_diff)
        if pc.app_w2_diff is not None:
            appearance_inputs["app_w2_diff"] = torch.zeros_like(pc.app_w2_diff)
            appearance_inputs["app_b2_diff"] = torch.zeros_like(pc.app_b2_diff)
    if not pc.appearance_config.enable_specular_residual and pc.app_w_spec is not None:
        appearance_inputs["app_w_spec"] = torch.zeros_like(pc.app_w_spec)
        appearance_inputs["app_b_spec"] = torch.zeros_like(pc.app_b_spec)
        if pc.app_w2_spec is not None:
            appearance_inputs["app_w2_spec"] = torch.zeros_like(pc.app_w2_spec)
            appearance_inputs["app_b2_spec"] = torch.zeros_like(pc.app_b2_spec)

    if pc.app_w_mask is not None:
        if pc.appearance_config.residual_mode == "diffuse_only":
            appearance_inputs["app_w_mask"] = torch.zeros_like(pc.app_w_mask)
            appearance_inputs["app_b_mask"] = torch.full_like(pc.app_b_mask, -12.0)
            if pc.app_w2_mask is not None:
                appearance_inputs["app_w2_mask"] = torch.zeros_like(pc.app_w2_mask)
                appearance_inputs["app_b2_mask"] = torch.full_like(pc.app_b2_mask, -12.0)
        elif pc.appearance_config.residual_mode == "specular_only":
            appearance_inputs["app_w_mask"] = torch.zeros_like(pc.app_w_mask)
            appearance_inputs["app_b_mask"] = torch.full_like(pc.app_b_mask, 12.0)
            if pc.app_w2_mask is not None:
                appearance_inputs["app_w2_mask"] = torch.zeros_like(pc.app_w2_mask)
                appearance_inputs["app_b2_mask"] = torch.full_like(pc.app_b2_mask, 12.0)
        elif not pc.appearance_config.enable_specular_mask:
            appearance_inputs["app_w_mask"] = torch.zeros_like(pc.app_w_mask)
            appearance_inputs["app_b_mask"] = torch.zeros_like(pc.app_b_mask)
            if pc.app_w2_mask is not None:
                appearance_inputs["app_w2_mask"] = torch.zeros_like(pc.app_w2_mask)
                appearance_inputs["app_b2_mask"] = torch.zeros_like(pc.app_b2_mask)
    return appearance_inputs


def compute_appearance_regularizers(pc, viewpoint_camera, appearance_state):
    zero = pc.get_xyz.new_tensor(0.0)
    stats = {
        "residual_reg": zero,
        "gate_reg": zero,
        "smooth_reg": zero,
        "delta_mean": zero,
        "delta_abs_mean": zero,
        "delta_max_abs": zero,
        "gate_mean": zero,
        "gate_std": zero,
        "gate_max": zero,
        "local_dir_abs_mean": zero,
        "anisotropy_ratio_mean": zero,
        "delta_diff_abs_mean": zero,
        "delta_spec_abs_mean": zero,
        "spec_mask_mean": zero,
        "spec_mask_std": zero,
        "spec_mask_max": zero,
        "spec_mask_temperature": zero,
        "lambda_t": zero,
        "diff_consistency_reg": zero,
        "branch_diversity_reg": zero,
        "opacity_delta_abs_mean": zero,
        "opacity_delta_max": zero,
        "scale_delta_abs_mean": zero,
        "scale_delta_max": zero,
        "rotation_delta_abs_mean": zero,
        "rotation_delta_max": zero,
        "opacity_grad_norm": zero,
        "scale_grad_norm": zero,
        "rotation_grad_norm": zero,
        "opacity_update_norm": zero,
        "scale_update_norm": zero,
        "rotation_update_norm": zero,
        "stage2_anchor_reg": zero,
        "stage2_substage": "none",
    }
    if not appearance_state["enabled"]:
        stats["lambda_t"] = zero.new_tensor(float(appearance_state["lambda_t"]))
        if hasattr(pc, "get_stage2_refine_stats"):
            for key, value in pc.get_stage2_refine_stats().items():
                if isinstance(value, torch.Tensor):
                    stats[key] = value.detach()
                elif isinstance(value, str):
                    stats[key] = value
                else:
                    stats[key] = zero.new_tensor(float(value))
        return zero, stats

    h, aux_stats = build_appearance_features_python(pc, viewpoint_camera)
    outputs = _compute_branch_outputs_python(pc, h)
    delta = outputs["delta"]
    delta_diff = outputs["delta_diff"]
    delta_spec = outputs["delta_spec"]
    spec_mask = outputs["spec_mask"]
    gate = outputs["gate"]
    if outputs["decoupled"] and outputs["mask_learned"]:
        spec_mask_reg = spec_mask.mean()
        entropy = -(spec_mask * torch.log(spec_mask.clamp_min(1e-6)) + (1.0 - spec_mask) * torch.log((1.0 - spec_mask).clamp_min(1e-6))).mean()
    else:
        spec_mask_reg = zero
        entropy = zero
    if outputs["decoupled"] and pc.appearance_config.enable_diffuse_residual and pc.appearance_config.lambda_diff_consistency > 0.0:
        latent_dim = int(pc.appearance_config.latent_dim)
        h_diff = torch.cat([h[:, :latent_dim], h[:, latent_dim + 3:latent_dim + 4]], dim=1)
        diff_w = torch.cat(
            [
                pc.app_w_diff[:, :latent_dim],
                pc.app_w_diff[:, latent_dim + 3:latent_dim + 4],
            ],
            dim=1,
        )
        diff_w2 = getattr(pc, "app_w2_diff", None)
        h_diff_perturbed = h_diff.clone()
        h_diff_perturbed[:, -1:] = h_diff_perturbed[:, -1:] * 1.01
        delta_diff_perturbed = torch.tanh(
            _linear_or_two_layer(
                h_diff_perturbed,
                diff_w,
                pc.app_b_diff,
                diff_w2,
                getattr(pc, "app_b2_diff", None),
                pc.appearance_config,
            )
        )
        diff_consistency = (delta_diff - delta_diff_perturbed).abs().mean()
    else:
        diff_consistency = zero
    if outputs["decoupled"] and pc.appearance_config.enable_diffuse_residual and pc.appearance_config.enable_specular_residual:
        branch_diversity = (delta_spec - delta_diff).abs().mean()
    else:
        branch_diversity = zero

    residual_reg = delta.pow(2).mean()
    gate_reg = gate.mean()
    anchor_reg = pc.compute_stage2_anchor_regularization() if hasattr(pc, "compute_stage2_anchor_regularization") else zero
    total_reg = (
        pc.appearance_config.lambda_residual_reg * residual_reg
        + pc.appearance_config.lambda_gate_reg * gate_reg
        + pc.appearance_config.lambda_spec_mask_reg * spec_mask_reg
        + pc.appearance_config.spec_mask_entropy_reg * entropy
        + pc.appearance_config.lambda_diff_consistency * diff_consistency
        - pc.appearance_config.lambda_branch_diversity * branch_diversity
        + anchor_reg
    )
    refine_stats = pc.get_stage2_refine_stats() if hasattr(pc, "get_stage2_refine_stats") else {}
    stats.update(
        {
            "residual_reg": residual_reg.detach(),
            "gate_reg": gate_reg.detach(),
            "delta_mean": delta.mean().detach(),
            "delta_abs_mean": delta.abs().mean().detach(),
            "delta_max_abs": delta.abs().max().detach(),
            "gate_mean": gate.mean().detach(),
            "gate_std": gate.std(unbiased=False).detach(),
            "gate_max": gate.max().detach(),
            "delta_diff_abs_mean": delta_diff.abs().mean().detach(),
            "delta_spec_abs_mean": delta_spec.abs().mean().detach(),
            "spec_mask_mean": spec_mask.mean().detach(),
            "spec_mask_std": spec_mask.std(unbiased=False).detach(),
            "spec_mask_max": spec_mask.max().detach(),
            "spec_mask_temperature": zero.new_tensor(float(pc.appearance_config.spec_mask_temperature)),
            "diff_consistency_reg": diff_consistency.detach(),
            "branch_diversity_reg": branch_diversity.detach(),
            "stage2_anchor_reg": anchor_reg.detach(),
            "lambda_t": zero.new_tensor(float(appearance_state["lambda_t"])),
        }
    )
    for key, value in refine_stats.items():
        if isinstance(value, torch.Tensor):
            stats[key] = value.detach()
        elif isinstance(value, str):
            stats[key] = value
        else:
            stats[key] = zero.new_tensor(float(value))
    for key, value in aux_stats.items():
        stats[key] = value
    return total_reg, stats



