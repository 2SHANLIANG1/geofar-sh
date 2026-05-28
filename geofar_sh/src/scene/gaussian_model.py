#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.import_utils import ensure_local_diff_gaussian_rasterization
ensure_local_diff_gaussian_rasterization()
from scene.appearance_residual import (
    AppearanceResidualConfig,
    build_appearance_forward_state,
    appearance_head_parameter_norm,
    build_appearance_config,
    build_appearance_head,
    build_appearance_meta_dict,
    appearance_feature_dim,
    resolve_appearance_mode,
)
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.appearance_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.appearance_config = AppearanceResidualConfig()
        self.appearance_residual_head = None
        self.appearance_residual_enabled = False
        self.appearance_training_active = False
        self.appearance_residual_load_status = "baseline_mode"
        self.appearance_residual_load_warning = ""
        self.appearance_residual_loaded = False
        self._appearance_latent = torch.empty(0)
        self.app_w_rgb = None
        self.app_b_rgb = None
        self.app_w_gate = None
        self.app_b_gate = None
        self.app_w_diff = None
        self.app_b_diff = None
        self.app_w_spec = None
        self.app_b_spec = None
        self.app_w_mask = None
        self.app_b_mask = None
        self.app_w2_gate = None
        self.app_b2_gate = None
        self.app_w2_diff = None
        self.app_b2_diff = None
        self.app_w2_spec = None
        self.app_b2_spec = None
        self.app_w2_mask = None
        self.app_b2_mask = None
        self._stage2_ref_xyz = None
        self._stage2_ref_features_dc = None
        self._stage2_ref_features_rest = None
        self._stage2_ref_opacity = None
        self._stage2_ref_scaling = None
        self._stage2_ref_rotation = None
        self._last_xyz_grad_norm = torch.tensor(0.0)
        self._last_f_dc_grad_norm = torch.tensor(0.0)
        self._last_f_rest_grad_norm = torch.tensor(0.0)
        self._last_xyz_update_norm = torch.tensor(0.0)
        self._last_f_dc_update_norm = torch.tensor(0.0)
        self._last_f_rest_update_norm = torch.tensor(0.0)
        self._last_opacity_update_norm = torch.tensor(0.0)
        self._last_scale_update_norm = torch.tensor(0.0)
        self._last_rotation_update_norm = torch.tensor(0.0)
        self._last_opacity_grad_norm = torch.tensor(0.0)
        self._last_scale_grad_norm = torch.tensor(0.0)
        self._last_rotation_grad_norm = torch.tensor(0.0)
        self._stage2_substage = "none"
        self.densify_event_count = 0
        self.prune_event_count = 0
        self.last_densify_added = 0
        self.last_prune_removed = 0
        self.setup_functions()

    def _set_empty_appearance_latent(self, num_points, device=None, dtype=None):
        if device is None:
            device = self._xyz.device
        if dtype is None:
            dtype = self._xyz.dtype
        self._appearance_latent = nn.Parameter(
            torch.zeros((int(num_points), 0), device=device, dtype=dtype).requires_grad_(False)
        )

    def _init_appearance_parameters(self, device=None, dtype=None):
        if device is None:
            device = self._xyz.device
        if dtype is None:
            dtype = self._xyz.dtype
        latent_dim = self.appearance_config.latent_dim if self.appearance_residual_enabled else 0
        feat_dim = appearance_feature_dim(self.appearance_config)
        use_two_layer = bool(self.appearance_config.use_two_layer_ddsr_heads)
        hidden_dim = int(max(self.appearance_config.ddsr_head_hidden_dim, 1))
        self._appearance_latent = nn.Parameter(
            torch.zeros((self._xyz.shape[0], latent_dim), device=device, dtype=dtype).requires_grad_(self.appearance_residual_enabled)
        )
        self.app_w_rgb = nn.Parameter((0.01 * torch.randn((3, feat_dim), device=device, dtype=dtype)).requires_grad_(True))
        self.app_b_rgb = nn.Parameter(torch.zeros((3,), device=device, dtype=dtype).requires_grad_(True))
        gate_out = hidden_dim if use_two_layer else 1
        diff_out = hidden_dim if use_two_layer else 3
        mask_out = hidden_dim if use_two_layer else 1
        self.app_w_gate = nn.Parameter((0.01 * torch.randn((gate_out, feat_dim), device=device, dtype=dtype)).requires_grad_(True))
        self.app_b_gate = nn.Parameter(torch.zeros((gate_out,), device=device, dtype=dtype).requires_grad_(True))
        self.app_w_diff = nn.Parameter((0.01 * torch.randn((diff_out, feat_dim), device=device, dtype=dtype)).requires_grad_(True))
        self.app_b_diff = nn.Parameter(torch.zeros((diff_out,), device=device, dtype=dtype).requires_grad_(True))
        self.app_w_spec = nn.Parameter((0.01 * torch.randn((diff_out, feat_dim), device=device, dtype=dtype)).requires_grad_(True))
        self.app_b_spec = nn.Parameter(torch.zeros((diff_out,), device=device, dtype=dtype).requires_grad_(True))
        self.app_w_mask = nn.Parameter((0.01 * torch.randn((mask_out, feat_dim), device=device, dtype=dtype)).requires_grad_(True))
        self.app_b_mask = nn.Parameter(torch.zeros((mask_out,), device=device, dtype=dtype).requires_grad_(True))
        if use_two_layer:
            self.app_w2_gate = nn.Parameter((0.01 * torch.randn((1, hidden_dim), device=device, dtype=dtype)).requires_grad_(True))
            self.app_b2_gate = nn.Parameter(torch.zeros((1,), device=device, dtype=dtype).requires_grad_(True))
            self.app_w2_diff = nn.Parameter((0.01 * torch.randn((3, hidden_dim), device=device, dtype=dtype)).requires_grad_(True))
            self.app_b2_diff = nn.Parameter(torch.zeros((3,), device=device, dtype=dtype).requires_grad_(True))
            self.app_w2_spec = nn.Parameter((0.01 * torch.randn((3, hidden_dim), device=device, dtype=dtype)).requires_grad_(True))
            self.app_b2_spec = nn.Parameter(torch.zeros((3,), device=device, dtype=dtype).requires_grad_(True))
            self.app_w2_mask = nn.Parameter((0.01 * torch.randn((1, hidden_dim), device=device, dtype=dtype)).requires_grad_(True))
            self.app_b2_mask = nn.Parameter(torch.zeros((1,), device=device, dtype=dtype).requires_grad_(True))
        else:
            self.app_w2_gate = self.app_b2_gate = None
            self.app_w2_diff = self.app_b2_diff = None
            self.app_w2_spec = self.app_b2_spec = None
            self.app_w2_mask = self.app_b2_mask = None

    def _build_appearance_optimizer(self, training_args):
        if not self.appearance_training_active or self.app_w_rgb is None:
            self.appearance_optimizer = None
            return
        param_groups = []
        param_groups.append({"params": [self.app_w_rgb, self.app_b_rgb], "lr": training_args.lr_fastkan, "name": "appearance_main"})
        param_groups.append({"params": [self.app_w_gate, self.app_b_gate], "lr": training_args.lr_fastkan_gate, "name": "appearance_gate"})
        if self.appearance_config.use_decoupled_residual:
            param_groups.append({"params": [self.app_w_diff, self.app_b_diff], "lr": training_args.lr_fastkan, "name": "appearance_diff"})
            param_groups.append({"params": [self.app_w_spec, self.app_b_spec], "lr": training_args.lr_fastkan, "name": "appearance_spec"})
            param_groups.append({"params": [self.app_w_mask, self.app_b_mask], "lr": training_args.lr_fastkan_gate, "name": "appearance_mask"})
            if self.appearance_config.use_two_layer_ddsr_heads and self.app_w2_gate is not None:
                param_groups.append({"params": [self.app_w2_gate, self.app_b2_gate], "lr": training_args.lr_fastkan_gate, "name": "appearance_gate_2"})
                param_groups.append({"params": [self.app_w2_diff, self.app_b2_diff], "lr": training_args.lr_fastkan, "name": "appearance_diff_2"})
                param_groups.append({"params": [self.app_w2_spec, self.app_b2_spec], "lr": training_args.lr_fastkan, "name": "appearance_spec_2"})
                param_groups.append({"params": [self.app_w2_mask, self.app_b2_mask], "lr": training_args.lr_fastkan_gate, "name": "appearance_mask_2"})
        self.appearance_optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15) if param_groups else None

    def get_appearance_activation_iter(self):
        if not self.appearance_residual_enabled:
            return None
        if self.appearance_config.enable_stage2:
            return int(max(self.appearance_config.enable_step, self.appearance_config.stage2_start))
        return int(self.appearance_config.enable_step)

    def activate_appearance_training(self, training_args):
        if not self.appearance_residual_enabled or self.appearance_training_active:
            return False
        device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
        dtype = self._xyz.dtype if torch.is_tensor(self._xyz) else torch.float32
        self._init_appearance_parameters(device=device, dtype=dtype)
        if self.optimizer is not None and self._appearance_latent.numel() > 0:
            has_latent_group = any(group["name"] == "appearance_latent" for group in self.optimizer.param_groups)
            if not has_latent_group:
                self.optimizer.add_param_group(
                    {"params": [self._appearance_latent], "lr": training_args.lr_appearance_latent, "name": "appearance_latent"}
                )
        self.appearance_training_active = True
        self._build_appearance_optimizer(training_args)
        return True

    def _appearance_dims_match(self):
        expected_latent_dim = int(self.appearance_config.latent_dim) if self.appearance_residual_enabled else 0
        expected_feat_dim = appearance_feature_dim(self.appearance_config)
        use_two_layer = bool(self.appearance_config.use_two_layer_ddsr_heads)
        expected_hidden = int(min(max(self.appearance_config.ddsr_head_hidden_dim, 1), 12))
        loaded_latent_dim = int(self._appearance_latent.shape[1]) if torch.is_tensor(self._appearance_latent) and self._appearance_latent.ndim == 2 else 0
        tensors = [self.app_w_rgb, self.app_w_gate]
        if self.appearance_config.use_decoupled_residual:
            tensors.extend([self.app_w_diff, self.app_w_spec, self.app_w_mask])
        loaded_feat_dims = [
            int(t.shape[1]) if t is not None and torch.is_tensor(t) and t.ndim == 2 else -1
            for t in tensors
        ]
        first_layers_ok = loaded_latent_dim == expected_latent_dim and all(dim == expected_feat_dim for dim in loaded_feat_dims)
        if not use_two_layer:
            return first_layers_ok
        second_layers = [
            (self.app_w2_gate, 1),
            (self.app_w2_diff, 3),
            (self.app_w2_spec, 3),
            (self.app_w2_mask, 1),
        ]
        second_layers_ok = all(
            t is not None and torch.is_tensor(t) and t.ndim == 2 and int(t.shape[0]) == out_dim and int(t.shape[1]) == expected_hidden
            for t, out_dim in second_layers
        )
        return first_layers_ok and second_layers_ok

    def capture(self):
        if not self.appearance_training_active:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                None,
            )
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.appearance_optimizer.state_dict() if self.appearance_optimizer is not None else None,
            self._appearance_latent,
            self.app_w_rgb,
            self.app_b_rgb,
            self.app_w_gate,
            self.app_b_gate,
            self.app_w_diff,
            self.app_b_diff,
            self.app_w_spec,
            self.app_b_spec,
            self.app_w_mask,
            self.app_b_mask,
            self.app_w2_gate,
            self.app_b2_gate,
            self.app_w2_diff,
            self.app_b2_diff,
            self.app_w2_spec,
            self.app_b2_spec,
            self.app_w2_mask,
            self.app_b2_mask,
        )
    
    def restore(self, model_args, training_args):
        expected_latent_dim = int(self.appearance_config.latent_dim) if self.appearance_residual_enabled else 0
        expected_feat_dim = appearance_feature_dim(self.appearance_config)
        appearance_state_compatible = True

        if len(model_args) == 12:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale) = model_args
            appearance_opt_dict = None
            self.appearance_training_active = False
            self._set_empty_appearance_latent(self._xyz.shape[0], device=self._xyz.device, dtype=self._xyz.dtype)
            self.app_w_rgb = self.app_b_rgb = None
            self.app_w_gate = self.app_b_gate = None
            self.app_w_diff = self.app_b_diff = None
            self.app_w_spec = self.app_b_spec = None
            self.app_w_mask = self.app_b_mask = None
            self.app_w2_gate = self.app_b2_gate = None
            self.app_w2_diff = self.app_b2_diff = None
            self.app_w2_spec = self.app_b2_spec = None
            self.app_w2_mask = self.app_b2_mask = None
        elif len(model_args) == 13:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            appearance_opt_dict) = model_args
            self.appearance_training_active = False
            self._set_empty_appearance_latent(self._xyz.shape[0], device=self._xyz.device, dtype=self._xyz.dtype)
            self.app_w_rgb = self.app_b_rgb = None
            self.app_w_gate = self.app_b_gate = None
            self.app_w_diff = self.app_b_diff = None
            self.app_w_spec = self.app_b_spec = None
            self.app_w_mask = self.app_b_mask = None
            self.app_w2_gate = self.app_b2_gate = None
            self.app_w2_diff = self.app_b2_diff = None
            self.app_w2_spec = self.app_b2_spec = None
            self.app_w2_mask = self.app_b2_mask = None
            appearance_state_compatible = False
        elif len(model_args) == 18:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            appearance_opt_dict,
            self._appearance_latent,
            self.app_w_rgb,
            self.app_b_rgb,
            self.app_w_gate,
            self.app_b_gate) = model_args
            self.appearance_training_active = self.appearance_residual_enabled
            self.app_w_diff = None
            self.app_b_diff = None
            self.app_w_spec = None
            self.app_b_spec = None
            self.app_w_mask = None
            self.app_b_mask = None
        elif len(model_args) == 24:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            appearance_opt_dict,
            self._appearance_latent,
            self.app_w_rgb,
            self.app_b_rgb,
            self.app_w_gate,
            self.app_b_gate,
            self.app_w_diff,
            self.app_b_diff,
            self.app_w_spec,
            self.app_b_spec,
            self.app_w_mask,
            self.app_b_mask) = model_args
            self.appearance_training_active = self.appearance_residual_enabled
            self.app_w2_gate = None
            self.app_b2_gate = None
            self.app_w2_diff = None
            self.app_b2_diff = None
            self.app_w2_spec = None
            self.app_b2_spec = None
            self.app_w2_mask = None
            self.app_b2_mask = None
        else:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            appearance_opt_dict,
            self._appearance_latent,
            self.app_w_rgb,
            self.app_b_rgb,
            self.app_w_gate,
            self.app_b_gate,
            self.app_w_diff,
            self.app_b_diff,
            self.app_w_spec,
            self.app_b_spec,
            self.app_w_mask,
            self.app_b_mask,
            self.app_w2_gate,
            self.app_b2_gate,
            self.app_w2_diff,
            self.app_b2_diff,
            self.app_w2_spec,
            self.app_b2_spec,
            self.app_w2_mask,
            self.app_b2_mask) = model_args
            self.appearance_training_active = self.appearance_residual_enabled

        if len(model_args) >= 18:
            self._appearance_latent = nn.Parameter(self._appearance_latent.requires_grad_(True))
            self.app_w_rgb = nn.Parameter(self.app_w_rgb.requires_grad_(True))
            self.app_b_rgb = nn.Parameter(self.app_b_rgb.requires_grad_(True))
            self.app_w_gate = nn.Parameter(self.app_w_gate.requires_grad_(True))
            self.app_b_gate = nn.Parameter(self.app_b_gate.requires_grad_(True))
            if self.app_w_diff is not None:
                self.app_w_diff = nn.Parameter(self.app_w_diff.requires_grad_(True))
                self.app_b_diff = nn.Parameter(self.app_b_diff.requires_grad_(True))
                self.app_w_spec = nn.Parameter(self.app_w_spec.requires_grad_(True))
                self.app_b_spec = nn.Parameter(self.app_b_spec.requires_grad_(True))
                self.app_w_mask = nn.Parameter(self.app_w_mask.requires_grad_(True))
                self.app_b_mask = nn.Parameter(self.app_b_mask.requires_grad_(True))
            if self.app_w2_gate is not None:
                self.app_w2_gate = nn.Parameter(self.app_w2_gate.requires_grad_(True))
                self.app_b2_gate = nn.Parameter(self.app_b2_gate.requires_grad_(True))
                self.app_w2_diff = nn.Parameter(self.app_w2_diff.requires_grad_(True))
                self.app_b2_diff = nn.Parameter(self.app_b2_diff.requires_grad_(True))
                self.app_w2_spec = nn.Parameter(self.app_w2_spec.requires_grad_(True))
                self.app_b2_spec = nn.Parameter(self.app_b2_spec.requires_grad_(True))
                self.app_w2_mask = nn.Parameter(self.app_w2_mask.requires_grad_(True))
                self.app_b2_mask = nn.Parameter(self.app_b2_mask.requires_grad_(True))

            if not self._appearance_dims_match():
                loaded_latent_dim = int(self._appearance_latent.shape[1]) if self._appearance_latent.ndim == 2 else 0
                loaded_feat_dim = int(self.app_w_rgb.shape[1]) if self.app_w_rgb is not None and self.app_w_rgb.ndim == 2 else 0
                print(
                    f"[WARN] Appearance checkpoint dimensions ({loaded_latent_dim}, {loaded_feat_dim}) "
                    f"do not match current config ({expected_latent_dim}, {expected_feat_dim}); "
                    "reinitializing appearance parameters for compatibility."
                )
                self._init_appearance_parameters(device=self._xyz.device, dtype=self._xyz.dtype)
                appearance_state_compatible = False

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError as exc:
            print(f"[WARN] Optimizer state is incompatible with current appearance parameter groups; reinitializing optimizer state. Details: {exc}")
        if self.appearance_optimizer is not None and appearance_opt_dict is not None and appearance_state_compatible:
            try:
                self.appearance_optimizer.load_state_dict(appearance_opt_dict)
            except ValueError as exc:
                print(f"[WARN] Appearance optimizer state is incompatible with current configuration; reinitializing appearance optimizer. Details: {exc}")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_appearance_latent(self):
        return self._appearance_latent
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def configure_appearance_residual(self, args):
        self.appearance_config = build_appearance_config(args)
        self.appearance_residual_enabled = resolve_appearance_mode(args) != "none"
        self.appearance_training_active = False
        self.appearance_residual_head = build_appearance_head(self.appearance_config)

    def has_appearance_residual_loaded(self):
        return bool(self.appearance_residual_loaded)

    def mark_appearance_residual_loaded(self, loaded=True):
        self.appearance_residual_loaded = bool(loaded)

    def get_appearance_meta(self):
        return build_appearance_meta_dict(self.appearance_config)

    def get_appearance_forward_state(self, iteration=None):
        return build_appearance_forward_state(self, iteration)

    def get_appearance_parameter_norm(self):
        return appearance_head_parameter_norm(self)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._set_empty_appearance_latent(fused_point_cloud.shape[0], device=fused_point_cloud.device, dtype=fused_point_cloud.dtype)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.appearance_training_active and self._appearance_latent.numel() > 0:
            l.append({'params': [self._appearance_latent], 'lr': training_args.lr_appearance_latent, "name": "appearance_latent"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self._build_appearance_optimizer(training_args)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def has_stage2_refinement(self):
        cfg = self.appearance_config
        return bool(
            self.appearance_residual_enabled
            or cfg.stage2_refine_sh
            or cfg.stage2_joint_refine
            or cfg.stage2_refine_xyz
            or cfg.freeze_exposure_in_stage2
        )

    def is_stage2_window_active(self, iteration):
        cfg = self.appearance_config
        if not getattr(cfg, "enable_stage2", True):
            return False
        if not self.has_stage2_refinement():
            return False
        if iteration is None:
            return True
        if iteration < cfg.stage2_start:
            return False
        if cfg.stage2_iters <= 0:
            return True
        return iteration < (cfg.stage2_start + cfg.stage2_iters)

    def apply_stage2_trainability(self, iteration):
        cfg = self.appearance_config
        freeze_window = cfg.freeze_geometry_in_stage2 and self.is_stage2_window_active(iteration)
        sh_trainable = bool(cfg.stage2_refine_xyz or cfg.stage2_refine_sh or cfg.stage2_joint_refine)

        self._xyz.requires_grad_(not freeze_window or bool(cfg.stage2_refine_xyz))
        self._features_dc.requires_grad_(not freeze_window or sh_trainable)
        self._features_rest.requires_grad_(not freeze_window or sh_trainable)
        self._opacity.requires_grad_(not freeze_window or bool(cfg.stage2_refine_opacity))
        self._scaling.requires_grad_(not freeze_window or bool(cfg.stage2_refine_scale))
        self._rotation.requires_grad_(not freeze_window or bool(cfg.stage2_refine_rotation))

        if hasattr(self, "_exposure") and self._exposure is not None:
            self._exposure.requires_grad_(not (freeze_window and bool(cfg.freeze_exposure_in_stage2)))

        appearance_trainable = bool(
            self.appearance_training_active
            and self.appearance_residual_enabled
            and self.is_stage2_window_active(iteration)
        )
        if torch.is_tensor(self._appearance_latent):
            self._appearance_latent.requires_grad_(appearance_trainable and self._appearance_latent.numel() > 0)
        for tensor in (
            self.app_w_rgb, self.app_b_rgb, self.app_w_gate, self.app_b_gate,
            self.app_w_diff, self.app_b_diff, self.app_w_spec, self.app_b_spec,
            self.app_w_mask, self.app_b_mask, self.app_w2_gate, self.app_b2_gate,
            self.app_w2_diff, self.app_b2_diff, self.app_w2_spec, self.app_b2_spec,
            self.app_w2_mask, self.app_b2_mask,
        ):
            if tensor is not None:
                tensor.requires_grad_(appearance_trainable)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        exposure_frozen = (
            self.appearance_config.freeze_geometry_in_stage2
            and self.appearance_config.freeze_exposure_in_stage2
            and self.is_stage2_window_active(iteration)
        )
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = 0.0 if exposure_frozen else self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                xyz_lr = lr
            elif param_group["name"] == "appearance_latent":
                param_group["lr"] = self.appearance_config.lr_latent

        stage2_enabled = self.is_stage2_window_active(iteration)
        if (
            self.appearance_config.enable_stage2
            and self.appearance_config.freeze_geometry_in_stage2
            and stage2_enabled
        ):
            geom_refine_active = self.is_stage2_geom_refine_active(iteration)
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    param_group["lr"] = self.appearance_config.stage2_lr_xyz if geom_refine_active and self.appearance_config.stage2_refine_xyz else 0.0
                elif param_group["name"] in {"f_dc", "f_rest"}:
                    if geom_refine_active and (self.appearance_config.stage2_refine_xyz or self.appearance_config.stage2_refine_sh or self.appearance_config.stage2_joint_refine):
                        param_group["lr"] = self.appearance_config.stage2_lr_f_dc if param_group["name"] == "f_dc" else self.appearance_config.stage2_lr_f_rest
                    else:
                        param_group["lr"] = 0.0
                elif param_group["name"] == "opacity":
                    param_group["lr"] = self.appearance_config.stage2_lr_opacity if geom_refine_active and self.appearance_config.stage2_refine_opacity else 0.0
                elif param_group["name"] == "scaling":
                    param_group["lr"] = self.appearance_config.stage2_lr_scale if geom_refine_active and self.appearance_config.stage2_refine_scale else 0.0
                elif param_group["name"] == "rotation":
                    param_group["lr"] = self.appearance_config.stage2_lr_rotation if geom_refine_active and self.appearance_config.stage2_refine_rotation else 0.0

        if self.appearance_optimizer is not None:
            for param_group in self.appearance_optimizer.param_groups:
                if param_group["name"] in {"appearance_main", "appearance_diff", "appearance_spec", "appearance_diff_2", "appearance_spec_2"}:
                    param_group["lr"] = self.appearance_config.lr_main
                elif param_group["name"] in {"appearance_gate", "appearance_mask", "appearance_gate_2", "appearance_mask_2"}:
                    param_group["lr"] = self.appearance_config.lr_gate
        return xyz_lr

    def get_stage2_relative_iter(self, iteration):
        return int(iteration) - int(self.appearance_config.stage2_start) + 1

    def is_stage2_geom_refine_active(self, iteration):
        if not self.is_stage2_window_active(iteration):
            return False
        return self.get_stage2_relative_iter(iteration) > int(self.appearance_config.stage2_geom_unfreeze_iter)

    def get_stage2_substage(self, iteration=None):
        if not self.appearance_config.enable_stage2:
            return "disabled"
        if iteration is None:
            return self._stage2_substage
        if (self.appearance_config.stage2_joint_refine or self.appearance_config.stage2_refine_xyz) and self.is_stage2_geom_refine_active(iteration):
            return "joint-refine"
        return "appearance-only"

    def update_stage2_substage(self, iteration):
        if not self.get_appearance_forward_state(iteration)["enabled"]:
            self._stage2_substage = "none"
        else:
            self._stage2_substage = self.get_stage2_substage(iteration)
        return self._stage2_substage

    def ensure_stage2_refine_reference(self):
        if self._stage2_ref_xyz is None:
            self._stage2_ref_xyz = self._xyz.detach().clone()
        if self._stage2_ref_features_dc is None:
            self._stage2_ref_features_dc = self._features_dc.detach().clone()
        if self._stage2_ref_features_rest is None:
            self._stage2_ref_features_rest = self._features_rest.detach().clone()
        if self._stage2_ref_opacity is None:
            self._stage2_ref_opacity = self._opacity.detach().clone()
        if self._stage2_ref_scaling is None:
            self._stage2_ref_scaling = self._scaling.detach().clone()
        if self._stage2_ref_rotation is None:
            self._stage2_ref_rotation = self._rotation.detach().clone()

    def update_stage2_geometry_grad_stats(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
        zero = torch.tensor(0.0, device=device)
        self._last_xyz_grad_norm = self._xyz.grad.detach().norm() if self._xyz.grad is not None else zero
        self._last_f_dc_grad_norm = self._features_dc.grad.detach().norm() if self._features_dc.grad is not None else zero
        self._last_f_rest_grad_norm = self._features_rest.grad.detach().norm() if self._features_rest.grad is not None else zero
        self._last_opacity_grad_norm = self._opacity.grad.detach().norm() if self._opacity.grad is not None else zero
        self._last_scale_grad_norm = self._scaling.grad.detach().norm() if self._scaling.grad is not None else zero
        self._last_rotation_grad_norm = self._rotation.grad.detach().norm() if self._rotation.grad is not None else zero

    def clip_stage2_geometry_gradients(self, iteration):
        if not self.is_stage2_geom_refine_active(iteration):
            return
        params = []
        if self.appearance_config.stage2_refine_xyz and self._xyz.grad is not None:
            params.append(self._xyz)
        if (self.appearance_config.stage2_refine_xyz or self.appearance_config.stage2_refine_sh) and self._features_dc.grad is not None:
            params.append(self._features_dc)
        if (self.appearance_config.stage2_refine_xyz or self.appearance_config.stage2_refine_sh) and self._features_rest.grad is not None:
            params.append(self._features_rest)
        if self.appearance_config.stage2_refine_opacity and self._opacity.grad is not None:
            params.append(self._opacity)
        if self.appearance_config.stage2_refine_scale and self._scaling.grad is not None:
            params.append(self._scaling)
        if self.appearance_config.stage2_refine_rotation and self._rotation.grad is not None:
            params.append(self._rotation)
        if params:
            torch.nn.utils.clip_grad_norm_(params, float(self.appearance_config.stage2_geom_grad_clip))
            self.update_stage2_geometry_grad_stats()

    def clamp_stage2_refine_parameters(self, iteration=None):
        if iteration is not None and not self.is_stage2_geom_refine_active(iteration):
            return
        if self._stage2_ref_opacity is None or self._stage2_ref_scaling is None or self._stage2_ref_rotation is None:
            return
        with torch.no_grad():
            self._last_xyz_update_norm = self._xyz.new_tensor(0.0)
            self._last_f_dc_update_norm = self._features_dc.new_tensor(0.0)
            self._last_f_rest_update_norm = self._features_rest.new_tensor(0.0)
            if self.appearance_config.stage2_refine_opacity:
                before = self._opacity.detach().clone()
                lo = self._stage2_ref_opacity - float(self.appearance_config.stage2_opacity_delta_clip)
                hi = self._stage2_ref_opacity + float(self.appearance_config.stage2_opacity_delta_clip)
                self._opacity.data.clamp_(min=lo, max=hi)
                self._last_opacity_update_norm = (self._opacity.detach() - before).norm()
            else:
                self._last_opacity_update_norm = self._opacity.new_tensor(0.0)
            if self.appearance_config.stage2_refine_scale:
                before = self._scaling.detach().clone()
                lo = self._stage2_ref_scaling - float(self.appearance_config.stage2_scale_delta_clip)
                hi = self._stage2_ref_scaling + float(self.appearance_config.stage2_scale_delta_clip)
                self._scaling.data.clamp_(min=lo, max=hi)
                self._last_scale_update_norm = (self._scaling.detach() - before).norm()
            else:
                self._last_scale_update_norm = self._scaling.new_tensor(0.0)
            if self.appearance_config.stage2_refine_rotation:
                before = self._rotation.detach().clone()
                lo = self._stage2_ref_rotation - float(self.appearance_config.stage2_rotation_delta_clip)
                hi = self._stage2_ref_rotation + float(self.appearance_config.stage2_rotation_delta_clip)
                self._rotation.data.clamp_(min=lo, max=hi)
                self._last_rotation_update_norm = (self._rotation.detach() - before).norm()
            else:
                self._last_rotation_update_norm = self._rotation.new_tensor(0.0)

    def compute_stage2_anchor_regularization(self):
        if not (self.appearance_config.stage2_joint_refine or self.appearance_config.stage2_refine_xyz):
            return self._xyz.new_tensor(0.0)
        if self._stage2_ref_xyz is None:
            return self._xyz.new_tensor(0.0)
        reg = self._xyz.new_tensor(0.0)
        cfg = self.appearance_config
        if cfg.stage2_anchor_lambda_xyz > 0.0:
            reg = reg + float(cfg.stage2_anchor_lambda_xyz) * (self._xyz - self._stage2_ref_xyz).pow(2).mean()
        if cfg.stage2_anchor_lambda_sh > 0.0:
            reg = reg + float(cfg.stage2_anchor_lambda_sh) * (
                (self._features_dc - self._stage2_ref_features_dc).pow(2).mean()
                + (self._features_rest - self._stage2_ref_features_rest).pow(2).mean()
            )
        if cfg.stage2_anchor_lambda_opacity > 0.0:
            reg = reg + float(cfg.stage2_anchor_lambda_opacity) * (self._opacity - self._stage2_ref_opacity).pow(2).mean()
        if cfg.stage2_anchor_lambda_scale > 0.0:
            reg = reg + float(cfg.stage2_anchor_lambda_scale) * (self._scaling - self._stage2_ref_scaling).pow(2).mean()
        if cfg.stage2_anchor_lambda_rotation > 0.0:
            reg = reg + float(cfg.stage2_anchor_lambda_rotation) * (self._rotation - self._stage2_ref_rotation).pow(2).mean()
        return reg

    def get_stage2_refine_stats(self):
        device = self._xyz.device if torch.is_tensor(self._xyz) else "cuda"
        zero = torch.tensor(0.0, device=device)
        if self._stage2_ref_opacity is None or self._stage2_ref_scaling is None or self._stage2_ref_rotation is None:
            return {
                "xyz_delta_abs_mean": zero,
                "f_dc_delta_abs_mean": zero,
                "f_rest_delta_abs_mean": zero,
                "opacity_delta_abs_mean": zero,
                "opacity_delta_max": zero,
                "scale_delta_abs_mean": zero,
                "scale_delta_max": zero,
                "rotation_delta_abs_mean": zero,
                "rotation_delta_max": zero,
                "xyz_grad_norm": zero,
                "f_dc_grad_norm": zero,
                "f_rest_grad_norm": zero,
                "opacity_grad_norm": zero,
                "scale_grad_norm": zero,
                "rotation_grad_norm": zero,
                "xyz_update_norm": zero,
                "f_dc_update_norm": zero,
                "f_rest_update_norm": zero,
                "opacity_update_norm": zero,
                "scale_update_norm": zero,
                "rotation_update_norm": zero,
                "stage2_substage": "none",
                "num_gaussians": int(self.get_xyz.shape[0]) if self.get_xyz.numel() > 0 else 0,
                "densify_event_count": int(self.densify_event_count),
                "prune_event_count": int(self.prune_event_count),
                "last_densify_added": int(self.last_densify_added),
                "last_prune_removed": int(self.last_prune_removed),
            }
        opacity_delta = (self._opacity.detach() - self._stage2_ref_opacity).abs()
        scale_delta = (self._scaling.detach() - self._stage2_ref_scaling).abs()
        rotation_delta = (self._rotation.detach() - self._stage2_ref_rotation).abs()
        xyz_delta = (self._xyz.detach() - self._stage2_ref_xyz).abs()
        f_dc_delta = (self._features_dc.detach() - self._stage2_ref_features_dc).abs()
        f_rest_delta = (self._features_rest.detach() - self._stage2_ref_features_rest).abs()
        return {
            "xyz_delta_abs_mean": xyz_delta.mean(),
            "f_dc_delta_abs_mean": f_dc_delta.mean(),
            "f_rest_delta_abs_mean": f_rest_delta.mean(),
            "opacity_delta_abs_mean": opacity_delta.mean(),
            "opacity_delta_max": opacity_delta.max(),
            "scale_delta_abs_mean": scale_delta.mean(),
            "scale_delta_max": scale_delta.max(),
            "rotation_delta_abs_mean": rotation_delta.mean(),
            "rotation_delta_max": rotation_delta.max(),
            "xyz_grad_norm": self._last_xyz_grad_norm.detach() if torch.is_tensor(self._last_xyz_grad_norm) else zero,
            "f_dc_grad_norm": self._last_f_dc_grad_norm.detach() if torch.is_tensor(self._last_f_dc_grad_norm) else zero,
            "f_rest_grad_norm": self._last_f_rest_grad_norm.detach() if torch.is_tensor(self._last_f_rest_grad_norm) else zero,
            "opacity_grad_norm": self._last_opacity_grad_norm.detach() if torch.is_tensor(self._last_opacity_grad_norm) else zero,
            "scale_grad_norm": self._last_scale_grad_norm.detach() if torch.is_tensor(self._last_scale_grad_norm) else zero,
            "rotation_grad_norm": self._last_rotation_grad_norm.detach() if torch.is_tensor(self._last_rotation_grad_norm) else zero,
            "xyz_update_norm": self._last_xyz_update_norm.detach() if torch.is_tensor(self._last_xyz_update_norm) else zero,
            "f_dc_update_norm": self._last_f_dc_update_norm.detach() if torch.is_tensor(self._last_f_dc_update_norm) else zero,
            "f_rest_update_norm": self._last_f_rest_update_norm.detach() if torch.is_tensor(self._last_f_rest_update_norm) else zero,
            "opacity_update_norm": self._last_opacity_update_norm.detach() if torch.is_tensor(self._last_opacity_update_norm) else zero,
            "scale_update_norm": self._last_scale_update_norm.detach() if torch.is_tensor(self._last_scale_update_norm) else zero,
            "rotation_update_norm": self._last_rotation_update_norm.detach() if torch.is_tensor(self._last_rotation_update_norm) else zero,
            "stage2_substage": self.get_stage2_substage(),
            "num_gaussians": int(self.get_xyz.shape[0]),
            "densify_event_count": int(self.densify_event_count),
            "prune_event_count": int(self.prune_event_count),
            "last_densify_added": int(self.last_densify_added),
            "last_prune_removed": int(self.last_prune_removed),
        }

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self._appearance_latent.numel() > 0:
            for i in range(self._appearance_latent.shape[1]):
                l.append('app_latent_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        appearance_latent = self._appearance_latent.detach().cpu().numpy() if self._appearance_latent.numel() > 0 else np.zeros((xyz.shape[0], 0), dtype=np.float32)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, appearance_latent), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        latent_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("app_latent_")]
        latent_names = sorted(latent_names, key=lambda x: int(x.split('_')[-1]))
        if self.appearance_residual_enabled and latent_names:
            self.appearance_training_active = True
            appearance_latent = np.zeros((xyz.shape[0], len(latent_names)))
            for idx, attr_name in enumerate(latent_names):
                appearance_latent[:, idx] = np.asarray(plydata.elements[0][attr_name])
            loaded_appearance_latent = torch.tensor(appearance_latent, dtype=torch.float, device="cuda")
            self._init_appearance_parameters(device=self._xyz.device, dtype=self._xyz.dtype)
            self._appearance_latent = nn.Parameter(
                loaded_appearance_latent.requires_grad_(True)
            )
        else:
            self.appearance_training_active = False
            self._set_empty_appearance_latent(xyz.shape[0], device=self._xyz.device, dtype=self._xyz.dtype)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        self.last_prune_removed = int(mask.sum().item()) if torch.is_tensor(mask) else 0
        if self.last_prune_removed > 0:
            self.prune_event_count += 1
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "appearance_latent" in optimizable_tensors:
            self._appearance_latent = optimizable_tensors["appearance_latent"]
        elif self._appearance_latent.shape[0] == valid_points_mask.shape[0]:
            self._appearance_latent = nn.Parameter(
                self._appearance_latent[valid_points_mask].requires_grad_(self.appearance_residual_enabled)
            )

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self.reset_stage2_refine_reference()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_appearance_latent, new_tmp_radii):
        self.last_densify_added = int(new_xyz.shape[0])
        if self.last_densify_added > 0:
            self.densify_event_count += 1
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if any(group["name"] == "appearance_latent" for group in self.optimizer.param_groups):
            d["appearance_latent"] = new_appearance_latent

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "appearance_latent" in optimizable_tensors:
            self._appearance_latent = optimizable_tensors["appearance_latent"]
        elif self._appearance_latent.shape[0] > 0:
            self._appearance_latent = nn.Parameter(
                torch.cat((self._appearance_latent, new_appearance_latent), dim=0).requires_grad_(self.appearance_residual_enabled)
            )

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.reset_stage2_refine_reference()

    def reset_stage2_refine_reference(self):
        self._stage2_ref_xyz = None
        self._stage2_ref_features_dc = None
        self._stage2_ref_features_rest = None
        self._stage2_ref_opacity = None
        self._stage2_ref_scaling = None
        self._stage2_ref_rotation = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_appearance_latent = self._appearance_latent[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_appearance_latent, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_appearance_latent = self._appearance_latent[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_appearance_latent, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def stage2_residual_guided_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, hard_region_score=0.0):
        score = float(max(0.0, min(1.0, hard_region_score)))
        guided_threshold = float(max_grad) / (1.0 + score)
        self.densify_and_prune(guided_threshold, min_opacity, extent, max_screen_size, radii)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



