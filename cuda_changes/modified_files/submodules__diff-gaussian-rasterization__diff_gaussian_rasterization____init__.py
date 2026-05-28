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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    appearance_latent,
    app_w_rgb,
    app_b_rgb,
    app_w_gate,
    app_b_gate,
    app_w_diff,
    app_b_diff,
    app_w_spec,
    app_b_spec,
    app_w_mask,
    app_b_mask,
    app_w2_gate,
    app_b2_gate,
    app_w2_diff,
    app_b2_diff,
    app_w2_spec,
    app_b2_spec,
    app_w2_mask,
    app_b2_mask,
    appearance_latent_dim,
    appearance_lambda,
    appearance_gate_floor,
    spec_mask_temperature,
    appearance_enabled,
    appearance_detach_xyz_grad,
    appearance_detach_shape_grad,
    use_local_aniso_encoding,
    use_decoupled_residual,
    disable_global_gate,
    use_two_layer_ddsr_heads,
    ddsr_head_hidden_dim,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        appearance_latent,
        app_w_rgb,
        app_b_rgb,
        app_w_gate,
        app_b_gate,
        app_w_diff,
        app_b_diff,
        app_w_spec,
        app_b_spec,
        app_w_mask,
        app_b_mask,
        app_w2_gate,
        app_b2_gate,
        app_w2_diff,
        app_b2_diff,
        app_w2_spec,
        app_b2_spec,
        app_w2_mask,
        app_b2_mask,
        appearance_latent_dim,
        appearance_lambda,
        appearance_gate_floor,
        spec_mask_temperature,
        appearance_enabled,
        appearance_detach_xyz_grad,
        appearance_detach_shape_grad,
        use_local_aniso_encoding,
        use_decoupled_residual,
        disable_global_gate,
        use_two_layer_ddsr_heads,
        ddsr_head_hidden_dim,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        appearance_latent,
        app_w_rgb,
        app_b_rgb,
        app_w_gate,
        app_b_gate,
        app_w_diff,
        app_b_diff,
        app_w_spec,
        app_b_spec,
        app_w_mask,
        app_b_mask,
        app_w2_gate,
        app_b2_gate,
        app_w2_diff,
        app_b2_diff,
        app_w2_spec,
        app_b2_spec,
        app_w2_mask,
        app_b2_mask,
        appearance_latent_dim,
        appearance_lambda,
        appearance_gate_floor,
        spec_mask_temperature,
        appearance_enabled,
        appearance_detach_xyz_grad,
        appearance_detach_shape_grad,
        use_local_aniso_encoding,
        use_decoupled_residual,
        disable_global_gate,
        use_two_layer_ddsr_heads,
        ddsr_head_hidden_dim,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            appearance_latent,
            app_w_rgb,
            app_b_rgb,
            app_w_gate,
            app_b_gate,
            app_w_diff,
            app_b_diff,
            app_w_spec,
            app_b_spec,
            app_w_mask,
            app_b_mask,
            app_w2_gate,
            app_b2_gate,
            app_w2_diff,
            app_b2_diff,
            app_w2_spec,
            app_b2_spec,
            app_w2_mask,
            app_b2_mask,
            appearance_latent_dim,
            appearance_lambda,
            appearance_gate_floor,
            spec_mask_temperature,
            appearance_enabled,
            appearance_detach_xyz_grad,
            appearance_detach_shape_grad,
            use_local_aniso_encoding,
            use_decoupled_residual,
            disable_global_gate,
            use_two_layer_ddsr_heads,
            ddsr_head_hidden_dim,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.appearance_lambda = appearance_lambda
        ctx.appearance_latent_dim = appearance_latent_dim
        ctx.appearance_gate_floor = appearance_gate_floor
        ctx.spec_mask_temperature = spec_mask_temperature
        ctx.appearance_enabled = appearance_enabled
        ctx.appearance_detach_xyz_grad = appearance_detach_xyz_grad
        ctx.appearance_detach_shape_grad = appearance_detach_shape_grad
        ctx.use_local_aniso_encoding = use_local_aniso_encoding
        ctx.use_decoupled_residual = use_decoupled_residual
        ctx.disable_global_gate = disable_global_gate
        ctx.use_two_layer_ddsr_heads = use_two_layer_ddsr_heads
        ctx.ddsr_head_hidden_dim = ddsr_head_hidden_dim
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            opacities,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            appearance_latent,
            app_w_rgb,
            app_b_rgb,
            app_w_gate,
            app_b_gate,
            app_w_diff,
            app_b_diff,
            app_w_spec,
            app_b_spec,
            app_w_mask,
            app_b_mask,
            app_w2_gate,
            app_b2_gate,
            app_w2_diff,
            app_b2_diff,
            app_w2_spec,
            app_b2_spec,
            app_w2_mask,
            app_b2_mask,
        )
        return color, radii, depths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities,
         geomBuffer, binningBuffer, imgBuffer, appearance_latent, app_w_rgb, app_b_rgb,
         app_w_gate, app_b_gate, app_w_diff, app_b_diff, app_w_spec, app_b_spec,
         app_w_mask, app_b_mask, app_w2_gate, app_b2_gate, app_w2_diff, app_b2_diff,
         app_w2_spec, app_b2_spec, app_w2_mask, app_b2_mask) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                appearance_latent,
                app_w_rgb,
                app_b_rgb,
                app_w_gate,
                app_b_gate,
                app_w_diff,
                app_b_diff,
                app_w_spec,
                app_b_spec,
                app_w_mask,
                app_b_mask,
                app_w2_gate,
                app_b2_gate,
                app_w2_diff,
                app_b2_diff,
                app_w2_spec,
                app_b2_spec,
                app_w2_mask,
                app_b2_mask,
                ctx.appearance_latent_dim,
                ctx.appearance_lambda,
                ctx.appearance_gate_floor,
                ctx.spec_mask_temperature,
                ctx.appearance_enabled,
                ctx.appearance_detach_xyz_grad,
                ctx.appearance_detach_shape_grad,
                ctx.use_local_aniso_encoding,
                ctx.use_decoupled_residual,
                ctx.disable_global_gate,
                ctx.use_two_layer_ddsr_heads,
                ctx.ddsr_head_hidden_dim,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp,
         grad_sh, grad_scales, grad_rotations, grad_appearance_latent, grad_app_w_rgb,
         grad_app_b_rgb, grad_app_w_gate, grad_app_b_gate, grad_app_w_diff, grad_app_b_diff,
         grad_app_w_spec, grad_app_b_spec, grad_app_w_mask, grad_app_b_mask,
         grad_app_w2_gate, grad_app_b2_gate, grad_app_w2_diff, grad_app_b2_diff,
         grad_app_w2_spec, grad_app_b2_spec, grad_app_w2_mask, grad_app_b2_mask) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_appearance_latent,
            grad_app_w_rgb,
            grad_app_b_rgb,
            grad_app_w_gate,
            grad_app_b_gate,
            grad_app_w_diff,
            grad_app_b_diff,
            grad_app_w_spec,
            grad_app_b_spec,
            grad_app_w_mask,
            grad_app_b_mask,
            grad_app_w2_gate,
            grad_app_b2_gate,
            grad_app_w2_diff,
            grad_app_b2_diff,
            grad_app_w2_spec,
            grad_app_b2_spec,
            grad_app_w2_mask,
            grad_app_b2_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, appearance_inputs = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = means3D.new_empty((0,))
        if colors_precomp is None:
            colors_precomp = means3D.new_empty((0,))

        if scales is None:
            scales = means3D.new_empty((0,))
        if rotations is None:
            rotations = means3D.new_empty((0,))
        if cov3D_precomp is None:
            cov3D_precomp = means3D.new_empty((0,))
        if appearance_inputs is None:
            appearance_inputs = {}
        def _tensor_or_empty(name):
            value = appearance_inputs.get(name, None)
            return value if isinstance(value, torch.Tensor) else means3D.new_empty((0,))
        appearance_latent = appearance_inputs.get("appearance_latent", means3D.new_empty((0,)))
        app_w_rgb = _tensor_or_empty("app_w_rgb")
        app_b_rgb = _tensor_or_empty("app_b_rgb")
        app_w_gate = _tensor_or_empty("app_w_gate")
        app_b_gate = _tensor_or_empty("app_b_gate")
        app_w_diff = _tensor_or_empty("app_w_diff")
        app_b_diff = _tensor_or_empty("app_b_diff")
        app_w_spec = _tensor_or_empty("app_w_spec")
        app_b_spec = _tensor_or_empty("app_b_spec")
        app_w_mask = _tensor_or_empty("app_w_mask")
        app_b_mask = _tensor_or_empty("app_b_mask")
        app_w2_gate = _tensor_or_empty("app_w2_gate")
        app_b2_gate = _tensor_or_empty("app_b2_gate")
        app_w2_diff = _tensor_or_empty("app_w2_diff")
        app_b2_diff = _tensor_or_empty("app_b2_diff")
        app_w2_spec = _tensor_or_empty("app_w2_spec")
        app_b2_spec = _tensor_or_empty("app_b2_spec")
        app_w2_mask = _tensor_or_empty("app_w2_mask")
        app_b2_mask = _tensor_or_empty("app_b2_mask")
        appearance_latent_dim = int(appearance_inputs.get("latent_dim", 0))
        appearance_lambda = float(appearance_inputs.get("lambda_t", 0.0))
        appearance_gate_floor = float(appearance_inputs.get("gate_floor", 0.0))
        spec_mask_temperature = float(appearance_inputs.get("spec_mask_temperature", 1.0))
        appearance_enabled = bool(appearance_inputs.get("enabled", False))
        appearance_detach_xyz_grad = bool(appearance_inputs.get("detach_xyz_grad", True))
        appearance_detach_shape_grad = bool(appearance_inputs.get("detach_shape_grad", True))
        use_local_aniso_encoding = bool(appearance_inputs.get("use_local_aniso_encoding", False))
        use_decoupled_residual = bool(appearance_inputs.get("use_decoupled_residual", False))
        disable_global_gate = bool(appearance_inputs.get("disable_global_gate", False))
        use_two_layer_ddsr_heads = bool(appearance_inputs.get("use_two_layer_ddsr_heads", False))
        ddsr_head_hidden_dim = int(appearance_inputs.get("ddsr_head_hidden_dim", 0))

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            appearance_latent,
            app_w_rgb,
            app_b_rgb,
            app_w_gate,
            app_b_gate,
            app_w_diff,
            app_b_diff,
            app_w_spec,
            app_b_spec,
            app_w_mask,
            app_b_mask,
            app_w2_gate,
            app_b2_gate,
            app_w2_diff,
            app_b2_diff,
            app_w2_spec,
            app_b2_spec,
            app_w2_mask,
            app_b2_mask,
            appearance_latent_dim,
            appearance_lambda,
            appearance_gate_floor,
            spec_mask_temperature,
            appearance_enabled,
            appearance_detach_xyz_grad,
            appearance_detach_shape_grad,
            use_local_aniso_encoding,
            use_decoupled_residual,
            disable_global_gate,
            use_two_layer_ddsr_heads,
            ddsr_head_hidden_dim,
            raster_settings, 
        )




