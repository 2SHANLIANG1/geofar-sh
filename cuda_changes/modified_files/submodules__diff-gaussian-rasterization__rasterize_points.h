/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& appearance_latent,
	const torch::Tensor& app_w_rgb,
	const torch::Tensor& app_b_rgb,
	const torch::Tensor& app_w_gate,
	const torch::Tensor& app_b_gate,
	const torch::Tensor& app_w_diff,
	const torch::Tensor& app_b_diff,
	const torch::Tensor& app_w_spec,
	const torch::Tensor& app_b_spec,
	const torch::Tensor& app_w_mask,
	const torch::Tensor& app_b_mask,
	const torch::Tensor& app_w2_gate,
	const torch::Tensor& app_b2_gate,
	const torch::Tensor& app_w2_diff,
	const torch::Tensor& app_b2_diff,
	const torch::Tensor& app_w2_spec,
	const torch::Tensor& app_b2_spec,
	const torch::Tensor& app_w2_mask,
	const torch::Tensor& app_b2_mask,
	const int appearance_latent_dim,
	const float appearance_lambda,
	const float appearance_gate_floor,
	const float spec_mask_temperature,
	const bool appearance_enabled,
	const bool appearance_detach_xyz_grad,
	const bool appearance_detach_shape_grad,
	const bool use_local_aniso_encoding,
	const bool use_decoupled_residual,
	const bool disable_global_gate,
	const bool use_two_layer_ddsr_heads,
	const int ddsr_head_hidden_dim,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& appearance_latent,
	const torch::Tensor& app_w_rgb,
	const torch::Tensor& app_b_rgb,
	const torch::Tensor& app_w_gate,
	const torch::Tensor& app_b_gate,
	const torch::Tensor& app_w_diff,
	const torch::Tensor& app_b_diff,
	const torch::Tensor& app_w_spec,
	const torch::Tensor& app_b_spec,
	const torch::Tensor& app_w_mask,
	const torch::Tensor& app_b_mask,
	const torch::Tensor& app_w2_gate,
	const torch::Tensor& app_b2_gate,
	const torch::Tensor& app_w2_diff,
	const torch::Tensor& app_b2_diff,
	const torch::Tensor& app_w2_spec,
	const torch::Tensor& app_b2_spec,
	const torch::Tensor& app_w2_mask,
	const torch::Tensor& app_b2_mask,
	const int appearance_latent_dim,
	const float appearance_lambda,
	const float appearance_gate_floor,
	const float spec_mask_temperature,
	const bool appearance_enabled,
	const bool appearance_detach_xyz_grad,
	const bool appearance_detach_shape_grad,
	const bool use_local_aniso_encoding,
	const bool use_decoupled_residual,
	const bool disable_global_gate,
	const bool use_two_layer_ddsr_heads,
	const int ddsr_head_hidden_dim,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);



