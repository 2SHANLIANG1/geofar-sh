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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const float* appearance_latent,
		const float* app_w_rgb,
		const float* app_b_rgb,
		const float* app_w_gate,
		const float* app_b_gate,
		const float* app_w_diff,
		const float* app_b_diff,
		const float* app_w_spec,
		const float* app_b_spec,
		const float* app_w_mask,
		const float* app_b_mask,
		const float* app_w2_gate,
		const float* app_b2_gate,
		const float* app_w2_diff,
		const float* app_b2_diff,
		const float* app_w2_spec,
		const float* app_b2_spec,
		const float* app_w2_mask,
		const float* app_b2_mask,
		const int appearance_latent_dim,
		const float appearance_lambda,
		const float appearance_gate_floor,
		const float spec_mask_temperature,
		const bool appearance_enabled,
		const bool use_local_aniso_encoding,
		const bool use_decoupled_residual,
		const bool disable_global_gate,
		const bool use_two_layer_ddsr_heads,
		const int ddsr_head_hidden_dim,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);
}


#endif



