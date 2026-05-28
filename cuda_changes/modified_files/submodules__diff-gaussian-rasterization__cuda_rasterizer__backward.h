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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_depths,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const float* opacities,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
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
		const bool appearance_detach_xyz_grad,
		const bool appearance_detach_shape_grad,
		const bool use_local_aniso_encoding,
		const bool use_decoupled_residual,
		const bool disable_global_gate,
		const bool use_two_layer_ddsr_heads,
		const int ddsr_head_hidden_dim,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_ddepth,
		float* dL_dopacity,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		float* dL_dappearance_latent,
		float* dL_dapp_w_rgb,
		float* dL_dapp_b_rgb,
		float* dL_dapp_w_gate,
		float* dL_dapp_b_gate,
		float* dL_dapp_w_diff,
		float* dL_dapp_b_diff,
		float* dL_dapp_w_spec,
		float* dL_dapp_b_spec,
		float* dL_dapp_w_mask,
		float* dL_dapp_b_mask,
		float* dL_dapp_w2_gate,
		float* dL_dapp_b2_gate,
		float* dL_dapp_w2_diff,
		float* dL_dapp_b2_diff,
		float* dL_dapp_w2_spec,
		float* dL_dapp_b2_spec,
		float* dL_dapp_w2_mask,
		float* dL_dapp_b2_mask,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		bool antialiasing);
}

#endif



