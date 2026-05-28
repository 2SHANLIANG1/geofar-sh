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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  float* out_depthptr = out_depth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		appearance_latent.contiguous().data_ptr<float>(),
		app_w_rgb.contiguous().data_ptr<float>(),
		app_b_rgb.contiguous().data_ptr<float>(),
		app_w_gate.contiguous().data_ptr<float>(),
		app_b_gate.contiguous().data_ptr<float>(),
		app_w_diff.contiguous().data_ptr<float>(),
		app_b_diff.contiguous().data_ptr<float>(),
		app_w_spec.contiguous().data_ptr<float>(),
		app_b_spec.contiguous().data_ptr<float>(),
		app_w_mask.contiguous().data_ptr<float>(),
		app_b_mask.contiguous().data_ptr<float>(),
		app_w2_gate.contiguous().data_ptr<float>(),
		app_b2_gate.contiguous().data_ptr<float>(),
		app_w2_diff.contiguous().data_ptr<float>(),
		app_b2_diff.contiguous().data_ptr<float>(),
		app_w2_spec.contiguous().data_ptr<float>(),
		app_b2_spec.contiguous().data_ptr<float>(),
		app_w2_mask.contiguous().data_ptr<float>(),
		app_b2_mask.contiguous().data_ptr<float>(),
		appearance_latent_dim,
		appearance_lambda,
		appearance_gate_floor,
		spec_mask_temperature,
		appearance_enabled,
		use_local_aniso_encoding,
		use_decoupled_residual,
		disable_global_gate,
		use_two_layer_ddsr_heads,
		ddsr_head_hidden_dim,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth);
}

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
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  int latent_dim = appearance_latent.numel() > 0 ? appearance_latent.size(1) : 0;
  int feature_dim = latent_dim + 4 + (use_local_aniso_encoding ? 5 : 0);
  torch::Tensor dL_dappearance_latent = torch::zeros({P, latent_dim}, means3D.options());
  torch::Tensor dL_dapp_w_rgb = torch::zeros({3, feature_dim}, means3D.options());
  torch::Tensor dL_dapp_b_rgb = torch::zeros({3}, means3D.options());
  torch::Tensor dL_dapp_w_gate = torch::zeros_like(app_w_gate);
  torch::Tensor dL_dapp_b_gate = torch::zeros_like(app_b_gate);
  torch::Tensor dL_dapp_w_diff = torch::zeros_like(app_w_diff);
  torch::Tensor dL_dapp_b_diff = torch::zeros_like(app_b_diff);
  torch::Tensor dL_dapp_w_spec = torch::zeros_like(app_w_spec);
  torch::Tensor dL_dapp_b_spec = torch::zeros_like(app_b_spec);
  torch::Tensor dL_dapp_w_mask = torch::zeros_like(app_w_mask);
  torch::Tensor dL_dapp_b_mask = torch::zeros_like(app_b_mask);
  torch::Tensor dL_dapp_w2_gate = torch::zeros_like(app_w2_gate);
  torch::Tensor dL_dapp_b2_gate = torch::zeros_like(app_b2_gate);
  torch::Tensor dL_dapp_w2_diff = torch::zeros_like(app_w2_diff);
  torch::Tensor dL_dapp_b2_diff = torch::zeros_like(app_b2_diff);
  torch::Tensor dL_dapp_w2_spec = torch::zeros_like(app_w2_spec);
  torch::Tensor dL_dapp_b2_spec = torch::zeros_like(app_b2_spec);
  torch::Tensor dL_dapp_w2_mask = torch::zeros_like(app_w2_mask);
  torch::Tensor dL_dapp_b2_mask = torch::zeros_like(app_b2_mask);
  torch::Tensor dL_ddepths = torch::zeros({0, 1}, means3D.options());

  float* dL_ddepthsptr = nullptr;
  float* dL_dout_depthptr = nullptr;
  if(dL_dout_depth.size(0) != 0)
  {
	dL_ddepths = torch::zeros({P, 1}, means3D.options());
	dL_ddepths = dL_ddepths.contiguous();
	dL_ddepthsptr = dL_ddepths.data<float>();
	dL_dout_depthptr = dL_dout_depth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  appearance_latent.contiguous().data_ptr<float>(),
	  app_w_rgb.contiguous().data_ptr<float>(),
	  app_b_rgb.contiguous().data_ptr<float>(),
	  app_w_gate.contiguous().data_ptr<float>(),
	  app_b_gate.contiguous().data_ptr<float>(),
	  app_w_diff.contiguous().data_ptr<float>(),
	  app_b_diff.contiguous().data_ptr<float>(),
	  app_w_spec.contiguous().data_ptr<float>(),
	  app_b_spec.contiguous().data_ptr<float>(),
	  app_w_mask.contiguous().data_ptr<float>(),
	  app_b_mask.contiguous().data_ptr<float>(),
	  app_w2_gate.contiguous().data_ptr<float>(),
	  app_b2_gate.contiguous().data_ptr<float>(),
	  app_w2_diff.contiguous().data_ptr<float>(),
	  app_b2_diff.contiguous().data_ptr<float>(),
	  app_w2_spec.contiguous().data_ptr<float>(),
	  app_b2_spec.contiguous().data_ptr<float>(),
	  app_w2_mask.contiguous().data_ptr<float>(),
	  app_b2_mask.contiguous().data_ptr<float>(),
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
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dappearance_latent.contiguous().data_ptr<float>(),
	  dL_dapp_w_rgb.contiguous().data_ptr<float>(),
	  dL_dapp_b_rgb.contiguous().data_ptr<float>(),
	  dL_dapp_w_gate.contiguous().data_ptr<float>(),
	  dL_dapp_b_gate.contiguous().data_ptr<float>(),
	  dL_dapp_w_diff.contiguous().data_ptr<float>(),
	  dL_dapp_b_diff.contiguous().data_ptr<float>(),
	  dL_dapp_w_spec.contiguous().data_ptr<float>(),
	  dL_dapp_b_spec.contiguous().data_ptr<float>(),
	  dL_dapp_w_mask.contiguous().data_ptr<float>(),
	  dL_dapp_b_mask.contiguous().data_ptr<float>(),
	  dL_dapp_w2_gate.contiguous().data_ptr<float>(),
	  dL_dapp_b2_gate.contiguous().data_ptr<float>(),
	  dL_dapp_w2_diff.contiguous().data_ptr<float>(),
	  dL_dapp_b2_diff.contiguous().data_ptr<float>(),
	  dL_dapp_w2_spec.contiguous().data_ptr<float>(),
	  dL_dapp_b2_spec.contiguous().data_ptr<float>(),
	  dL_dapp_w2_mask.contiguous().data_ptr<float>(),
	  dL_dapp_b2_mask.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dappearance_latent, dL_dapp_w_rgb, dL_dapp_b_rgb, dL_dapp_w_gate, dL_dapp_b_gate, dL_dapp_w_diff, dL_dapp_b_diff, dL_dapp_w_spec, dL_dapp_b_spec, dL_dapp_w_mask, dL_dapp_b_mask, dL_dapp_w2_gate, dL_dapp_b2_gate, dL_dapp_w2_diff, dL_dapp_b2_diff, dL_dapp_w2_spec, dL_dapp_b2_spec, dL_dapp_w2_mask, dL_dapp_b2_mask);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}



