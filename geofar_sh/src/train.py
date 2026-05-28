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

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from utils.import_utils import ensure_local_diff_gaussian_rasterization
ensure_local_diff_gaussian_rasterization()

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from scene.appearance_residual import compute_appearance_regularizers
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "unavailable"
    except Exception:
        return "unavailable"


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def scalarize(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def variant_name(gaussians):
    if gaussians.appearance_config.mode != "none":
        return "sh_cuda_appearance_residual"
    return "baseline_3dgs"


def prepare_output_and_logger(dataset, opt, pipe):
    if not dataset.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok=True)

    log_path = os.path.join(dataset.model_path, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    merged_cfg = {
        "model": vars(dataset),
        "optimization": vars(opt),
        "pipeline": vars(pipe),
        "command": "python " + " ".join(sys.argv),
        "git_commit": get_git_commit_hash(),
        "created_at_unix": time.time(),
    }
    with open(os.path.join(dataset.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**{**vars(dataset), **vars(opt), **vars(pipe)})))
    save_json(os.path.join(dataset.model_path, "config.yaml"), merged_cfg)

    tb_writer = SummaryWriter(dataset.model_path) if TENSORBOARD_FOUND else None
    if not TENSORBOARD_FOUND:
        print("Tensorboard not available: not logging progress")

    run_state = {
        "variant": "uninitialized",
        "git_commit": merged_cfg["git_commit"],
        "command": merged_cfg["command"],
        "train_history": [],
        "evaluations": [],
        "best_eval": None,
        "timing": {},
        "memory": {},
    }
    save_json(os.path.join(dataset.model_path, "metrics.json"), run_state)
    return tb_writer, run_state


def append_train_history(run_state, iteration, loss, ll1, depth_loss, appearance_stats):
    record = {
            "iteration": int(iteration),
            "loss": float(loss),
            "l1": float(ll1),
            "depth_loss": float(depth_loss),
            "lambda_t": float(appearance_stats["lambda_t"]),
            "delta_abs_mean": float(appearance_stats["delta_abs_mean"]),
            "delta_max_abs": float(appearance_stats["delta_max_abs"]),
            "gate_mean": float(appearance_stats["gate_mean"]),
            "gate_std": float(appearance_stats["gate_std"]),
            "gate_max": float(appearance_stats["gate_max"]),
            "local_dir_abs_mean": float(appearance_stats["local_dir_abs_mean"]),
            "anisotropy_ratio_mean": float(appearance_stats["anisotropy_ratio_mean"]),
            "delta_diff_abs_mean": float(appearance_stats["delta_diff_abs_mean"]),
            "delta_spec_abs_mean": float(appearance_stats["delta_spec_abs_mean"]),
            "spec_mask_mean": float(appearance_stats["spec_mask_mean"]),
            "spec_mask_std": float(appearance_stats["spec_mask_std"]),
            "spec_mask_max": float(appearance_stats["spec_mask_max"]),
            "spec_mask_temperature": float(appearance_stats["spec_mask_temperature"]),
            "diff_consistency_reg": float(appearance_stats["diff_consistency_reg"]),
            "branch_diversity_reg": float(appearance_stats["branch_diversity_reg"]),
            "opacity_delta_abs_mean": float(appearance_stats["opacity_delta_abs_mean"]),
            "opacity_delta_max": float(appearance_stats["opacity_delta_max"]),
            "scale_delta_abs_mean": float(appearance_stats["scale_delta_abs_mean"]),
            "scale_delta_max": float(appearance_stats["scale_delta_max"]),
            "rotation_delta_abs_mean": float(appearance_stats["rotation_delta_abs_mean"]),
            "rotation_delta_max": float(appearance_stats["rotation_delta_max"]),
            "opacity_grad_norm": float(appearance_stats["opacity_grad_norm"]),
            "scale_grad_norm": float(appearance_stats["scale_grad_norm"]),
            "rotation_grad_norm": float(appearance_stats["rotation_grad_norm"]),
            "opacity_update_norm": float(appearance_stats["opacity_update_norm"]),
            "scale_update_norm": float(appearance_stats["scale_update_norm"]),
            "rotation_update_norm": float(appearance_stats["rotation_update_norm"]),
            "stage2_substage": str(appearance_stats.get("stage2_substage", "none")),
            "residual_reg": float(appearance_stats["residual_reg"]),
            "gate_reg": float(appearance_stats["gate_reg"]),
            "smooth_reg": float(appearance_stats["smooth_reg"]),
        }
    for key in (
        "stage2_anchor_reg",
        "xyz_delta_abs_mean", "f_dc_delta_abs_mean", "f_rest_delta_abs_mean",
        "xyz_grad_norm", "f_dc_grad_norm", "f_rest_grad_norm",
        "num_gaussians", "densify_event_count", "prune_event_count", "last_densify_added", "last_prune_removed",
    ):
        if key in appearance_stats:
            value = appearance_stats[key]
            record[key] = float(value) if not isinstance(value, str) else value
    run_state["train_history"].append(record)


def append_basic_train_history(run_state, iteration, loss, ll1, depth_loss, gaussians):
    run_state["train_history"].append(
        {
            "iteration": int(iteration),
            "loss": float(loss),
            "l1": float(ll1),
            "depth_loss": float(depth_loss),
            "num_gaussians": int(gaussians.get_xyz.shape[0]),
            "densify_event_count": int(gaussians.densify_event_count),
            "prune_event_count": int(gaussians.prune_event_count),
            "last_densify_added": int(gaussians.last_densify_added),
            "last_prune_removed": int(gaussians.last_prune_removed),
        }
    )


def update_profile_timing(run_state, render_ms, appearance_reg_ms, backward_ms, optim_ms):
    run_state["timing"]["profile"] = {
        "render_ms": float(render_ms),
        "appearance_reg_ms": float(appearance_reg_ms),
        "backward_ms": float(backward_ms),
        "optim_ms": float(optim_ms),
        "step_ms": float(render_ms + appearance_reg_ms + backward_ms + optim_ms),
    }


def current_stage_name(gaussians, iteration):
    if gaussians.is_stage2_window_active(iteration):
        if gaussians.appearance_residual_enabled:
            return f"stage2_{gaussians.get_stage2_substage(iteration)}"
        if gaussians.appearance_config.stage2_refine_sh:
            return "stage2_sh_refine_only"
        if gaussians.appearance_config.freeze_exposure_in_stage2:
            return "stage2_exposure_frozen"
    appearance_state = gaussians.get_appearance_forward_state(iteration)
    if not appearance_state["enabled"]:
        return "stage1_baseline"
    if not appearance_state.get("enable_stage2", True):
        return "appearance_no_stage2"
    return f"stage2_{gaussians.get_stage2_substage(iteration)}"


def should_track_appearance_metrics(appearance_state):
    return bool(appearance_state["enabled"] or appearance_state["lambda_t"] > 0.0)


def maybe_update_best(scene, run_state, eval_record, iteration):
    if eval_record["split"] != "test":
        return
    current_psnr = eval_record["psnr"]
    best = run_state["best_eval"]
    if best is not None and current_psnr <= best["psnr"]:
        return
    run_state["best_eval"] = {
        "iteration": int(iteration),
        "psnr": float(eval_record["psnr"]),
        "ssim": float(eval_record["ssim"]),
        "lpips": float(eval_record["lpips"]),
    }
    best_path = os.path.join(scene.model_path, "best_checkpoint.pth")
    torch.save((scene.gaussians.capture(), iteration), best_path)
    best_summary_path = os.path.join(scene.model_path, "best_metrics.json")
    save_json(best_summary_path, run_state["best_eval"])


def evaluate_split(iteration, split_name, cameras, scene, render_args, tb_writer, train_test_exp):
    if not cameras:
        return None

    l1_accum = 0.0
    psnr_accum = 0.0
    ssim_accum = 0.0
    lpips_accum = 0.0
    timings = []

    for idx, viewpoint in enumerate(cameras):
        start_time = time.perf_counter()
        rendered = render(viewpoint, scene.gaussians, *render_args, iteration=iteration)["render"]
        timings.append(time.perf_counter() - start_time)
        image = torch.clamp(rendered, 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
        if tb_writer and idx < 5:
            tb_writer.add_images(f"{split_name}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
            if iteration == 0:
                tb_writer.add_images(f"{split_name}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
        l1_accum += l1_loss(image, gt_image).mean().double().item()
        psnr_accum += psnr(image, gt_image).mean().double().item()
        ssim_accum += ssim(image, gt_image).mean().double().item()
        if LPIPS_AVAILABLE:
            lpips_accum += lpips(image, gt_image, net_type="vgg").mean().double().item()

    count = len(cameras)
    return {
        "iteration": int(iteration),
        "split": split_name,
        "l1": l1_accum / count,
        "psnr": psnr_accum / count,
        "ssim": ssim_accum / count,
        "lpips": (lpips_accum / count) if LPIPS_AVAILABLE else -1.0,
        "render_time_sec": sum(timings) / max(count, 1),
    }


def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene, render_args, train_test_exp, run_state):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time_ms", elapsed, iteration)

    eval_records = []
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [
            ("test", scene.getTestCameras()),
            ("train", [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)] if scene.getTrainCameras() else []),
        ]
        for split_name, cameras in validation_configs:
            record = evaluate_split(iteration, split_name, cameras, scene, render_args, tb_writer, train_test_exp)
            if record is None:
                continue
            eval_records.append(record)
            print(
                "\n[ITER {}] Evaluating {}: L1 {:.6f} PSNR {:.6f} SSIM {:.6f} LPIPS {:.6f}".format(
                    iteration,
                    split_name,
                    record["l1"],
                    record["psnr"],
                    record["ssim"],
                    record["lpips"],
                )
            )
            if tb_writer:
                tb_writer.add_scalar(f"{split_name}/loss_viewpoint - l1_loss", record["l1"], iteration)
                tb_writer.add_scalar(f"{split_name}/loss_viewpoint - psnr", record["psnr"], iteration)
                tb_writer.add_scalar(f"{split_name}/loss_viewpoint - ssim", record["ssim"], iteration)
                if record["lpips"] >= 0.0:
                    tb_writer.add_scalar(f"{split_name}/loss_viewpoint - lpips", record["lpips"], iteration)
            run_state["evaluations"].append(record)
            maybe_update_best(scene, run_state, record, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return eval_records


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer, run_state = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.configure_appearance_residual(opt)
    run_state["variant"] = variant_name(gaussians)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)
    appearance_reg_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_delta_for_log = 0.0
    ema_gate_for_log = 0.0
    ema_iter_ms_for_log = 0.0
    save_json(os.path.join(dataset.model_path, "metrics.json"), run_state)
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    stage2_wall_start = None
    appearance_activation_iter = gaussians.get_appearance_activation_iter()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if (
            appearance_activation_iter is not None
            and iteration >= appearance_activation_iter
            and gaussians.activate_appearance_training(opt)
        ):
            print(f"[ITER {iteration}] Activating appearance training")
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                        iteration=iteration,
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        gaussians.apply_stage2_trainability(iteration)
        appearance_forward_state = gaussians.get_appearance_forward_state(iteration)
        appearance_metrics_active = should_track_appearance_metrics(appearance_forward_state)
        stage_name = current_stage_name(gaussians, iteration)
        gaussians.update_stage2_substage(iteration)
        if stage_name.startswith("stage2_"):
            gaussians.ensure_stage2_refine_reference()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            iteration=iteration,
        )
        render_end.record()
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Ll1depth = 0.0
        depth_weight = opt.depth_lambda if opt.depth_lambda > 0 else depth_l1_weight(iteration)
        if depth_weight > 0 and viewpoint_cam.depth_reliable and viewpoint_cam.depth_gt is not None:
            depth_pred = render_pkg["depth"]
            depth_gt = viewpoint_cam.depth_gt.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            valid_pixels = depth_mask.sum().clamp_min(1.0)
            Ll1depth_pure = torch.abs(depth_pred - depth_gt) * depth_mask
            Ll1depth = depth_weight * (Ll1depth_pure.sum() / valid_pixels)
            loss += Ll1depth
            Ll1depth = Ll1depth.item()

        appearance_reg, appearance_stats = compute_appearance_regularizers(gaussians, viewpoint_cam, render_pkg["appearance_outputs"])
        loss = loss + appearance_reg
        appearance_reg_end.record()
        loss.backward()
        if stage_name.startswith("stage2_"):
            gaussians.update_stage2_geometry_grad_stats()
            gaussians.clip_stage2_geometry_gradients(iteration)
            for key, value in gaussians.get_stage2_refine_stats().items():
                appearance_stats[key] = value
        backward_end.record()
        optim_end.record()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_for_log = 0.4 * Ll1depth + 0.6 * ema_depth_for_log
            if appearance_metrics_active:
                ema_delta_for_log = 0.4 * scalarize(appearance_stats["delta_abs_mean"]) + 0.6 * ema_delta_for_log
                ema_gate_for_log = 0.4 * scalarize(appearance_stats["gate_mean"]) + 0.6 * ema_gate_for_log

            should_log_history = (iteration % 10 == 0) or (iteration == opt.iterations)
            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Depth": f"{ema_depth_for_log:.7f}",
                }
                if appearance_metrics_active:
                    postfix["Delta"] = f"{ema_delta_for_log:.5f}"
                    postfix["Gate"] = f"{ema_gate_for_log:.5f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if should_log_history:
                if appearance_metrics_active:
                    append_train_history(run_state, iteration, loss.item(), Ll1.item(), Ll1depth, appearance_stats)
                else:
                    append_basic_train_history(run_state, iteration, loss.item(), Ll1.item(), Ll1depth, gaussians)

            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer and appearance_metrics_active:
                tb_writer.add_scalar("appearance/lambda_t", scalarize(appearance_stats["lambda_t"]), iteration)
                tb_writer.add_scalar("appearance/delta_abs_mean", scalarize(appearance_stats["delta_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/delta_max_abs", scalarize(appearance_stats["delta_max_abs"]), iteration)
                tb_writer.add_scalar("appearance/gate_mean", scalarize(appearance_stats["gate_mean"]), iteration)
                tb_writer.add_scalar("appearance/gate_std", scalarize(appearance_stats["gate_std"]), iteration)
                tb_writer.add_scalar("appearance/gate_max", scalarize(appearance_stats["gate_max"]), iteration)
                tb_writer.add_scalar("appearance/local_dir_abs_mean", scalarize(appearance_stats["local_dir_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/anisotropy_ratio_mean", scalarize(appearance_stats["anisotropy_ratio_mean"]), iteration)
                tb_writer.add_scalar("appearance/delta_diff_abs_mean", scalarize(appearance_stats["delta_diff_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/delta_spec_abs_mean", scalarize(appearance_stats["delta_spec_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/spec_mask_mean", scalarize(appearance_stats["spec_mask_mean"]), iteration)
                tb_writer.add_scalar("appearance/spec_mask_std", scalarize(appearance_stats["spec_mask_std"]), iteration)
                tb_writer.add_scalar("appearance/spec_mask_max", scalarize(appearance_stats["spec_mask_max"]), iteration)
                tb_writer.add_scalar("appearance/diff_consistency_reg", scalarize(appearance_stats["diff_consistency_reg"]), iteration)
                tb_writer.add_scalar("appearance/branch_diversity_reg", scalarize(appearance_stats["branch_diversity_reg"]), iteration)
                tb_writer.add_scalar("appearance/opacity_delta_abs_mean", scalarize(appearance_stats["opacity_delta_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/opacity_delta_max", scalarize(appearance_stats["opacity_delta_max"]), iteration)
                tb_writer.add_scalar("appearance/scale_delta_abs_mean", scalarize(appearance_stats["scale_delta_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/scale_delta_max", scalarize(appearance_stats["scale_delta_max"]), iteration)
                tb_writer.add_scalar("appearance/rotation_delta_abs_mean", scalarize(appearance_stats["rotation_delta_abs_mean"]), iteration)
                tb_writer.add_scalar("appearance/rotation_delta_max", scalarize(appearance_stats["rotation_delta_max"]), iteration)
                tb_writer.add_scalar("appearance/opacity_grad_norm", scalarize(appearance_stats["opacity_grad_norm"]), iteration)
                tb_writer.add_scalar("appearance/scale_grad_norm", scalarize(appearance_stats["scale_grad_norm"]), iteration)
                tb_writer.add_scalar("appearance/rotation_grad_norm", scalarize(appearance_stats["rotation_grad_norm"]), iteration)
                tb_writer.add_scalar("appearance/opacity_update_norm", scalarize(appearance_stats["opacity_update_norm"]), iteration)
                tb_writer.add_scalar("appearance/scale_update_norm", scalarize(appearance_stats["scale_update_norm"]), iteration)
                tb_writer.add_scalar("appearance/rotation_update_norm", scalarize(appearance_stats["rotation_update_norm"]), iteration)
                tb_writer.add_scalar("appearance/residual_reg", scalarize(appearance_stats["residual_reg"]), iteration)
                tb_writer.add_scalar("appearance/gate_reg", scalarize(appearance_stats["gate_reg"]), iteration)
                tb_writer.add_scalar("appearance/smooth_reg", scalarize(appearance_stats["smooth_reg"]), iteration)
                tb_writer.add_scalar("appearance/parameter_norm", gaussians.get_appearance_parameter_norm(), iteration)
            eval_records = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                run_state,
            )
            if iteration % 1000 == 0 and appearance_metrics_active:
                print(
                    f"[ITER {iteration}] Stage={stage_name} substage={appearance_stats.get('stage2_substage', 'none')} "
                    f"lambda_t={scalarize(appearance_stats['lambda_t']):.4f} "
                    f"gate_mean={scalarize(appearance_stats['gate_mean']):.6f} "
                    f"delta_abs_mean={scalarize(appearance_stats['delta_abs_mean']):.6f} "
                    f"rot_delta={scalarize(appearance_stats['rotation_delta_abs_mean']):.6f}"
                )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            stage2_enabled = gaussians.is_stage2_window_active(iteration)
            stage2_cfg = gaussians.appearance_config
            stage2_disable_densify = stage2_cfg.disable_densify_in_stage2 and stage2_enabled
            stage2_disable_prune = getattr(stage2_cfg, "disable_prune_in_stage2", stage2_disable_densify) and stage2_enabled
            if iteration < opt.densify_until_iter and not stage2_disable_densify:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if stage2_disable_prune:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        gaussians.tmp_radii = radii
                        gaussians.densify_and_clone(grads, opt.densify_grad_threshold, scene.cameras_extent)
                        gaussians.densify_and_split(grads, opt.densify_grad_threshold, scene.cameras_extent)
                        gaussians.tmp_radii = None
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if gaussians.appearance_optimizer is not None and appearance_metrics_active:
                    gaussians.appearance_optimizer.step()
                    gaussians.appearance_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if stage_name.startswith("stage2_"):
                    gaussians.clamp_stage2_refine_parameters(iteration)
                optim_end.record()
                iter_end.record()

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

            should_flush_metrics = eval_records or iteration == opt.iterations
            if iteration in checkpoint_iterations or iteration in saving_iterations:
                should_flush_metrics = True
            if appearance_metrics_active:
                should_flush_metrics = should_flush_metrics or (iteration % 10 == 0)
            else:
                should_flush_metrics = should_flush_metrics or (iteration % 1000 == 0)
            if should_flush_metrics:
                torch.cuda.synchronize()
                if stage_name.startswith("stage2_") and stage2_wall_start is None:
                    stage2_wall_start = time.perf_counter()
                ema_iter_ms_for_log = 0.4 * float(iter_start.elapsed_time(iter_end)) + 0.6 * ema_iter_ms_for_log
                run_state["timing"] = {
                    "elapsed_sec": time.perf_counter() - wall_start,
                    "last_iter_ms": float(iter_start.elapsed_time(iter_end)),
                    "ema_iter_ms": ema_iter_ms_for_log,
                }
                run_state["memory"] = {
                    "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                    "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
                }
                run_state["stage"] = stage_name
                run_state["final_num_gaussians"] = int(scene.gaussians.get_xyz.shape[0])
                if getattr(opt, "appearance_profile", False):
                    update_profile_timing(
                        run_state,
                        render_ms=iter_start.elapsed_time(render_end),
                        appearance_reg_ms=render_end.elapsed_time(appearance_reg_end),
                        backward_ms=appearance_reg_end.elapsed_time(backward_end),
                        optim_ms=backward_end.elapsed_time(optim_end),
                    )
                save_json(os.path.join(dataset.model_path, "metrics.json"), run_state)

    total_train_sec = time.perf_counter() - wall_start
    stage2_time_sec = 0.0 if stage2_wall_start is None else max(total_train_sec - (stage2_wall_start - wall_start), 0.0)
    stage1_time_sec = max(total_train_sec - stage2_time_sec, 0.0)
    run_state["timing"]["total_train_sec"] = total_train_sec
    run_state["timing"]["stage1_time_sec"] = stage1_time_sec
    run_state["timing"]["stage2_time_sec"] = stage2_time_sec
    run_state["memory"]["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
    run_state["final_num_gaussians"] = int(scene.gaussians.get_xyz.shape[0])
    final_ckpt = os.path.join(scene.model_path, "final_checkpoint.pth")
    torch.save((gaussians.capture(), opt.iterations), final_ckpt)
    if os.path.exists(final_ckpt):
        run_state["final_checkpoint"] = final_ckpt
    save_json(os.path.join(dataset.model_path, "metrics.json"), run_state)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
    print("\nTraining complete.")



