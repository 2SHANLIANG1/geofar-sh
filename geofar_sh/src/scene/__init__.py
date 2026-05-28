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

import os
import random
import json
import torch
from scene.appearance_residual import (
    inspect_appearance_weight_file,
    load_appearance_meta,
    compare_appearance_meta,
    summarize_appearance_render_status,
)
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
            appearance_state_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "appearance_residual.pth",
            )
            load_status, warning_message = inspect_appearance_weight_file(
                enabled=self.gaussians.appearance_residual_enabled,
                weight_path=appearance_state_path,
            )
            self.gaussians.appearance_residual_load_status = load_status
            self.gaussians.appearance_residual_load_warning = warning_message
            if warning_message:
                print(warning_message)
            meta_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "appearance_residual_meta.json",
            )
            loaded_meta, meta_warning = load_appearance_meta(meta_path)
            if meta_warning and self.gaussians.appearance_residual_enabled:
                print(meta_warning)
            if self.gaussians.appearance_residual_enabled and os.path.exists(appearance_state_path):
                appearance_state_dict = torch.load(appearance_state_path, map_location="cuda")
                def _copy_if_compatible(name):
                    if name not in appearance_state_dict or not hasattr(self.gaussians, name):
                        return
                    target = getattr(self.gaussians, name)
                    if target is None:
                        return
                    source = appearance_state_dict[name]
                    if source is None:
                        return
                    if tuple(target.shape) != tuple(source.shape):
                        print(f"[WARN] Skipping incompatible appearance tensor {name}: expected {tuple(target.shape)} loaded {tuple(source.shape)}")
                        return
                    target.data.copy_(source)
                if "app_w_rgb" in appearance_state_dict:
                    for name in (
                        "app_w_rgb", "app_b_rgb", "app_w_gate", "app_b_gate",
                        "app_w_diff", "app_b_diff", "app_w_spec", "app_b_spec",
                        "app_w_mask", "app_b_mask", "app_w2_gate", "app_b2_gate",
                        "app_w2_diff", "app_b2_diff", "app_w2_spec", "app_b2_spec",
                        "app_w2_mask", "app_b2_mask",
                    ):
                        _copy_if_compatible(name)
                self.gaussians.mark_appearance_residual_loaded(True)
                meta_mismatch_warning = compare_appearance_meta(self.gaussians.get_appearance_meta(), loaded_meta)
                if meta_mismatch_warning:
                    print(meta_mismatch_warning)
                print(f"Appearance residual weights loaded from {appearance_state_path}")
            elif not self.gaussians.appearance_residual_enabled:
                print("Render/eval is running in baseline mode without appearance residual.")
            else:
                self.gaussians.mark_appearance_residual_loaded(False)
            print("Appearance render status: {}".format(
                summarize_appearance_render_status(
                    enabled=self.gaussians.appearance_residual_enabled,
                    loaded=self.gaussians.has_appearance_residual_loaded(),
                )
            ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        with open(os.path.join(point_cloud_path, "appearance_residual_meta.json"), "w") as meta_f:
            json.dump(self.gaussians.get_appearance_meta(), meta_f, indent=2)
        if self.gaussians.appearance_residual_enabled and self.gaussians.app_w_rgb is not None:
            torch.save(
                {
                    "app_w_rgb": self.gaussians.app_w_rgb.detach(),
                    "app_b_rgb": self.gaussians.app_b_rgb.detach(),
                    "app_w_gate": self.gaussians.app_w_gate.detach(),
                    "app_b_gate": self.gaussians.app_b_gate.detach(),
                    "app_w_diff": self.gaussians.app_w_diff.detach(),
                    "app_b_diff": self.gaussians.app_b_diff.detach(),
                    "app_w_spec": self.gaussians.app_w_spec.detach(),
                    "app_b_spec": self.gaussians.app_b_spec.detach(),
                    "app_w_mask": self.gaussians.app_w_mask.detach(),
                    "app_b_mask": self.gaussians.app_b_mask.detach(),
                    "app_w2_gate": self.gaussians.app_w2_gate.detach() if self.gaussians.app_w2_gate is not None else None,
                    "app_b2_gate": self.gaussians.app_b2_gate.detach() if self.gaussians.app_b2_gate is not None else None,
                    "app_w2_diff": self.gaussians.app_w2_diff.detach() if self.gaussians.app_w2_diff is not None else None,
                    "app_b2_diff": self.gaussians.app_b2_diff.detach() if self.gaussians.app_b2_diff is not None else None,
                    "app_w2_spec": self.gaussians.app_w2_spec.detach() if self.gaussians.app_w2_spec is not None else None,
                    "app_b2_spec": self.gaussians.app_b2_spec.detach() if self.gaussians.app_b2_spec is not None else None,
                    "app_w2_mask": self.gaussians.app_w2_mask.detach() if self.gaussians.app_w2_mask is not None else None,
                    "app_b2_mask": self.gaussians.app_b2_mask.detach() if self.gaussians.app_b2_mask is not None else None,
                },
                os.path.join(point_cloud_path, "appearance_residual.pth"),
            )
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]



