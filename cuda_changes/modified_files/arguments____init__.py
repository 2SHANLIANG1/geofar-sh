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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.depth_lambda = 0.0
        self.random_background = False
        self.optimizer_type = "default"
        self.use_fastkan_residual = False
        self.use_mlp_residual = False
        self.use_appearance_residual = False
        self.fastkan_hidden_dim = 16
        self.fastkan_num_layers = 2
        self.fastkan_num_basis = 8
        self.fastkan_basis_gamma = 4.0
        self.residual_scale = 0.05
        self.appearance_residual_enable_step = 7000
        self.appearance_residual_warmup_steps = 5000
        self.appearance_residual_schedule = "linear"
        self.lambda_residual_reg = 1e-4
        self.lambda_gate_reg = 0.0
        self.lambda_smooth_reg = 0.0
        self.lr_fastkan = 0.0001
        self.lr_fastkan_gate = 0.00005
        self.disable_residual_gate = False
        self.appearance_smooth_epsilon = 0.03
        self.appearance_latent_dim = 8
        self.appearance_gate_floor = 0.02
        self.appearance_stage2_start = 15000
        self.appearance_stage2_iters = 7000
        self.disable_stage2 = False
        self.appearance_freeze_geometry_in_stage2 = True
        self.appearance_disable_densify_in_stage2 = True
        self.appearance_detach_xyz_grad = True
        self.appearance_detach_shape_grad = True
        self.appearance_lambda_max = 1.0
        self.appearance_lambda_warmup_iters = 5000
        self.lr_appearance_latent = 0.0005
        self.appearance_profile = False
        self.stage2_joint_refine = False
        self.stage2_refine_sh = False
        self.stage2_lr_xyz = 1.6e-6
        self.stage2_lr_f_dc = 2.5e-5
        self.stage2_lr_f_rest = 1.25e-6
        self.disable_stage2_refine_opacity = False
        self.disable_stage2_refine_scale = False
        self.disable_stage2_refine_rotation = False
        self.stage2_geom_unfreeze_iter = 1500
        # Paper main method: CUDA-fused appearance residual + DDSR.
        self.use_local_aniso_encoding = False
        self.use_decoupled_residual = True
        self.disable_decoupled_residual = False
        self.lambda_spec_mask_reg = 0.0
        self.spec_mask_entropy_reg = 0.0
        self.spec_mask_temperature = 2.0
        self.disable_global_gate = False
        self.use_two_layer_ddsr_heads = False
        self.disable_two_layer_ddsr_heads = False
        self.ddsr_head_hidden_dim = 8
        self.ddsr_head_activation = "silu"
        self.appearance_head_hidden_dim = 8
        self.appearance_head_num_layers = 1
        self.appearance_head_activation = "silu"
        self.lambda_diff_consistency = 0.0001
        self.lambda_branch_diversity = 0.000005
        # Paper evidence / ablation aliases. String form allows explicit true/false
        # without changing the parser behavior of existing boolean flags.
        self.freeze_xyz = "auto"
        self.freeze_scaling = "auto"
        self.freeze_rotation = "auto"
        self.freeze_opacity = "auto"
        self.freeze_exposure = "auto"
        self.disable_densify_stage2 = "auto"
        self.disable_prune_stage2 = "auto"
        self.enable_sh_refine = "auto"
        self.enable_appearance_residual = "auto"
        self.enable_diffuse_residual = "auto"
        self.enable_specular_residual = "auto"
        self.enable_specular_mask = "auto"
        self.enable_global_gate = "auto"
        self.residual_mode = "auto"
        self.lambda_mask_reg = -1.0
        self.lambda_mask_entropy = -1.0
        self.lambda_diffuse_consistency = -1.0
        self.appearance_compute_mode = "auto"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    provided_dests = set()
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    for token in cmdlne_string:
        if token in option_to_dest:
            provided_dests.add(option_to_dest[token])

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if k in provided_dests or k not in merged_dict:
            merged_dict[k] = v
    return Namespace(**merged_dict)



