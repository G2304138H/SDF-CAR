import os
import json
import math
import time
import yaml
import torch
import numpy as np
import os.path as osp
from datetime import datetime, timezone
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .network import get_network
from .encoder import get_encoder
from src.render import run_network
from .dataset import TIGREDataset as Dataset
from .dataset.stage2_npz import (
    binary_mask_dice,
    embed_roi_mask_in_reference_grid,
    load_stage2_projection_case,
    validate_view_indices,
)

from src.render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1)/2)

    x = (m[2,1] - m[1,2])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2] - m[2,0])**2 + (m[1,0] -m[0,1])**2)
    y = (m[0,2] - m[2,0])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    z = (m[1,0] - m[0,1])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    axis=(x,y,z)

    return axis, angle

class Trainer:
    def __init__(self, cfg, device="cuda"):
        initialization_started = time.perf_counter()
        self.case_started_at_utc = datetime.now(timezone.utc).isoformat()

        # Args
        self.conf = cfg
        self.epochs = cfg["train"]["epoch"]
        self.netchunk = cfg["render"]["netchunk"]
        
        # Memory optimization settings
        self.use_mixed_precision = cfg.get("train", {}).get("mixed_precision", True)
        self.memory_efficient_eval = cfg.get("train", {}).get("memory_efficient_eval", True)
        
        # Initialize AMP scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_mixed_precision)
        
        # Setup output directories for batch processing
        base_output_dir = cfg["exp"].get("output_recon_dir", "./logs/reconstructions/")
        self.current_model_id = cfg["exp"].get("current_model_id", 1)
        
        # Create nested folder structure: base_dir/model_id/experiment_name/
        original_model_id = str(self.current_model_id).split('_')[0]
        experiment_name = str(self.current_model_id)
        self.output_recon_dir = osp.join(base_output_dir, original_model_id, experiment_name)
        
        os.makedirs(self.output_recon_dir, exist_ok=True)

        # Load CT geometry configuration
        configPath = cfg['exp']['dataconfig']
        with open(configPath, "r") as handle:
            data = yaml.safe_load(handle)

        # The original release only supports a GT-volume path and synthesizes
        # its two targets from that volume.  ``projection_npz`` activates the
        # actual paper use case: optimize from external 2D masks and cameras.
        projection_npz = cfg["exp"].get("projection_npz")
        self.external_projection_case = None
        self.external_view_indices = None
        self.reference_volume_npz = cfg["exp"].get("reference_volume_npz")
        self.visualization_outputs = []
        self.visualization_errors = []
        self.binary_projection_targets = False
        self.projection_attenuation = float(
            cfg.get("train", {}).get("projection_attenuation", 1.0)
        )
        self.minimum_gradient_norm = float(
            cfg.get("train", {}).get("minimum_gradient_norm", 1.0e-12)
        )
        self.zero_gradient_patience = int(
            cfg.get("train", {}).get("zero_gradient_patience", 20)
        )
        self.sdf_distance_epsilon = float(
            cfg.get("train", {}).get("sdf_distance_epsilon", 1.0e-6)
        )
        self.sdf_loss_warmup_epochs = int(
            cfg.get("train", {}).get("sdf_loss_warmup_epochs", 100)
        )
        self.sdf_loss_ramp_epochs = int(
            cfg.get("train", {}).get("sdf_loss_ramp_epochs", 400)
        )
        if self.minimum_gradient_norm < 0.0:
            raise ValueError("train.minimum_gradient_norm must be non-negative.")
        if self.zero_gradient_patience <= 0:
            raise ValueError("train.zero_gradient_patience must be positive.")
        if (
            not math.isfinite(self.sdf_distance_epsilon)
            or self.sdf_distance_epsilon <= 0.0
            or self.sdf_distance_epsilon >= 0.5
        ):
            raise ValueError(
                "train.sdf_distance_epsilon must be finite and strictly "
                "between 0 and 0.5."
            )
        if self.sdf_loss_warmup_epochs < 0:
            raise ValueError("train.sdf_loss_warmup_epochs must be non-negative.")
        if self.sdf_loss_ramp_epochs < 0:
            raise ValueError("train.sdf_loss_ramp_epochs must be non-negative.")
        self.current_epoch = 0
        self.dead_gradient_epochs = 0
        self.trainer_stability_fixes = [
            "positive configurable SDF output bias prevents a dense 0.5-occupancy initialization",
            "ODL projection and binary ray-to-mask exponential run in FP32",
            "Kornia distance-transform inputs are epsilon-bounded to prevent log(0) NaN gradients",
            "projection-only warm-up and an SDF-weight ramp isolate and stabilize the two loss paths",
            "2D SDF training loss fails fast unless differentiable Kornia is installed",
            "loss components, tensor ranges, gradient norm, AMP skips, and parameter updates are logged",
            "non-finite loss or gradients skip the optimizer step so parameters are not corrupted",
            "training aborts after consecutive dead-gradient epochs instead of silently wasting the run",
        ]
        self.training_log_path = osp.join(
            self.output_recon_dir,
            f"training_log_{self.current_model_id}.jsonl",
        )

        if projection_npz:
            if (
                cfg.get("projection_generation")
                and not osp.exists(osp.expanduser(str(projection_npz)))
            ):
                raise FileNotFoundError(
                    f"Generated projection NPZ not found: {projection_npz}. "
                    "Run `python generate_2d_projections.py --config "
                    "<the-same-case-yaml>` before train.py."
                )
            source_origin_distance_m = float(
                cfg["exp"].get("source_origin_distance_m", 0.75)
            )
            case = load_stage2_projection_case(
                projection_npz,
                source_origin_distance_m=source_origin_distance_m,
            )
            self.binary_projection_targets = case.is_binary_mask
            view_indices = validate_view_indices(
                cfg["exp"].get("view_indices", [0, 1]), case.num_views
            )
            if view_indices.size != 2:
                raise ValueError(
                    "This SDF-CAR trainer currently requires exactly two external "
                    f"views, got {view_indices.tolist()}."
                )

            detector_h, detector_w = case.detector_shape
            volume_size = np.asarray(
                cfg["exp"].get("reconstruction_nVoxel", data["nVoxel"]),
                dtype=np.int64,
            )
            if volume_size.shape == ():
                volume_size = np.repeat(volume_size, 3)
            if volume_size.shape != (3,) or np.any(volume_size <= 0):
                raise ValueError(
                    "exp.reconstruction_nVoxel must be a positive scalar or XYZ triple."
                )
            volume_extent_m = cfg["exp"].get("reconstruction_extent_m")
            if volume_extent_m is None:
                volume_extent_m = case.isocenter_fov_m
            extent_m = np.asarray(volume_extent_m, dtype=np.float64)
            if extent_m.shape == ():
                extent_m = np.repeat(extent_m, 3)
            if extent_m.shape != (3,) or np.any(extent_m <= 0.0):
                raise ValueError(
                    "exp.reconstruction_extent_m must be a positive scalar or XYZ triple."
                )

            data.update({
                "numTrain": 2,
                "DSD": [case.sid_m * 1000.0] * 2,
                "DSO": [case.source_origin_distance_m * 1000.0] * 2,
                "DDE": [case.detector_origin_distance_m * 1000.0] * 2,
                "nDetector": [detector_h, detector_w],
                "dDetector": [case.detector_pixel_spacing_m * 1000.0] * 2,
                "nVoxel": volume_size.astype(int).tolist(),
                "dVoxel": (extent_m * 1000.0 / volume_size).tolist(),
            })
            self.external_projection_case = case
            self.external_view_indices = view_indices
            print(f"External projection NPZ: {case.path}")
            print(
                "Projection representation: "
                f"{case.projection_representation}"
            )
            print(f"Selected views: {view_indices.tolist()}")
            equivalent_angles = case.sdfcar_projection_angles_deg(view_indices)
            selected_source, selected_detector, _, _ = case.camera_frames(view_indices)
            selected_rays = selected_detector - selected_source
            selected_rays /= np.linalg.norm(selected_rays, axis=1, keepdims=True)
            for view_position, index in enumerate(view_indices):
                print(
                    f"  {int(index)}: {case.clinical_views[index]} | "
                    f"theta={case.theta_deg[index]:g}, phi={case.phi_deg[index]:g} | "
                    "SDF-CAR direction pair="
                    f"{equivalent_angles[view_position].tolist()} | "
                    "ODL source-to-detector ray="
                    f"{selected_rays[view_position].astype(float).tolist()}"
                )

        # Setup data paths from main config (CCTA.yaml) - much cleaner!
        input_data_dir = cfg["exp"].get("input_data_dir", "./data/GT_volumes/")
        
        # Extract original model ID from experiment name (format: {model_id}_lr{lr}_loss{type})
        original_model_id = str(self.current_model_id).split('_')[0]
        gt_volume_filename = f"{original_model_id}.npy"
        gt_volume_path = osp.join(input_data_dir, gt_volume_filename)
        
        print(f"Processing experiment {self.current_model_id}")
        print(f"Original model ID: {original_model_id}")
        if self.external_projection_case is None:
            print(f"GT volume path: {gt_volume_path}")
            if not os.path.exists(gt_volume_path):
                raise FileNotFoundError(f"Ground truth volume not found: {gt_volume_path}")
        else:
            print("GT volume is not used by optimization.")

        dsd = data["DSD"] # Distance Source Detector   mm   
        dso = data["DSO"] # Distance Source Origin      mm 
        dde = data["DDE"]

        # Detector parameters
        proj_size = np.array(data["nDetector"])  # number of pixels              (px)
        proj_reso = np.array(data["dDetector"]) 

        # Image parameters
        image_size = np.array(data["nVoxel"])  # number of voxels              (vx)
        image_reso = np.array(data["dVoxel"])  # size of each voxel            (mm)
   
        if self.external_projection_case is None:
            first_proj_angle = [-data["first_projection_angle"][1], data["first_projection_angle"][0]]
            second_proj_angle = [-data["second_projection_angle"][1], data["second_projection_angle"][0]]
        
        # Apply camera noise if enabled
        if (self.external_projection_case is None and
                cfg.get("train", {}).get("camera_noise_enabled", False)):
            pos_noise_std = cfg.get("train", {}).get("camera_position_noise_std", 0.0)
            ori_noise_std = cfg.get("train", {}).get("camera_orientation_noise_std", 0.0)
            
            if pos_noise_std > 0:
                dso[0] += np.random.normal(0, pos_noise_std)
                dso[1] += np.random.normal(0, pos_noise_std)
            
            if ori_noise_std > 0:
                first_proj_angle[0] += np.random.normal(0, ori_noise_std)
                first_proj_angle[1] += np.random.normal(0, ori_noise_std)
                second_proj_angle[0] += np.random.normal(0, ori_noise_std)
                second_proj_angle[1] += np.random.normal(0, ori_noise_std)

        if self.external_projection_case is not None:
            source, detector, u_axis, v_axis = (
                self.external_projection_case.camera_frames(self.external_view_indices)
            )
            projectors = []
            for view_position in range(2):
                source_to_detector = detector[view_position] - source[view_position]
                source_to_detector /= np.linalg.norm(source_to_detector)
                # ODL detector coordinates follow output array [row, column].
                # Stage-2 rows increase along v and columns increase along u.
                detector_axes = [v_axis[view_position], u_axis[view_position]]
                projectors.append(ConeBeam3DProjector(
                    image_size, image_reso, 0.0, u_axis[view_position],
                    proj_size, proj_reso, dde[view_position], dso[view_position],
                    src_to_det_init=source_to_detector,
                    det_axes_init=detector_axes,
                ))
            self.ct_projector_first, self.ct_projector_second = projectors
            selected_masks = self.external_projection_case.images[
                self.external_view_indices
            ]
            data["projections"] = torch.tensor(
                selected_masks[None, ...], dtype=torch.float32, device=device
            )
            train_projs_one = data["projections"][:, 0:1]
            train_projs_two = data["projections"][:, 1:2]
            print(f"Loaded external projections: {data['projections'].shape}")
        else:
            # First projection.
            from_source_vec = (0, -dso[0], 0)
            from_rot_vec = (-1, 0, 0)
            to_source_vec = axis_rotation((0, 0, 1), angle=first_proj_angle[0] / 180 * np.pi, vectors=from_source_vec)
            to_rot_vec = axis_rotation((0, 0, 1), angle=first_proj_angle[0] / 180 * np.pi, vectors=from_rot_vec)
            to_source_vec = axis_rotation(to_rot_vec[0], angle=first_proj_angle[1] / 180 * np.pi, vectors=to_source_vec[0])
            rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
            proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
            self.ct_projector_first = ConeBeam3DProjector(
                image_size, image_reso, proj_angle, proj_axis, proj_size,
                proj_reso, dde[0], dso[0]
            )

            # Second projection.
            from_source_vec = (0, -dso[1], 0)
            from_rot_vec = (-1, 0, 0)
            to_source_vec = axis_rotation((0, 0, 1), angle=second_proj_angle[0] / 180 * np.pi, vectors=from_source_vec)
            to_rot_vec = axis_rotation((0, 0, 1), angle=second_proj_angle[0] / 180 * np.pi, vectors=from_rot_vec)
            to_source_vec = axis_rotation(to_rot_vec[0], angle=second_proj_angle[1] / 180 * np.pi, vectors=to_source_vec[0])
            rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
            proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
            self.ct_projector_second = ConeBeam3DProjector(
                image_size, image_reso, proj_angle, proj_axis, proj_size,
                proj_reso, dde[1], dso[1]
            )

            # Legacy/reproduction path: synthesize targets from a 3D GT volume.
            phantom = np.load(gt_volume_path)
            phantom = np.transpose(phantom, (1, 2, 0))[::, ::-1, ::-1]
            phantom = np.transpose(phantom, (2, 1, 0))[::-1, ::, ::].copy()
            phantom = torch.tensor(phantom, dtype=torch.float32, device=device)[None, ...]
            train_projs_one = self.ct_projector_first.forward_project(phantom)
            train_projs_two = self.ct_projector_second.forward_project(phantom)
            data["projections"] = torch.cat((train_projs_one, train_projs_two), 1)
            print(f"Generated projections from 3D volume: {data['projections'].shape}")
        
        # Dataset preparation based on mode
        self.use_sdf = cfg.get("train", {}).get("use_sdf", True)
        
        # Always generate 2D SDF targets from ground truth occupancy projections
        # (needed for SDF loss computation regardless of use_sdf mode)
        from src.render.sdf_utils import occupancy_to_sdf_2d
        proj_sdf_one = occupancy_to_sdf_2d(train_projs_one.squeeze(0).squeeze(0), voxel_size=proj_reso[0], use_kornia=False)  # [512, 512]
        proj_sdf_two = occupancy_to_sdf_2d(train_projs_two.squeeze(0).squeeze(0), voxel_size=proj_reso[1], use_kornia=False)  # [512, 512]
        data["sdf_projections"] = torch.cat((proj_sdf_one[None, None, :], proj_sdf_two[None, None, :]), 1)  # [1, 2, 512, 512]
        print(f"Generated 2D SDF targets from GT occupancy projections: {data['sdf_projections'].shape}")

        # Dataset
        self.dataconfig = data
        self.train_dset = Dataset(data, device)
        self.voxels = self.train_dset.voxels
        
        # Set last_activation based on use_sdf
        cfg["network"]["use_sdf"] = True if self.use_sdf else False
            
        # Load loss weights from config (respect user's settings regardless of use_sdf)
        self.loss_weights = cfg.get("train", {}).get("current_loss_weights", [1.0, 1.0])
        self.projection_weight, self.sdf_loss_weight = (
            float(weight) for weight in self.loss_weights
        )
        if self.projection_weight < 0.0 or self.sdf_loss_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if self.projection_weight == 0.0 and self.sdf_loss_weight == 0.0:
            raise ValueError("At least one training loss weight must be positive.")
        if self.sdf_loss_weight > 0.0:
            from src.render.sdf_utils import (
                require_differentiable_distance_transform,
            )

            require_differentiable_distance_transform()
        
        print(f"SDF Mode: {self.use_sdf}")
        print(f"Loss Weights - Projection: {self.projection_weight}, SDF: {self.sdf_loss_weight}")

        # Network
        net_type = cfg["network"]["net_type"]
        network = get_network(net_type)
        encoder = get_encoder(**cfg["encoder"])
        network_config = {k: v for k, v in cfg["network"].items() if k != "net_type"}
        self.net = network(encoder, **network_config).to(device)
        self.grad_vars = list(self.net.parameters())

        # Optimizer with memory-efficient settings
        weight_decay_val = cfg["train"].get("weight_decay", 1e-6)
        if isinstance(weight_decay_val, str):
            weight_decay_val = float(weight_decay_val)
            
        self.optimizer = torch.optim.AdamW(
            params=self.grad_vars, 
            lr=cfg["train"]["lrate"], 
            betas=(0.9, 0.999),
            weight_decay=weight_decay_val,  # Ensure proper type conversion
            eps=1e-8
        )

        self.training_losses = []
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None

        self._synchronize_cuda_for_timing()
        self.initialization_time_seconds = (
            time.perf_counter() - initialization_started
        )

    @staticmethod
    def _synchronize_cuda_for_timing():
        """Wait for queued GPU work before reading a wall-clock boundary."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def render_occupancy_projections(self, occupancy, *, return_raw=False):
        """Project a 3D occupancy field into the two configured detector views."""
        # ODL/ASTRA and the exponential silhouette map are numerically sensitive
        # in FP16. Keep this physical rendering path in FP32 while allowing the
        # MLP itself to use mixed precision.
        with torch.amp.autocast(device_type=occupancy.device.type, enabled=False):
            occupancy_fp32 = occupancy.float()
            projection_one = self.ct_projector_first.forward_project(
                occupancy_fp32
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            projection_two = self.ct_projector_second.forward_project(
                occupancy_fp32
            )
            raw_projections = torch.cat((projection_one, projection_two), dim=1)
        projections = raw_projections
        if self.binary_projection_targets:
            from src.render.sdf_utils import ray_integral_to_binary_probability

            projections = ray_integral_to_binary_probability(
                raw_projections, self.projection_attenuation
            )
        if return_raw:
            return projections, raw_projections
        return projections

    def _initialize_training_log(self):
        target_projections = self.train_dset.projs.detach().float()
        metadata = {
            "event": "trainer_stability_fixes",
            "case_id": str(self.current_model_id),
            "fixes": self.trainer_stability_fixes,
            "minimum_gradient_norm": self.minimum_gradient_norm,
            "zero_gradient_patience": self.zero_gradient_patience,
            "sdf_distance_epsilon": self.sdf_distance_epsilon,
            "sdf_loss_warmup_epochs": self.sdf_loss_warmup_epochs,
            "sdf_loss_ramp_epochs": self.sdf_loss_ramp_epochs,
            "mixed_precision_network": bool(self.use_mixed_precision),
            "sdf_initial_bias": float(
                self.conf.get("network", {}).get("sdf_initial_bias", 0.1)
            ),
            "sdf_alpha": float(
                self.conf.get("train", {}).get("sdf_alpha", 50.0)
            ),
            "projection_attenuation": self.projection_attenuation,
            "projection_representation": (
                "binary_mask" if self.binary_projection_targets
                else "line_integral_mm"
            ),
            "target_projection_min": float(target_projections.amin().item()),
            "target_projection_max": float(target_projections.amax().item()),
            "target_projection_mean": float(target_projections.mean().item()),
        }
        with open(self.training_log_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata) + "\n")
        print("Trainer stability fixes active:")
        for fix in self.trainer_stability_fixes:
            print(f"  - {fix}")
        print(f"Training diagnostics log: {self.training_log_path}")

    def _append_training_log(self, epoch, diagnostics):
        record = {"event": "epoch", "epoch": int(epoch), **diagnostics}
        with open(self.training_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def save_loss_plot(self):
        """
        Save training loss plot for current model.
        """
        if len(self.training_losses) == 0:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, 'b-', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title(f'Training Loss - Model {self.current_model_id}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        loss_plot_path = osp.join(self.output_recon_dir, f"loss_plot_{self.current_model_id}.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved loss plot: {loss_plot_path}")

    def start(self):
        """
        Main loop with memory optimizations.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._synchronize_cuda_for_timing()
        run_started = time.perf_counter()
        optimization_started = time.perf_counter()
        self._initialize_training_log()

        for idx_epoch in tqdm(range(1, self.epochs+1)):
            
            # Clear cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Train
            self.net.train()
            self.current_epoch = idx_epoch
            
            # Memory-efficient training step
            loss_train = self.train_step_memory_efficient(self.train_dset)
            
            # Track loss for plotting
            current_loss = loss_train['loss']
            self.training_losses.append(current_loss)
            self._append_training_log(idx_epoch, loss_train)

            print(
                f"epoch={idx_epoch}/{self.epochs}, loss={current_loss:.6f}, "
                f"projection={loss_train['projection_loss']:.6f}, "
                f"sdf2d={loss_train['sdf_2d_loss']:.6f}, "
                f"sdf_weight={loss_train['sdf_2d_effective_weight']:.4f}, "
                f"grad_norm={loss_train['gradient_norm']:.3e}, "
                f"param_delta={loss_train['parameter_probe_max_delta']:.3e}, "
                f"occupancy=[{loss_train['occupancy_min']:.3e},"
                f"{loss_train['occupancy_max']:.3e}], "
                f"raw_rays=[{loss_train['raw_projection_min']:.3e},"
                f"{loss_train['raw_projection_max']:.3e}], "
                f"pred=[{loss_train['projection_min']:.3e},"
                f"{loss_train['projection_max']:.3e}], "
                f"amp_step_skipped={loss_train['amp_step_skipped']}"
            )
            if (
                not np.isfinite(current_loss)
                or not loss_train["gradients_finite"]
            ):
                if loss_train["sdf_2d_effective_weight"] == 0.0:
                    failure_source = (
                        "Failure occurred during projection-only warm-up; "
                        "the Kornia SDF loss was not in the backward graph. "
                        "Inspect the ODL/ASTRA projection path"
                    )
                else:
                    failure_source = (
                        "Failure occurred after the geometric SDF loss was "
                        "enabled; inspect both loss components"
                    )
                raise RuntimeError(
                    "Non-finite loss or gradient detected. "
                    f"{failure_source}. Aborting immediately; "
                    f"inspect {self.training_log_path}."
                )
            if self.dead_gradient_epochs >= self.zero_gradient_patience:
                raise RuntimeError(
                    "Optimization has produced no usable gradient for "
                    f"{self.dead_gradient_epochs} consecutive epochs. Aborting "
                    "instead of continuing a constant-loss run. Inspect "
                    f"{self.training_log_path}."
                )

        self._synchronize_cuda_for_timing()
        self.optimization_time_seconds = (
            time.perf_counter() - optimization_started
        )
        self.mean_epoch_time_seconds = (
            self.optimization_time_seconds / self.epochs
            if self.epochs > 0 else 0.0
        )
        print(
            "Optimization time: "
            f"{self.optimization_time_seconds:.3f} s "
            f"({self.optimization_time_seconds / 60.0:.3f} min), "
            f"mean {self.mean_epoch_time_seconds:.4f} s/epoch"
        )
            
        # Save loss plot after training completion
        self.save_loss_plot()
        
        # Evaluate and save the last model at the end
        print(f"\nEvaluating last model from epoch {self.epochs}")
        
        # Evaluate last model (current state)
        self._synchronize_cuda_for_timing()
        evaluation_started = time.perf_counter()
        self.net.eval()
        with torch.no_grad():
            if self.memory_efficient_eval:
                eval_chunk_size = self.netchunk // 4
                model_pred = self.run_network_chunked(
                    self.voxels, 
                    self.net,
                    eval_chunk_size
                )
            else:
                model_pred = run_network(self.voxels, self.net, self.netchunk)
            
            model_pred = (model_pred.squeeze()).detach().cpu().numpy()

            self._synchronize_cuda_for_timing()
            self.final_evaluation_time_seconds = (
                time.perf_counter() - evaluation_started
            )
            self.total_compute_time_seconds = (
                self.initialization_time_seconds
                + self.optimization_time_seconds
                + self.final_evaluation_time_seconds
            )
            
            # Save last model results with all the comprehensive outputs
            self._save_best_model_results(model_pred, self.epochs)

        self._synchronize_cuda_for_timing()
        self.end_to_end_time_seconds = (
            self.initialization_time_seconds
            + time.perf_counter() - run_started
        )
        self.case_completed_at_utc = datetime.now(timezone.utc).isoformat()
        self._save_case_timing()
        
        tqdm.write(f"Training complete! Saved last model from epoch {self.epochs}")

    def _save_case_timing(self):
        """Write human-readable timing and environment metadata for this case."""
        timing = {
            "case_id": str(self.current_model_id),
            "status": "completed",
            "epochs": int(self.epochs),
            "started_at_utc": self.case_started_at_utc,
            "completed_at_utc": self.case_completed_at_utc,
            "initialization_time_seconds": self.initialization_time_seconds,
            "optimization_time_seconds": self.optimization_time_seconds,
            "optimization_time_minutes": self.optimization_time_seconds / 60.0,
            "mean_epoch_time_seconds": self.mean_epoch_time_seconds,
            "final_evaluation_time_seconds": self.final_evaluation_time_seconds,
            "total_compute_time_seconds": self.total_compute_time_seconds,
            "end_to_end_time_seconds": self.end_to_end_time_seconds,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "trainer_stability_fixes": self.trainer_stability_fixes,
            "training_diagnostics_log": self.training_log_path,
        }
        if torch.cuda.is_available():
            timing.update({
                "cuda_device": torch.cuda.get_device_name(),
                "peak_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated() / 1024 ** 3
                ),
            })
        if hasattr(self, "mask_dsc"):
            timing.update({
                "mask_dsc": self.mask_dsc,
                "prediction_foreground_voxels": (
                    self.prediction_foreground_voxels
                ),
                "reference_foreground_voxels": self.reference_foreground_voxels,
                "intersection_foreground_voxels": (
                    self.intersection_foreground_voxels
                ),
            })
        timing["visualization_outputs"] = self.visualization_outputs
        timing["visualization_errors"] = self.visualization_errors

        timing_path = osp.join(
            self.output_recon_dir,
            f"timing_{self.current_model_id}.json",
        )
        with open(timing_path, "w", encoding="utf-8") as handle:
            json.dump(timing, handle, indent=2)
            handle.write("\n")
        print(f"Saved case timing: {timing_path}")
        
    def _save_best_model_results(self, model_pred, epoch):
        """Save comprehensive results for the best model."""
        print(f"Saving best model results from epoch {epoch}...")
        
        # Handle SDF vs Occupancy modes differently
        pred_tensor = torch.tensor(model_pred, dtype=torch.float32, device=self.train_dset.projs.device)[None, ...]
        
        if self.use_sdf:
            # SDF mode: model generates SDF, we convert to occupancy for projections
            print("SDF mode: Model generated 3D SDF")
            sdf_3d_filename = f"sdf_3d_{self.current_model_id}.npy"
            sdf_3d_path = osp.join(self.output_recon_dir, sdf_3d_filename)
            np.save(sdf_3d_path, model_pred)
            print(f"Saved 3D SDF prediction: {sdf_3d_path}")
            
            # Convert SDF to occupancy for projections
            from src.render.sdf_utils import sdf_to_occupancy
            occupancy_for_proj = sdf_to_occupancy(
                pred_tensor, alpha=getattr(self, "sdf_alpha", 50.0)
            )
            
            # Save converted occupancy
            occupancy_3d_filename = f"recon_occupancy_{self.current_model_id}.npy"
            occupancy_3d_path = osp.join(self.output_recon_dir, occupancy_3d_filename)
            occupancy_3d_data = occupancy_for_proj.squeeze().detach().cpu().numpy()
            np.save(occupancy_3d_path, occupancy_3d_data)
            print(f"Saved occupancy converted from SDF: {occupancy_3d_path}")
            
        else:
            # Occupancy mode: model generates occupancy directly
            print("Occupancy mode: Model generated 3D occupancy")
            occupancy_3d_filename = f"recon_occupancy_{self.current_model_id}.npy"
            occupancy_3d_path = osp.join(self.output_recon_dir, occupancy_3d_filename)
            np.save(occupancy_3d_path, model_pred)
            print(f"Saved 3D occupancy prediction: {occupancy_3d_path}")
            
            occupancy_for_proj = pred_tensor
            occupancy_3d_data = model_pred

        if self.external_projection_case is not None:
            self._save_external_reconstruction_npz(model_pred, occupancy_3d_data)
        
        # Save all other comprehensive outputs (GT, projections, images, comparisons, network)
        self._save_comprehensive_outputs(pred_tensor, occupancy_for_proj, epoch)

    def _save_external_reconstruction_npz(self, sdf_roi, occupancy_roi):
        """Save ROI fields and a reference-grid binary volume in one NPZ."""
        case = self.external_projection_case
        source, detector, u_axis, v_axis = case.camera_frames(
            self.external_view_indices
        )
        source_to_detector = detector - source
        source_to_detector /= np.linalg.norm(
            source_to_detector, axis=1, keepdims=True
        )
        threshold = float(
            self.conf.get("exp", {}).get("output_occupancy_threshold", 0.5)
        )
        roi_mask = (np.asarray(occupancy_roi) >= threshold).astype(np.uint8)
        roi_spacing_mm = np.asarray(self.dataconfig["dVoxel"], dtype=np.float32)
        center_m = (
            np.zeros(3, dtype=np.float32)
            if case.projection_center_offset_m is None
            else case.projection_center_offset_m.astype(np.float32)
        )
        surface_prediction_mask = roi_mask
        surface_reference_mask = None
        surface_spacing_mm = roi_spacing_mm

        payload = {
            "sdf_roi_xyz": np.asarray(sdf_roi, dtype=np.float32),
            "occupancy_roi_xyz": np.asarray(occupancy_roi, dtype=np.float32),
            "roi_mask_xyz": roi_mask,
            "roi_spacing_mm": roi_spacing_mm,
            "roi_center_xyz_mm": center_m * 1000.0,
            "roi_axis_order": np.asarray("XYZ"),
            "view_indices": self.external_view_indices.astype(np.int32),
            "theta_deg": case.theta_deg[self.external_view_indices],
            "phi_deg": case.phi_deg[self.external_view_indices],
            "sdfcar_projection_angles_deg": (
                case.sdfcar_projection_angles_deg(self.external_view_indices)
            ),
            "clinical_views": case.clinical_views[self.external_view_indices],
            "input_images": case.images[self.external_view_indices],
            # Exact centred ODL camera frame. Detector array rows follow V and
            # columns follow U, matching the Stage-2 raster convention.
            "odl_source_xyz_m": source,
            "odl_detector_center_xyz_m": detector,
            "odl_source_to_detector_unit_xyz": source_to_detector,
            "odl_detector_row_axis_xyz": v_axis,
            "odl_detector_column_axis_xyz": u_axis,
            "projection_center_offset_m": center_m,
            "sid_m": np.asarray(case.sid_m, dtype=np.float32),
            "source_origin_distance_m": np.asarray(
                case.source_origin_distance_m, dtype=np.float32
            ),
            "imager_pixel_spacing_mm": np.asarray(
                case.detector_pixel_spacing_m * 1000.0, dtype=np.float32
            ),
            "occupancy_threshold": np.asarray(threshold, dtype=np.float32),
            "initialization_time_seconds": np.asarray(
                self.initialization_time_seconds, dtype=np.float64
            ),
            "optimization_time_seconds": np.asarray(
                self.optimization_time_seconds, dtype=np.float64
            ),
            "mean_epoch_time_seconds": np.asarray(
                self.mean_epoch_time_seconds, dtype=np.float64
            ),
            "final_evaluation_time_seconds": np.asarray(
                self.final_evaluation_time_seconds, dtype=np.float64
            ),
            "total_compute_time_seconds": np.asarray(
                self.total_compute_time_seconds, dtype=np.float64
            ),
            "source_projection_npz": np.asarray(str(case.path)),
        }

        if self.reference_volume_npz:
            if case.projection_center_offset_m is None:
                raise ValueError(
                    "reference_volume_npz output mapping requires "
                    "projection_center_offset in the projection NPZ."
                )
            with np.load(self.reference_volume_npz, allow_pickle=False) as reference:
                if "vol" not in reference.files or "spacing" not in reference.files:
                    raise KeyError(
                        "reference_volume_npz must contain 'vol' and 'spacing'."
                    )
                reference_vol = np.asarray(reference["vol"])
                reference_shape = np.asarray(reference_vol.shape, dtype=np.int32)
                reference_spacing = np.asarray(
                    reference["spacing"], dtype=np.float32
                ).reshape(3)
                full_mask = embed_roi_mask_in_reference_grid(
                    roi_mask,
                    roi_spacing_mm=roi_spacing_mm,
                    roi_center_xyz_mm=center_m * 1000.0,
                    reference_shape_xyz=reference_shape,
                    reference_spacing_xyz_mm=reference_spacing,
                )
                surface_prediction_mask = full_mask
                surface_reference_mask = reference_vol
                surface_spacing_mm = reference_spacing
                (
                    self.mask_dsc,
                    self.prediction_foreground_voxels,
                    self.reference_foreground_voxels,
                    self.intersection_foreground_voxels,
                ) = binary_mask_dice(full_mask, reference_vol)
                print(
                    "Reference-volume mask DSC: "
                    f"{self.mask_dsc:.6f} "
                    f"(prediction={self.prediction_foreground_voxels}, "
                    f"reference={self.reference_foreground_voxels}, "
                    f"intersection={self.intersection_foreground_voxels})"
                )
                payload.update({
                    "vol": full_mask,
                    "spacing": reference_spacing,
                    "vol_axis_order": np.asarray("XYZ"),
                    "reference_shape_xyz": reference_shape,
                    "mask_dsc": np.asarray(self.mask_dsc, dtype=np.float64),
                    "prediction_foreground_voxels": np.asarray(
                        self.prediction_foreground_voxels, dtype=np.int64
                    ),
                    "reference_foreground_voxels": np.asarray(
                        self.reference_foreground_voxels, dtype=np.int64
                    ),
                    "intersection_foreground_voxels": np.asarray(
                        self.intersection_foreground_voxels, dtype=np.int64
                    ),
                    "source_reference_npz": np.asarray(
                        str(osp.abspath(self.reference_volume_npz))
                    ),
                })
                if "source_nii" in reference.files:
                    payload["source_nii"] = np.asarray(reference["source_nii"])
        else:
            payload.update({
                "vol": roi_mask,
                "spacing": roi_spacing_mm,
                "vol_axis_order": np.asarray("XYZ"),
            })

        output_path = osp.join(
            self.output_recon_dir,
            f"reconstruction_{self.current_model_id}.npz",
        )
        np.savez_compressed(output_path, **payload)
        print(f"Saved combined NPZ reconstruction: {output_path}")
        self._save_external_surface_gifs(
            surface_prediction_mask,
            surface_reference_mask,
            surface_spacing_mm,
        )

    def _save_external_surface_gifs(
        self,
        prediction_mask,
        reference_mask,
        spacing_mm,
    ):
        visualization = self.conf.get("visualization", {})
        if not visualization.get("save_surface_gifs", True):
            return
        try:
            from src.reconstruction_visualization import save_mask_surface_gif

            frames = int(visualization.get("surface_gif_frames", 24))
            fps = int(visualization.get("surface_gif_fps", 5))
            max_faces = int(visualization.get("surface_gif_max_faces", 200_000))
            masks = []
            if reference_mask is not None:
                masks.append((
                    reference_mask,
                    "3d_ground_truth_surface.gif",
                    f"Case {self.current_model_id}: ground-truth surface",
                ))
            masks.append((
                prediction_mask,
                "3d_prediction_surface.gif",
                f"Case {self.current_model_id}: prediction surface",
            ))
            for mask, filename, title in masks:
                path = save_mask_surface_gif(
                    mask,
                    spacing_mm,
                    osp.join(self.output_recon_dir, filename),
                    title=title,
                    num_frames=frames,
                    fps=fps,
                    max_faces=max_faces,
                )
                self.visualization_outputs.append(str(path))
                print(f"Saved rotating surface GIF: {path}")
        except Exception as error:
            message = f"Surface GIF generation failed: {type(error).__name__}: {error}"
            self.visualization_errors.append(message)
            print(f"WARNING: {message}")

    def _save_external_projection_pngs(self, projections):
        visualization = self.conf.get("visualization", {})
        if not visualization.get("save_predicted_projection_pngs", True):
            return
        try:
            from src.reconstruction_visualization import (
                save_predicted_projection_pngs,
            )

            paths = save_predicted_projection_pngs(
                projections,
                self.external_view_indices,
                self.output_recon_dir,
            )
            self.visualization_outputs.extend(str(path) for path in paths)
            for path in paths:
                print(f"Saved predicted projection PNG: {path}")
        except Exception as error:
            message = (
                "Predicted projection PNG generation failed: "
                f"{type(error).__name__}: {error}"
            )
            self.visualization_errors.append(message)
            print(f"WARNING: {message}")

    def run_network_chunked(self, inputs, fn, chunk_size):
        """
        Memory-efficient network inference with smaller chunks.
        """
        uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        output_chunks = []
        
        for i in range(0, uvt_flat.shape[0], chunk_size):
            chunk = uvt_flat[i:i + chunk_size]
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
                chunk_output = fn(chunk)
            output_chunks.append(chunk_output)
            
            # Clear intermediate tensors
            del chunk, chunk_output
            if torch.cuda.is_available() and i % (chunk_size * 4) == 0:  # Periodic cleanup
                torch.cuda.empty_cache()
        
        out_flat = torch.cat(output_chunks, 0)
        out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
        
        # Clean up
        del output_chunks, out_flat
        
        return out

    def train_step_memory_efficient(self, data):
        """
        Memory-efficient training step with gradient accumulation and mixed precision.
        """
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        loss = self.compute_loss(data)
        diagnostics = {
            key: float(loss[key].detach().float().item())
            for key in (
                "loss",
                "projection_loss",
                "sdf_2d_loss",
                "sdf_2d_effective_weight",
                "occupancy_min",
                "occupancy_max",
                "raw_projection_min",
                "raw_projection_max",
                "projection_min",
                "projection_max",
            )
        }
        losses_finite = all(
            np.isfinite(diagnostics[key])
            for key in ("loss", "projection_loss", "sdf_2d_loss")
        )
        diagnostics.update({
            "loss_stage": (
                "projection_warmup"
                if diagnostics["sdf_2d_effective_weight"] == 0.0
                and self.sdf_loss_weight > 0.0
                else (
                    "projection_only"
                    if self.sdf_loss_weight == 0.0
                    else "hybrid"
                )
            ),
            "total_loss_finite": bool(np.isfinite(diagnostics["loss"])),
            "projection_loss_finite": bool(
                np.isfinite(diagnostics["projection_loss"])
            ),
            "sdf_2d_loss_finite": bool(
                np.isfinite(diagnostics["sdf_2d_loss"])
            ),
        })
        probe_parameter = self.grad_vars[-1]
        probe_before = probe_parameter.detach().float().clone()
        scale_before = float(self.scaler.get_scale())
        gradient_norms = []
        gradients_finite = bool(losses_finite)

        if losses_finite:
            # Backward pass with gradient scaling. Unscale before measuring the
            # real gradient or deciding whether the optimizer may modify the
            # model.
            self.scaler.scale(loss["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            for parameter in self.grad_vars:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                gradients_finite = gradients_finite and bool(
                    torch.isfinite(gradient).all().item()
                )
                gradient_norms.append(torch.linalg.vector_norm(gradient.float()))

        if gradient_norms:
            stacked_gradient_norms = torch.stack(gradient_norms)
            if torch.isfinite(stacked_gradient_norms).all():
                gradient_norm = float(
                    torch.linalg.vector_norm(stacked_gradient_norms).item()
                )
            else:
                gradient_norm = float("nan")
                gradients_finite = False
        else:
            gradient_norm = 0.0 if losses_finite else float("nan")

        optimizer_step_safe = (
            losses_finite
            and gradients_finite
            and np.isfinite(gradient_norm)
        )
        if optimizer_step_safe:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Never feed NaN gradients to AdamW. This is particularly important
        # when AMP is disabled, because there is then no GradScaler safeguard.
        scale_after = float(self.scaler.get_scale())
        parameter_probe_delta = float(
            torch.max(
                torch.abs(probe_parameter.detach().float() - probe_before)
            ).item()
        )
        amp_step_skipped = (
            not optimizer_step_safe
            or scale_after < scale_before
        )

        dead_gradient = (
            amp_step_skipped
            or gradient_norm <= self.minimum_gradient_norm
        )
        self.dead_gradient_epochs = (
            self.dead_gradient_epochs + 1 if dead_gradient else 0
        )

        diagnostics.update({
            "losses_finite": bool(losses_finite),
            "gradient_norm": gradient_norm,
            "gradients_finite": bool(gradients_finite),
            "parameter_probe_max_delta": parameter_probe_delta,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "amp_step_skipped": bool(amp_step_skipped),
            "dead_gradient_epochs": int(self.dead_gradient_epochs),
        })

        del loss, probe_before
        
        # Clear gradients and cache
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return diagnostics

    def train_step(self, data):
        """
        Legacy training step - kept for compatibility
        """
        return self.train_step_memory_efficient(data)
        
    def _save_comprehensive_outputs(self, sdf_pred_tensor, occupancy_pred, epoch):
        """Save comprehensive outputs including projections, images, and comparisons."""
        # The legacy path copies the synthetic source GT for reproduction.  In
        # direct-NPZ mode, the reference voxel file is evaluation/output-grid
        # metadata only and is never copied into the optimizer targets.
        if self.external_projection_case is None:
            gt_volume_filename = f"gt_volume_{self.current_model_id}.npy"
            gt_volume_path = osp.join(self.output_recon_dir, gt_volume_filename)
            input_data_dir = self.conf["exp"].get("input_data_dir", "./data/GT_volumes/")
            original_model_id = str(self.current_model_id).split('_')[0]
            original_gt_path = osp.join(input_data_dir, f"{original_model_id}.npy")
            gt_volume = np.load(original_gt_path)
            np.save(gt_volume_path, gt_volume)
            print(f"Saved ground truth 3D model: {gt_volume_path}")
        
        # Save ground truth projections
        gt_projs_filename = f"gt_projections_{self.current_model_id}.npy"
        gt_projs_path = osp.join(self.output_recon_dir, gt_projs_filename)
        gt_projs_data = self.train_dset.projs.detach().cpu().numpy()
        np.save(gt_projs_path, gt_projs_data)
        print(f"Saved ground truth projections: {gt_projs_path}")
        
        # Generate predicted projections
        pred_projs = self.render_occupancy_projections(occupancy_pred)
        
        pred_projs_filename = f"pred_projections_{self.current_model_id}.npy"
        pred_projs_path = osp.join(self.output_recon_dir, pred_projs_filename)
        pred_projs_data = pred_projs.detach().cpu().numpy()
        np.save(pred_projs_path, pred_projs_data)
        print(f"Saved predicted projections: {pred_projs_path}")
        if self.external_projection_case is not None:
            self._save_external_projection_pngs(pred_projs_data)
        
        # Always save ground truth 2D SDF (generated for all modes now)
        gt_sdf_2d_filename = f"sdf_2d_gt_{self.current_model_id}.npy"
        gt_sdf_2d_path = osp.join(self.output_recon_dir, gt_sdf_2d_filename)
        gt_sdf_2d_data = self.dataconfig["sdf_projections"].detach().cpu().numpy()
        np.save(gt_sdf_2d_path, gt_sdf_2d_data)
        print(f"Saved 2D SDF ground truth: {gt_sdf_2d_path}")
        
        # Generate and save 2D SDF predictions if model was trained with SDF loss
        if self.sdf_loss_weight > 0:
            detector_pixel_size = self.dataconfig["dDetector"][0]
            
            from src.render.sdf_utils import occupancy_to_sdf_2d
            sdf_2d_view1 = occupancy_to_sdf_2d(
                pred_projs[0, 0], voxel_size=detector_pixel_size,
                use_kornia=False
            )
            sdf_2d_view2 = occupancy_to_sdf_2d(
                pred_projs[0, 1], voxel_size=detector_pixel_size,
                use_kornia=False
            )
            pred_sdf_2d = torch.stack(
                [sdf_2d_view1, sdf_2d_view2], dim=0
            )[None, ...]
            
            pred_sdf_2d_filename = f"sdf_2d_pred_{self.current_model_id}.npy"
            pred_sdf_2d_path = osp.join(self.output_recon_dir, pred_sdf_2d_filename)
            pred_sdf_2d_data = pred_sdf_2d.detach().cpu().numpy()
            np.save(pred_sdf_2d_path, pred_sdf_2d_data)
            print(f"Saved 2D SDF predictions: {pred_sdf_2d_path}")
        else:
            # No SDF loss used during training
            pred_sdf_2d_data = None
            print("No SDF loss used - skipping 2D SDF prediction generation")
        
        # Create comparison images based on whether SDF loss was used
        comparison_path = osp.join(self.output_recon_dir, f"comparison_{self.current_model_id}.png")
        
        if self.sdf_loss_weight > 0 and pred_sdf_2d_data is not None:
            # SDF mode: show both occupancy and SDF projections
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(2):  # Two views
                # Ground truth occupancy projection
                axes[i, 0].imshow(gt_projs_data[0, i], cmap='gray')
                axes[i, 0].set_title(f'GT Occupancy View {i+1}')
                axes[i, 0].axis('off')
                
                # Predicted occupancy projection
                axes[i, 1].imshow(pred_projs_data[0, i], cmap='gray')
                axes[i, 1].set_title(f'Pred Occupancy View {i+1}')
                axes[i, 1].axis('off')
                
                # Ground truth SDF (if available)
                if gt_sdf_2d_data is not None:
                    axes[i, 2].imshow(gt_sdf_2d_data[0, i], cmap='RdBu_r')
                    axes[i, 2].set_title(f'GT SDF View {i+1}')
                    axes[i, 2].axis('off')
                else:
                    axes[i, 2].text(0.5, 0.5, 'No GT SDF', ha='center', va='center')
                    axes[i, 2].axis('off')
                
                # Predicted SDF
                axes[i, 3].imshow(pred_sdf_2d_data[0, i], cmap='RdBu_r')
                axes[i, 3].set_title(f'Pred SDF View {i+1}')
                axes[i, 3].axis('off')
        else:
            # Occupancy mode: only show occupancy projections
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            
            for i in range(2):  # Two views
                # Ground truth occupancy projection
                axes[i, 0].imshow(gt_projs_data[0, i], cmap='gray')
                axes[i, 0].set_title(f'GT Occupancy View {i+1}')
                axes[i, 0].axis('off')
                
                # Predicted occupancy projection
                axes[i, 1].imshow(pred_projs_data[0, i], cmap='gray')
                axes[i, 1].set_title(f'Pred Occupancy View {i+1}')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison image: {comparison_path}")
        
        # Save network weights
        network_filename = f"network_{self.current_model_id}.pth"
        network_path = osp.join(self.output_recon_dir, network_filename)
        torch.save({
            'network': self.net.state_dict(),
            'model_id': self.current_model_id,
            'epoch': epoch,
            'config': self.conf
        }, network_path)
        print(f"Saved last network: {network_path}")

    def compute_loss(self, data):
        """
        Training step
        """
        raise NotImplementedError()
