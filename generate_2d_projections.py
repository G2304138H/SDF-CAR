"""Generate SDF-CAR ray projections from a reference voxel NPZ and case YAML."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config.configloading import load_config
from src.dataset.stage2_npz import (
    extract_reference_grid_roi,
    load_stage2_projection_case,
    validate_view_indices,
)


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two-view ray-length training targets from the reference "
            "voxel volume using the exact cameras configured for SDF-CAR."
        )
    )
    parser.add_argument(
        "--config",
        default="./config/CCTA_npz_case1.yaml",
        help="Case YAML used by both this generator and train.py.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device used by the ODL/ASTRA projector, for example cuda or cuda:1.",
    )
    return parser


def _required_path(value: Any, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing required YAML setting: {label}.")
    return Path(str(value)).expanduser().resolve()


def _reconstruction_grid(
    config: dict[str, Any],
    camera_case,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp = config.get("exp", {})
    data_config_path = _required_path(exp.get("dataconfig"), "exp.dataconfig")
    with data_config_path.open("r", encoding="utf-8") as handle:
        base_data = yaml.safe_load(handle) or {}

    shape = np.asarray(
        exp.get("reconstruction_nVoxel", base_data.get("nVoxel")),
        dtype=np.int64,
    )
    if shape.shape == ():
        shape = np.repeat(shape, 3)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError(
            "exp.reconstruction_nVoxel must be a positive scalar or XYZ triple."
        )

    extent_m_value = exp.get("reconstruction_extent_m")
    if extent_m_value is None:
        extent_m_value = camera_case.isocenter_fov_m
    extent_m = np.asarray(extent_m_value, dtype=np.float64)
    if extent_m.shape == ():
        extent_m = np.repeat(extent_m, 3)
    if extent_m.shape != (3,) or np.any(extent_m <= 0.0):
        raise ValueError(
            "exp.reconstruction_extent_m must be a positive scalar or XYZ triple."
        )
    spacing_mm = extent_m * 1000.0 / shape
    return shape, extent_m, spacing_mm


def _forward_project_reference_roi(
    roi_mask_xyz: np.ndarray,
    roi_spacing_xyz_mm: np.ndarray,
    camera_case,
    device_name: str,
) -> np.ndarray:
    """Return one raw line-integral image for every camera in the metadata NPZ."""
    import torch

    from src.render.ct_geometry_projector import ConeBeam3DProjector

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Projection generation requires an NVIDIA CUDA GPU because this "
            "repository uses ODL's astra_cuda backend."
        )
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device.")
    if device.index is not None:
        torch.cuda.set_device(device.index)

    volume = torch.as_tensor(
        np.ascontiguousarray(roi_mask_xyz, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )[None, ...]
    source, detector, u_axis, v_axis = camera_case.camera_frames()
    projections = []
    for view_index in range(camera_case.num_views):
        source_to_detector = detector[view_index] - source[view_index]
        source_to_detector /= np.linalg.norm(source_to_detector)
        projector = ConeBeam3DProjector(
            np.asarray(roi_mask_xyz.shape, dtype=np.int64),
            np.asarray(roi_spacing_xyz_mm, dtype=np.float64),
            0.0,
            u_axis[view_index],
            np.asarray(camera_case.detector_shape, dtype=np.int64),
            np.repeat(camera_case.detector_pixel_spacing_m * 1000.0, 2),
            camera_case.detector_origin_distance_m * 1000.0,
            camera_case.source_origin_distance_m * 1000.0,
            src_to_det_init=source_to_detector,
            det_axes_init=[v_axis[view_index], u_axis[view_index]],
        )
        with torch.no_grad():
            projected = projector.forward_project(volume)
        projected_np = projected.detach().float().cpu().numpy().squeeze()
        if projected_np.shape != camera_case.detector_shape:
            raise RuntimeError(
                f"View {view_index} returned {projected_np.shape}; expected "
                f"{camera_case.detector_shape}."
            )
        projections.append(np.maximum(projected_np, 0.0).astype(np.float32))
        print(
            f"Projected view {view_index + 1}/{camera_case.num_views}: "
            f"max ray length={float(projected_np.max()):.6f} mm"
        )
        del projected, projector
        torch.cuda.empty_cache()
        gc.collect()
    return np.stack(projections, axis=0)


def _save_projection_pngs(
    images: np.ndarray,
    output_npz: Path,
    selected_view_indices: np.ndarray,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_dir = output_npz.parent / f"{output_npz.stem}_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    selected = set(int(index) for index in selected_view_indices)
    paths = []
    for view_index, image in enumerate(images):
        if view_index not in selected:
            continue
        path = png_dir / f"generated_projection_view_{view_index}.png"
        plt.imsave(
            path,
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )
        paths.append(path)
    return paths


def generate_projection_npz(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
) -> Path:
    exp = config.get("exp", {})
    generation = config.get("projection_generation", {})
    camera_npz = _required_path(
        generation.get("camera_metadata_npz"),
        "projection_generation.camera_metadata_npz",
    )
    reference_npz = _required_path(
        exp.get("reference_volume_npz"),
        "exp.reference_volume_npz",
    )
    output_npz = _required_path(exp.get("projection_npz"), "exp.projection_npz")
    if output_npz in {camera_npz, reference_npz}:
        raise ValueError(
            "exp.projection_npz is the generated destination and must not overwrite "
            "the camera metadata or reference-volume NPZ."
        )

    source_origin_distance_m = float(exp.get("source_origin_distance_m", 0.75))
    camera_case = load_stage2_projection_case(
        camera_npz,
        source_origin_distance_m=source_origin_distance_m,
    )
    selected_view_indices = validate_view_indices(
        exp.get("view_indices", [0, 1]), camera_case.num_views
    )
    if selected_view_indices.size != 2:
        raise ValueError("The current SDF-CAR trainer requires exactly two views.")
    if camera_case.projection_center_offset_m is None:
        raise ValueError(
            "Camera metadata NPZ must contain projection_center_offset so the "
            "reference voxel volume can be sampled in the same centred frame."
        )

    roi_shape, extent_m, roi_spacing_mm = _reconstruction_grid(config, camera_case)
    reference_threshold = float(generation.get("reference_volume_threshold", 0.0))
    with np.load(reference_npz, allow_pickle=False) as reference:
        if "vol" not in reference.files or "spacing" not in reference.files:
            raise KeyError("Reference NPZ must contain 'vol' and 'spacing'.")
        reference_mask = np.asarray(reference["vol"]) > reference_threshold
        reference_spacing_mm = np.asarray(
            reference["spacing"], dtype=np.float64
        ).reshape(3)

    roi_mask = extract_reference_grid_roi(
        reference_mask.astype(np.uint8),
        reference_spacing_xyz_mm=reference_spacing_mm,
        roi_shape_xyz=roi_shape,
        roi_spacing_xyz_mm=roi_spacing_mm,
        roi_center_xyz_mm=camera_case.projection_center_offset_m * 1000.0,
    )
    foreground_voxels = int(np.count_nonzero(roi_mask))
    if foreground_voxels == 0:
        raise ValueError(
            "The configured reconstruction ROI contains no reference foreground. "
            "Check projection_center_offset, voxel spacing, and XYZ axis order."
        )
    print(
        f"Reference ROI: shape={tuple(int(v) for v in roi_shape)}, "
        f"spacing_mm={roi_spacing_mm.tolist()}, foreground={foreground_voxels}"
    )

    line_integrals_mm = _forward_project_reference_roi(
        roi_mask,
        roi_spacing_mm,
        camera_case,
        device_name,
    )
    minimum_ray_length_mm = float(
        generation.get("minimum_ray_length_mm", 1.0e-6)
    )
    binary_images = (line_integrals_mm > minimum_ray_length_mm).astype(np.float32)
    selected_foreground = [
        int(np.count_nonzero(binary_images[index])) for index in selected_view_indices
    ]
    if any(count == 0 for count in selected_foreground):
        raise RuntimeError(
            "At least one selected generated projection is empty; refusing to save "
            f"an unusable training input. Counts: {selected_foreground}."
        )

    payload = {
        "sample_name": np.asarray(f"{camera_case.sample_name}_from_reference_voxel"),
        # Raw ray lengths match the original SDF-CAR synthetic-projection path.
        # Keeping these values avoids the saturated binary-silhouette transform.
        "images": line_integrals_mm,
        "binary_images": binary_images,
        "projection_representation": np.asarray("line_integral_mm"),
        "theta_deg": camera_case.theta_deg.astype(np.float32),
        "phi_deg": camera_case.phi_deg.astype(np.float32),
        "clinical_views": camera_case.clinical_views,
        "sid": np.asarray(camera_case.sid_m, dtype=np.float32),
        "imager_pixel_spacing": np.asarray(
            camera_case.detector_pixel_spacing_m * 1000.0, dtype=np.float32
        ),
        "imager_pixel_spacing_units": np.asarray("mm"),
        "projection_center_offset": camera_case.projection_center_offset_m,
        "source_view_indices": np.arange(camera_case.num_views, dtype=np.int32),
        "selected_view_indices": selected_view_indices.astype(np.int32),
        "projection_line_integrals_mm": line_integrals_mm,
        "reference_roi_mask_xyz": roi_mask.astype(np.uint8),
        "reference_roi_spacing_xyz_mm": roi_spacing_mm.astype(np.float32),
        "reference_roi_extent_xyz_m": extent_m.astype(np.float32),
        "source_camera_metadata_npz": np.asarray(str(camera_npz)),
        "source_reference_volume_npz": np.asarray(str(reference_npz)),
        "generation_config_json": np.asarray(json.dumps({
            "reference_volume_threshold": reference_threshold,
            "minimum_ray_length_mm": minimum_ray_length_mm,
            "projector": "ODL ConeBeamGeometry / astra_cuda",
        }, sort_keys=True)),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_npz.with_suffix(output_npz.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary_path, output_npz)
    png_paths = _save_projection_pngs(
        binary_images,
        output_npz,
        selected_view_indices,
    )
    print(f"Saved generated training projection NPZ: {output_npz}")
    for path in png_paths:
        print(f"Saved generated projection check: {path}")
    print(
        "Selected-view foreground pixels: "
        + ", ".join(
            f"view {int(index)}={count}"
            for index, count in zip(selected_view_indices, selected_foreground)
        )
    )
    return output_npz


def main() -> None:
    args = config_parser().parse_args()
    config = load_config(args.config)
    generate_projection_npz(config, device_name=args.device)


if __name__ == "__main__":
    main()
