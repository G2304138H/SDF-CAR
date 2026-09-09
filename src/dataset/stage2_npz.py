"""Stage-2 projection NPZ input and full-volume output helpers.

The projection files used by this project describe a general cone-beam camera
orientation with ``theta_deg`` and ``phi_deg``.  The helpers below reproduce the
camera construction used by ``vessel_tree_generator`` without importing that
repository or requiring PyTorch/ODL.  Coordinates are XYZ and camera distances
are stored in metres in the source NPZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


DEFAULT_SOURCE_ORIGIN_DISTANCE_M = 0.75


@dataclass(frozen=True)
class Stage2ProjectionCase:
    path: Path
    sample_name: str
    images: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    sid_m: float
    source_origin_distance_m: float
    detector_origin_distance_m: float
    detector_pixel_spacing_m: float
    clinical_views: np.ndarray
    projection_center_offset_m: Optional[np.ndarray]
    projection_representation: str

    @property
    def num_views(self) -> int:
        return int(self.images.shape[0])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return int(self.images.shape[1]), int(self.images.shape[2])

    @property
    def is_binary_mask(self) -> bool:
        return self.projection_representation == "binary_mask"

    @property
    def isocenter_fov_m(self) -> float:
        detector_width_m = self.detector_pixel_spacing_m * self.detector_shape[1]
        return detector_width_m * self.source_origin_distance_m / self.sid_m

    def camera_frames(
        self, view_indices: Optional[Sequence[int]] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = (
            np.arange(self.num_views, dtype=np.int64)
            if view_indices is None
            else validate_view_indices(view_indices, self.num_views)
        )
        return stage2_angles_to_camera_frames(
            theta_deg=self.theta_deg[indices],
            phi_deg=self.phi_deg[indices],
            sid_m=self.sid_m,
            source_origin_distance_m=self.source_origin_distance_m,
        )

    def cone_vectors(self, view_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        source, detector, u_axis, v_axis = self.camera_frames(view_indices)
        spacing = self.detector_pixel_spacing_m
        return np.asarray(
            np.concatenate(
                [source, detector, u_axis * spacing, v_axis * spacing], axis=1
            ),
            dtype=np.float32,
        )

    def sdfcar_projection_angles_deg(
        self, view_indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Return direction-equivalent pairs used by the SDF-CAR YAML.

        These pairs preserve the central ray.  The complete ODL camera still
        needs the detector U/V axes returned by :meth:`camera_frames` in order
        to preserve the input image's in-plane orientation.
        """
        indices = (
            np.arange(self.num_views, dtype=np.int64)
            if view_indices is None
            else validate_view_indices(view_indices, self.num_views)
        )
        return stage2_angles_to_sdfcar_angles(
            self.theta_deg[indices], self.phi_deg[indices]
        )


def _scalar(npz: np.lib.npyio.NpzFile, key: str) -> object:
    return np.asarray(npz[key]).reshape(()).item()


def validate_view_indices(view_indices: Sequence[int], num_views: int) -> np.ndarray:
    indices = np.asarray(tuple(int(index) for index in view_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("At least one view index is required.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"View indices must be unique, got {indices.tolist()}.")
    if int(indices.min()) < 0 or int(indices.max()) >= int(num_views):
        raise IndexError(
            f"View indices {indices.tolist()} are outside [0, {int(num_views) - 1}]."
        )
    return indices


def load_stage2_projection_case(
    path: str | Path,
    *,
    source_origin_distance_m: float = DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
    fallback_detector_pixel_spacing_mm: Optional[float] = None,
    fallback_sid_m: Optional[float] = None,
) -> Stage2ProjectionCase:
    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"images", "theta_deg", "phi_deg"}
        missing = sorted(required.difference(data.files))
        if "sid" not in data.files and fallback_sid_m is None:
            missing.append("sid")
        if (
            "imager_pixel_spacing" not in data.files
            and fallback_detector_pixel_spacing_mm is None
        ):
            missing.append("imager_pixel_spacing")
        if missing:
            raise KeyError(f"{npz_path} is missing required keys: {missing}.")

        images = np.asarray(data["images"], dtype=np.float32)
        theta_deg = np.asarray(data["theta_deg"], dtype=np.float32).reshape(-1)
        phi_deg = np.asarray(data["phi_deg"], dtype=np.float32).reshape(-1)
        if images.ndim != 3:
            raise ValueError(f"images must have shape [V,H,W], got {images.shape}.")
        if images.shape[0] != theta_deg.size or theta_deg.shape != phi_deg.shape:
            raise ValueError(
                "images, theta_deg, and phi_deg disagree on the number of views: "
                f"{images.shape[0]}, {theta_deg.shape}, {phi_deg.shape}."
            )
        if not np.isfinite(images).all():
            raise ValueError("Projection images contain NaN or infinity.")
        if not np.isfinite(theta_deg).all() or not np.isfinite(phi_deg).all():
            raise ValueError("Projection angles contain NaN or infinity.")

        sid_m = (
            float(_scalar(data, "sid"))
            if "sid" in data.files
            else float(fallback_sid_m)
        )
        source_distance_m = float(source_origin_distance_m)
        detector_distance_m = sid_m - source_distance_m
        if sid_m <= 0.0 or source_distance_m <= 0.0 or detector_distance_m <= 0.0:
            raise ValueError(
                "Expected SID > source-to-isocentre distance > 0, got "
                f"SID={sid_m} m and source distance={source_distance_m} m."
            )

        if "imager_pixel_spacing" in data.files:
            spacing_value = float(_scalar(data, "imager_pixel_spacing"))
            spacing_units = (
                str(_scalar(data, "imager_pixel_spacing_units")).strip().lower()
                if "imager_pixel_spacing_units" in data.files
                else "mm"
            )
            if spacing_units in {
                "mm", "millimeter", "millimeters", "millimetre", "millimetres",
            }:
                detector_pixel_spacing_m = spacing_value * 1.0e-3
            elif spacing_units in {"m", "meter", "meters", "metre", "metres"}:
                detector_pixel_spacing_m = spacing_value
            else:
                raise ValueError(
                    f"Unsupported imager_pixel_spacing_units={spacing_units!r}."
                )
        else:
            detector_pixel_spacing_m = (
                float(fallback_detector_pixel_spacing_mm) * 1.0e-3
            )
        if detector_pixel_spacing_m <= 0.0:
            raise ValueError("Detector pixel spacing must be positive.")

        sample_name = (
            str(_scalar(data, "sample_name"))
            if "sample_name" in data.files
            else npz_path.stem
        )
        clinical_views = (
            np.asarray(data["clinical_views"]).astype(str)
            if "clinical_views" in data.files
            else np.asarray([f"view_{index}" for index in range(images.shape[0])])
        )
        if clinical_views.shape != (images.shape[0],):
            raise ValueError(
                "clinical_views must have one label per image, got "
                f"{clinical_views.shape}."
            )
        center_offset = (
            np.asarray(data["projection_center_offset"], dtype=np.float32).reshape(3)
            if "projection_center_offset" in data.files
            else None
        )
        if center_offset is not None and not np.isfinite(center_offset).all():
            raise ValueError("projection_center_offset contains NaN or infinity.")
        if "projection_representation" in data.files:
            projection_representation = str(
                _scalar(data, "projection_representation")
            ).strip().lower()
        else:
            projection_representation = (
                "binary_mask"
                if float(images.min()) >= 0.0 and float(images.max()) <= 1.0
                else "line_integral_mm"
            )
        if projection_representation not in {"binary_mask", "line_integral_mm"}:
            raise ValueError(
                "projection_representation must be 'binary_mask' or "
                f"'line_integral_mm', got {projection_representation!r}."
            )

    return Stage2ProjectionCase(
        path=npz_path,
        sample_name=sample_name,
        images=np.ascontiguousarray(images),
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        sid_m=sid_m,
        source_origin_distance_m=source_distance_m,
        detector_origin_distance_m=detector_distance_m,
        detector_pixel_spacing_m=detector_pixel_spacing_m,
        clinical_views=clinical_views,
        projection_center_offset_m=center_offset,
        projection_representation=projection_representation,
    )


def stage2_angles_to_camera_frames(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    sid_m: float,
    source_origin_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the Stage-2 ``get_local_params`` camera convention."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64).reshape(-1))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64).reshape(-1))
    if theta.shape != phi.shape:
        raise ValueError("theta_deg and phi_deg must have the same shape.")

    detector_origin_distance_m = float(sid_m) - float(source_origin_distance_m)
    if detector_origin_distance_m <= 0.0:
        raise ValueError("SID must exceed the source-to-isocentre distance.")

    coordinate_change = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    coordinate_change_inv = coordinate_change.T
    detector_centres = []
    source_positions = []
    detector_u_axes = []
    detector_v_axes = []

    for theta_i, phi_i in zip(theta, phi):
        rotation_theta_native = np.asarray(
            [
                [np.cos(theta_i), -np.sin(theta_i), 0.0],
                [np.sin(theta_i), np.cos(theta_i), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation_phi_native = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi_i), np.sin(phi_i)],
                [0.0, -np.sin(phi_i), np.cos(phi_i)],
            ]
        )
        rotation_theta = (
            coordinate_change @ rotation_theta_native @ coordinate_change_inv
        )
        rotation_phi = coordinate_change @ rotation_phi_native @ coordinate_change_inv
        rotation = rotation_theta @ rotation_phi

        detector_direction = rotation @ np.asarray([0.0, 0.0, 1.0])
        detector_centre = detector_direction * detector_origin_distance_m
        source_position = -detector_direction * float(source_origin_distance_m)
        detector_u = rotation @ np.asarray([0.0, 1.0, 0.0])
        detector_v = rotation @ np.asarray([-1.0, 0.0, 0.0])

        detector_centres.append(detector_centre)
        source_positions.append(source_position)
        detector_u_axes.append(detector_u / np.linalg.norm(detector_u))
        detector_v_axes.append(detector_v / np.linalg.norm(detector_v))

    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            source_positions,
            detector_centres,
            detector_u_axes,
            detector_v_axes,
        )
    )


def stage2_angles_to_sdfcar_angles(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Convert Stage-2 ``theta/phi`` to SDF-CAR's two-angle convention.

    SDF-CAR stores a pair ``[A, B]`` and internally constructs its ray as

    ``[cos(A) sin(B), cos(A) cos(B), -sin(A)]``.

    Stage-2's central ray is the usual spherical direction

    ``[sin(phi) cos(theta), sin(phi) sin(theta), cos(phi)]``.

    Equating those expressions gives ``A = phi - 90`` and
    ``B = 90 - theta`` (degrees).  The returned second angle is wrapped to
    ``[-180, 180)`` for stable, human-readable metadata.

    This conversion intentionally describes only the central ray.  Use
    :func:`stage2_angles_to_camera_frames` when constructing an ODL camera so
    the detector axes (and therefore image roll) remain exact.
    """
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if theta.shape != phi.shape:
        raise ValueError("theta_deg and phi_deg must have the same shape.")
    if not np.isfinite(theta).all() or not np.isfinite(phi).all():
        raise ValueError("theta_deg and phi_deg must be finite.")

    first = phi - 90.0
    second = np.mod(90.0 - theta + 180.0, 360.0) - 180.0
    return np.stack((first, second), axis=-1).astype(np.float32)


def sdfcar_angles_to_central_ray(angles_deg: np.ndarray) -> np.ndarray:
    """Return SDF-CAR's source-to-detector unit ray for ``[..., 2]`` pairs."""
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim < 1 or angles.shape[-1] != 2:
        raise ValueError(
            f"angles_deg must have shape [...,2], got {angles.shape}."
        )
    if not np.isfinite(angles).all():
        raise ValueError("angles_deg must be finite.")

    first = np.deg2rad(angles[..., 0])
    second = np.deg2rad(angles[..., 1])
    rays = np.stack(
        (
            np.cos(first) * np.sin(second),
            np.cos(first) * np.cos(second),
            -np.sin(first),
        ),
        axis=-1,
    )
    return rays.astype(np.float32)


def embed_roi_mask_in_reference_grid(
    roi_mask_xyz: np.ndarray,
    *,
    roi_spacing_mm: Sequence[float],
    roi_center_xyz_mm: Sequence[float],
    reference_shape_xyz: Sequence[int],
    reference_spacing_xyz_mm: Sequence[float],
) -> np.ndarray:
    """Nearest-neighbour map of a centred XYZ ROI into a full XYZ voxel grid.

    The reference convention matches the supplied ImageCAS NPZ: reference voxel
    index ``[0,0,0]`` is treated as physical XYZ ``[0,0,0]`` millimetres.
    """
    roi = np.asarray(roi_mask_xyz)
    if roi.ndim != 3:
        raise ValueError(f"roi_mask_xyz must be 3D, got {roi.shape}.")
    roi_spacing = np.asarray(roi_spacing_mm, dtype=np.float64).reshape(3)
    roi_center = np.asarray(roi_center_xyz_mm, dtype=np.float64).reshape(3)
    reference_shape = np.asarray(reference_shape_xyz, dtype=np.int64).reshape(3)
    reference_spacing = np.asarray(
        reference_spacing_xyz_mm, dtype=np.float64
    ).reshape(3)
    if np.any(roi_spacing <= 0.0) or np.any(reference_spacing <= 0.0):
        raise ValueError("Voxel spacing must be positive.")
    if np.any(reference_shape <= 0):
        raise ValueError("Reference shape must be positive.")

    roi_shape = np.asarray(roi.shape, dtype=np.int64)
    roi_first_center = roi_center - (roi_shape - 1) * roi_spacing / 2.0
    reference_axes = [
        np.arange(int(size), dtype=np.float64) * spacing
        for size, spacing in zip(reference_shape, reference_spacing)
    ]
    roi_indices = [
        np.rint((axis - first) / spacing).astype(np.int64)
        for axis, first, spacing in zip(reference_axes, roi_first_center, roi_spacing)
    ]
    valid = [
        np.logical_and(index >= 0, index < int(size))
        for index, size in zip(roi_indices, roi_shape)
    ]
    output = np.zeros(tuple(int(value) for value in reference_shape), dtype=roi.dtype)
    if not all(np.any(axis_valid) for axis_valid in valid):
        return output

    reference_selection = np.ix_(*[np.flatnonzero(axis_valid) for axis_valid in valid])
    roi_selection = np.ix_(*[
        index[axis_valid] for index, axis_valid in zip(roi_indices, valid)
    ])
    output[reference_selection] = roi[roi_selection]
    return output


def extract_reference_grid_roi(
    reference_volume_xyz: np.ndarray,
    *,
    reference_spacing_xyz_mm: Sequence[float],
    roi_shape_xyz: Sequence[int],
    roi_spacing_xyz_mm: Sequence[float],
    roi_center_xyz_mm: Sequence[float],
) -> np.ndarray:
    """Nearest-neighbour sample a centred XYZ ROI from a reference grid.

    This is the inverse coordinate convention of
    :func:`embed_roi_mask_in_reference_grid`: reference index ``[0,0,0]`` is
    physical XYZ ``[0,0,0]`` millimetres, while the returned ROI is centred on
    ``roi_center_xyz_mm`` and follows the voxel-centre coordinates used by the
    SDF-CAR reconstruction grid.
    """
    reference = np.asarray(reference_volume_xyz)
    if reference.ndim != 3:
        raise ValueError(
            f"reference_volume_xyz must be 3D, got {reference.shape}."
        )
    reference_spacing = np.asarray(
        reference_spacing_xyz_mm, dtype=np.float64
    ).reshape(3)
    roi_shape = np.asarray(roi_shape_xyz, dtype=np.int64).reshape(3)
    roi_spacing = np.asarray(roi_spacing_xyz_mm, dtype=np.float64).reshape(3)
    roi_center = np.asarray(roi_center_xyz_mm, dtype=np.float64).reshape(3)
    if np.any(reference_spacing <= 0.0) or np.any(roi_spacing <= 0.0):
        raise ValueError("Voxel spacing must be positive.")
    if np.any(roi_shape <= 0):
        raise ValueError("ROI shape must be positive.")
    if not np.isfinite(roi_center).all():
        raise ValueError("ROI centre contains NaN or infinity.")

    roi_first_center = roi_center - (roi_shape - 1) * roi_spacing / 2.0
    physical_axes = [
        first + np.arange(int(size), dtype=np.float64) * spacing
        for first, size, spacing in zip(roi_first_center, roi_shape, roi_spacing)
    ]
    reference_indices = [
        np.rint(axis / spacing).astype(np.int64)
        for axis, spacing in zip(physical_axes, reference_spacing)
    ]
    valid = [
        np.logical_and(index >= 0, index < int(size))
        for index, size in zip(reference_indices, reference.shape)
    ]
    output = np.zeros(tuple(int(value) for value in roi_shape), dtype=reference.dtype)
    if not all(np.any(axis_valid) for axis_valid in valid):
        return output

    roi_selection = np.ix_(*[np.flatnonzero(axis_valid) for axis_valid in valid])
    reference_selection = np.ix_(*[
        index[axis_valid] for index, axis_valid in zip(reference_indices, valid)
    ])
    output[roi_selection] = reference[reference_selection]
    return output


def binary_mask_dice(
    prediction_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[float, int, int, int]:
    """Return Dice and auditable foreground/intersection voxel counts.

    Any nonzero value is treated as foreground. If both masks are empty, their
    Dice score is defined as 1.0.
    """
    prediction = np.asarray(prediction_mask) != 0
    reference = np.asarray(reference_mask) != 0
    if prediction.shape != reference.shape:
        raise ValueError(
            "Prediction and reference masks must have the same shape, got "
            f"{prediction.shape} and {reference.shape}."
        )

    prediction_voxels = int(np.count_nonzero(prediction))
    reference_voxels = int(np.count_nonzero(reference))
    intersection_voxels = int(np.count_nonzero(prediction & reference))
    denominator = prediction_voxels + reference_voxels
    dice = (
        1.0
        if denominator == 0
        else (2.0 * intersection_voxels) / denominator
    )
    return float(dice), prediction_voxels, reference_voxels, intersection_voxels


__all__ = [
    "DEFAULT_SOURCE_ORIGIN_DISTANCE_M",
    "Stage2ProjectionCase",
    "binary_mask_dice",
    "embed_roi_mask_in_reference_grid",
    "extract_reference_grid_roi",
    "load_stage2_projection_case",
    "sdfcar_angles_to_central_ray",
    "stage2_angles_to_camera_frames",
    "stage2_angles_to_sdfcar_angles",
    "validate_view_indices",
]
