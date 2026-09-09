"""Post-optimization visual checks for external-view reconstructions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_predicted_projection_pngs(
    projections: np.ndarray,
    view_indices: Sequence[int],
    output_dir: str | Path,
) -> list[Path]:
    """Save one display-ready grayscale PNG for each predicted detector view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(projections, dtype=np.float32)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3:
        raise ValueError(
            f"projections must have shape [V,H,W] or [1,V,H,W], got {values.shape}."
        )
    indices = np.asarray(view_indices, dtype=np.int64).reshape(-1)
    if values.shape[0] != indices.size:
        raise ValueError(
            "Projection count and view-index count differ: "
            f"{values.shape[0]} and {indices.size}."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for position, source_view_index in enumerate(indices):
        image = np.nan_to_num(
            values[position], nan=0.0, posinf=1.0, neginf=0.0
        )
        image = np.clip(image, 0.0, 1.0)
        path = destination / f"predicted_projection_view_{int(source_view_index)}.png"
        plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
        paths.append(path)
    return paths


def _binary_mask_mesh(
    mask_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an XYZ surface mesh, cropping first to reduce memory usage."""
    from skimage.measure import marching_cubes

    mask = np.asarray(mask_xyz) != 0
    if mask.ndim != 3:
        raise ValueError(f"mask_xyz must be 3D, got {mask.shape}.")
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64).reshape(3)
    if np.any(spacing <= 0.0):
        raise ValueError("Surface voxel spacing must be positive.")

    occupied_x = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if occupied_x.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
        )

    occupied_y = np.flatnonzero(np.any(mask, axis=(0, 2)))
    occupied_z = np.flatnonzero(np.any(mask, axis=(0, 1)))
    lower = np.maximum(
        np.asarray([occupied_x[0], occupied_y[0], occupied_z[0]]) - 1,
        0,
    )
    upper = np.minimum(
        np.asarray([occupied_x[-1], occupied_y[-1], occupied_z[-1]]) + 2,
        np.asarray(mask.shape),
    )
    selection = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    cropped = np.pad(mask[selection].astype(np.float32), 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(
        cropped,
        level=0.5,
        spacing=tuple(float(value) for value in spacing),
    )
    # Account for the cropped origin and the one-voxel zero padding.
    vertices += (lower - 1) * spacing
    return vertices.astype(np.float32), faces.astype(np.int32)


def save_mask_surface_gif(
    mask_xyz: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    output_path: str | Path,
    *,
    title: str,
    num_frames: int = 24,
    fps: int = 5,
    max_faces: int = 200_000,
) -> Path:
    """Save a rotating mask surface using the parametric evaluator trajectory."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frame_count = max(1, int(num_frames))
    frames_per_second = max(1, int(fps))
    vertices, faces = _binary_mask_mesh(mask_xyz, spacing_xyz_mm)
    if vertices.size:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)
        if max_faces > 0 and faces.shape[0] > int(max_faces):
            selected = np.linspace(
                0, faces.shape[0], num=int(max_faces), endpoint=False, dtype=np.int64
            )
            faces = faces[selected]
        triangles = vertices[faces]
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        half_extent = float(max(0.5 * np.max(maxs - mins), 1.0e-3))
    else:
        triangles = np.empty((0, 3, 3), dtype=np.float32)
        center = np.zeros(3, dtype=np.float32)
        half_extent = 1.0

    frames = []
    for frame_index in range(frame_count):
        figure = plt.figure(figsize=(5.2, 5.2), dpi=120)
        axis = figure.add_subplot(111, projection="3d")
        if triangles.size:
            surface = Poly3DCollection(
                triangles,
                facecolor="#d62728",
                edgecolor="none",
                alpha=0.62,
            )
            axis.add_collection3d(surface)
        axis.set_xlim(center[0] - half_extent, center[0] + half_extent)
        axis.set_ylim(center[1] - half_extent, center[1] + half_extent)
        axis.set_zlim(center[2] - half_extent, center[2] + half_extent)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.set_title(title)

        # Match methods/src/visualization.py::_save_3d_single_surface_gif.
        angle = 2.0 * np.pi * float(frame_index) / float(frame_count)
        axis.view_init(
            elev=22.0 + 6.0 * np.sin(angle),
            azim=360.0 * float(frame_index) / float(frame_count),
        )
        axis.set_axis_off()
        figure.tight_layout(pad=0.0)
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        frame = np.frombuffer(
            figure.canvas.buffer_rgba(), dtype=np.uint8
        ).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())
        plt.close(figure)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        path,
        frames,
        duration=1000.0 / float(frames_per_second),
        loop=0,
    )
    return path


__all__ = ["save_mask_surface_gif", "save_predicted_projection_pngs"]
