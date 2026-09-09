import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.stage2_npz import (
    binary_mask_dice,
    embed_roi_mask_in_reference_grid,
    extract_reference_grid_roi,
    load_stage2_projection_case,
    stage2_angles_to_camera_frames,
)


class Stage2NpzTest(unittest.TestCase):
    def test_loads_projection_case_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            np.savez_compressed(
                path,
                sample_name=np.asarray("case_1"),
                images=np.zeros((2, 8, 8), dtype=np.float32),
                theta_deg=np.asarray([0.0, 90.0], dtype=np.float32),
                phi_deg=np.asarray([0.0, 0.0], dtype=np.float32),
                sid=np.asarray(0.9, dtype=np.float32),
                imager_pixel_spacing=np.asarray(0.55, dtype=np.float32),
                imager_pixel_spacing_units=np.asarray("mm"),
                clinical_views=np.asarray(["a", "b"]),
                projection_center_offset=np.asarray([0.1, 0.2, 0.3]),
            )
            case = load_stage2_projection_case(path)

        self.assertEqual(case.num_views, 2)
        self.assertEqual(case.detector_shape, (8, 8))
        self.assertAlmostEqual(case.detector_origin_distance_m, 0.15, places=6)
        self.assertAlmostEqual(case.detector_pixel_spacing_m, 0.00055, places=8)
        self.assertEqual(case.projection_representation, "binary_mask")
        self.assertEqual(case.cone_vectors().shape, (2, 12))

    def test_loads_explicit_line_integral_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line_integrals.npz"
            np.savez_compressed(
                path,
                images=np.full((2, 8, 8), 2.5, dtype=np.float32),
                theta_deg=np.asarray([0.0, 90.0], dtype=np.float32),
                phi_deg=np.asarray([0.0, 0.0], dtype=np.float32),
                sid=np.asarray(0.9, dtype=np.float32),
                imager_pixel_spacing=np.asarray(0.55, dtype=np.float32),
                projection_representation=np.asarray("line_integral_mm"),
            )
            case = load_stage2_projection_case(path)

        self.assertFalse(case.is_binary_mask)
        self.assertEqual(case.projection_representation, "line_integral_mm")

    def test_zero_angles_match_stage2_base_camera(self):
        source, detector, u_axis, v_axis = stage2_angles_to_camera_frames(
            np.asarray([0.0]), np.asarray([0.0]), 0.9, 0.75
        )
        np.testing.assert_allclose(source[0], [0.0, 0.0, -0.75], atol=1e-7)
        np.testing.assert_allclose(detector[0], [0.0, 0.0, 0.15], atol=1e-7)
        np.testing.assert_allclose(u_axis[0], [0.0, 1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(v_axis[0], [-1.0, 0.0, 0.0], atol=1e-7)

    def test_embeds_centered_roi_in_reference_grid(self):
        roi = np.ones((3, 3, 3), dtype=np.uint8)
        output = embed_roi_mask_in_reference_grid(
            roi,
            roi_spacing_mm=(1.0, 1.0, 1.0),
            roi_center_xyz_mm=(2.0, 2.0, 2.0),
            reference_shape_xyz=(5, 5, 5),
            reference_spacing_xyz_mm=(1.0, 1.0, 1.0),
        )
        self.assertEqual(int(output.sum()), 27)
        self.assertTrue(output[1:4, 1:4, 1:4].all())

    def test_binary_mask_dice_returns_score_and_counts(self):
        prediction = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        reference = np.asarray([0, 1, 1, 0], dtype=np.uint8)
        dice, prediction_count, reference_count, intersection_count = (
            binary_mask_dice(prediction, reference)
        )
        self.assertEqual(dice, 0.5)
        self.assertEqual(prediction_count, 2)
        self.assertEqual(reference_count, 2)
        self.assertEqual(intersection_count, 1)

        empty_dice, _, _, _ = binary_mask_dice(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
        )
        self.assertEqual(empty_dice, 1.0)

    def test_extracts_reference_roi_at_physical_center(self):
        reference = np.arange(5 ** 3, dtype=np.int32).reshape(5, 5, 5)
        roi = extract_reference_grid_roi(
            reference,
            reference_spacing_xyz_mm=(1.0, 1.0, 1.0),
            roi_shape_xyz=(3, 3, 3),
            roi_spacing_xyz_mm=(1.0, 1.0, 1.0),
            roi_center_xyz_mm=(2.0, 2.0, 2.0),
        )
        np.testing.assert_array_equal(roi, reference[1:4, 1:4, 1:4])


if __name__ == "__main__":
    unittest.main()
