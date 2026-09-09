import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from generate_2d_projections import generate_projection_npz


class Generate2DProjectionsTest(unittest.TestCase):
    def test_writes_line_integral_training_npz_and_binary_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_path = root / "camera.npz"
            reference_path = root / "reference.npz"
            output_path = root / "generated.npz"
            np.savez_compressed(
                camera_path,
                sample_name=np.asarray("case_1"),
                images=np.zeros((2, 8, 8), dtype=np.float32),
                theta_deg=np.asarray([-40.0, 75.0], dtype=np.float32),
                phi_deg=np.asarray([80.0, 80.0], dtype=np.float32),
                sid=np.asarray(0.9, dtype=np.float32),
                imager_pixel_spacing=np.asarray(0.55, dtype=np.float32),
                imager_pixel_spacing_units=np.asarray("mm"),
                clinical_views=np.asarray(["view 0", "view 1"]),
                projection_center_offset=np.asarray(
                    [0.002, 0.002, 0.002], dtype=np.float32
                ),
            )
            reference = np.zeros((5, 5, 5), dtype=np.uint8)
            reference[1:4, 1:4, 1:4] = 1
            np.savez_compressed(
                reference_path,
                vol=reference,
                spacing=np.ones(3, dtype=np.float32),
            )
            config = {
                "exp": {
                    "dataconfig": "config/config.yml",
                    "projection_npz": str(output_path),
                    "reference_volume_npz": str(reference_path),
                    "source_origin_distance_m": 0.75,
                    "view_indices": [0, 1],
                    "reconstruction_nVoxel": [3, 3, 3],
                    "reconstruction_extent_m": [0.003, 0.003, 0.003],
                },
                "projection_generation": {
                    "camera_metadata_npz": str(camera_path),
                },
            }
            line_integrals = np.zeros((2, 8, 8), dtype=np.float32)
            line_integrals[:, 2:6, 2:6] = 2.5
            with (
                patch(
                    "generate_2d_projections._forward_project_reference_roi",
                    return_value=line_integrals,
                ),
                patch(
                    "generate_2d_projections._save_projection_pngs",
                    return_value=[],
                ),
            ):
                generated_path = generate_projection_npz(config)

            self.assertEqual(generated_path, output_path.resolve())
            with np.load(output_path, allow_pickle=False) as generated:
                np.testing.assert_array_equal(generated["images"], line_integrals)
                np.testing.assert_array_equal(
                    generated["binary_images"], line_integrals > 0.0
                )
                self.assertEqual(
                    generated["projection_representation"].item(),
                    "line_integral_mm",
                )
                self.assertEqual(
                    int(np.count_nonzero(generated["reference_roi_mask_xyz"])),
                    27,
                )
                np.testing.assert_allclose(
                    generated["sdfcar_projection_angles_deg"],
                    [[-10.0, 130.0], [-10.0, 15.0]],
                    atol=1e-6,
                )
                self.assertEqual(
                    generated["odl_source_to_detector_unit_xyz"].shape,
                    (2, 3),
                )
                self.assertEqual(
                    generated["odl_detector_row_axis_xyz"].shape,
                    (2, 3),
                )
                self.assertEqual(
                    generated["odl_detector_column_axis_xyz"].shape,
                    (2, 3),
                )


if __name__ == "__main__":
    unittest.main()
