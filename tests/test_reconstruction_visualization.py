import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from src.reconstruction_visualization import (
    save_mask_surface_gif,
    save_predicted_projection_pngs,
)


class ReconstructionVisualizationTest(unittest.TestCase):
    def test_saves_one_projection_png_per_source_view(self):
        projections = np.stack(
            [np.eye(8, dtype=np.float32), np.fliplr(np.eye(8, dtype=np.float32))]
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_predicted_projection_pngs(
                projections, [0, 3], directory
            )
            self.assertEqual(
                [path.name for path in paths],
                ["predicted_projection_view_0.png", "predicted_projection_view_3.png"],
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertTrue(all(imageio.imread(path).shape[:2] == (8, 8) for path in paths))

    def test_saves_rotating_binary_mask_surface_gif(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = save_mask_surface_gif(
                mask,
                (0.5, 0.5, 1.0),
                Path(directory) / "surface.gif",
                title="Test surface",
                num_frames=2,
                fps=1,
            )
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(len(imageio.mimread(path)), 2)
            self.assertEqual(imageio.get_reader(path).get_meta_data()["duration"], 1000)


if __name__ == "__main__":
    unittest.main()
