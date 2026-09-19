import tempfile
import unittest
from pathlib import Path

from src.batch_cases import make_case_config, resolve_projection_cases


class BatchProjectionCasesTest(unittest.TestCase):
    def test_resolves_selected_padded_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lca_0001.npz", "lca_0002.npz", "lca_0003.npz"):
                (root / name).touch()

            cases = resolve_projection_cases({
                "projection_npz_dir": str(root),
                "case_prefix": "lca",
                "case_ids": ["1", "2"],
                "case_id_width": 4,
            })

        self.assertEqual([case.case_name for case in cases], ["lca_0001", "lca_0002"])
        self.assertEqual(
            [case.projection_npz.name for case in cases],
            ["lca_0001.npz", "lca_0002.npz"],
        )

    def test_accepts_complete_case_name_without_adding_prefix_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lca_0007.npz").touch()
            cases = resolve_projection_cases({
                "projection_npz_dir": str(root),
                "case_prefix": "lca",
                "case_ids": ["lca_0007"],
            })

        self.assertEqual(cases[0].case_name, "lca_0007")

    def test_supports_nested_projection_filename_pattern(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_directory = root / "lca_0001"
            case_directory.mkdir()
            (case_directory / "2d.npz").touch()
            cases = resolve_projection_cases({
                "projection_npz_dir": str(root),
                "case_prefix": "lca",
                "case_ids": [1],
                "projection_filename_pattern": "{case_name}/2d.npz",
            })

        self.assertEqual(cases[0].projection_npz.name, "2d.npz")

    def test_prediction_only_case_config_removes_reference_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection = root / "lca_0001.npz"
            projection.touch()
            base = {
                "exp": {
                    "projection_npz_dir": str(root),
                    "case_prefix": "lca",
                    "case_ids": ["1"],
                    "prediction_only": True,
                    "reference_volume_npz": "/unused/voxel_1.npz",
                },
                "train": {"epoch": 10},
            }
            case = resolve_projection_cases(base["exp"])[0]
            config = make_case_config(base, case)

        self.assertEqual(config["exp"]["current_model_id"], "lca_0001")
        self.assertEqual(config["exp"]["projection_npz"], str(projection))
        self.assertNotIn("reference_volume_npz", config["exp"])
        self.assertIn("reference_volume_npz", base["exp"])

    def test_missing_selected_case_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "lca_0002"):
                resolve_projection_cases({
                    "projection_npz_dir": directory,
                    "case_prefix": "lca",
                    "case_ids": ["2"],
                })


if __name__ == "__main__":
    unittest.main()
