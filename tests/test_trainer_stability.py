import unittest


try:
    import torch

    from src.network.network import DensityNetwork
    from src.render.sdf_utils import (
        occupancy_to_sdf_2d,
        ray_integral_to_binary_probability,
        scheduled_loss_weight,
    )
except ModuleNotFoundError:
    torch = None
    DensityNetwork = None
    occupancy_to_sdf_2d = None
    ray_integral_to_binary_probability = None
    scheduled_loss_weight = None


@unittest.skipIf(torch is None, "PyTorch/SciPy training dependencies unavailable")
class TrainerStabilityTest(unittest.TestCase):
    class IdentityEncoder(torch.nn.Module if torch is not None else object):
        output_dim = 3

        def forward(self, inputs, bound):
            del bound
            return inputs

    def test_sdf_network_starts_with_sparse_occupancy_bias(self):
        network = DensityNetwork(
            self.IdentityEncoder(),
            num_layers=2,
            hidden_dim=4,
            out_dim=1,
            use_sdf=True,
            use_gradient_checkpointing=False,
            sdf_initial_bias=0.1,
        )
        self.assertAlmostEqual(
            float(network.layers[-1].bias.detach().item()), 0.1, places=6
        )
        initial_occupancy = torch.sigmoid(
            -50.0 * network.layers[-1].bias.detach()
        )
        self.assertLess(float(initial_occupancy.item()), 0.01)
        self.assertGreater(float(initial_occupancy.item()), 0.0)

    def test_binary_projection_map_retains_finite_nonzero_gradient(self):
        ray_lengths = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        probabilities = ray_integral_to_binary_probability(
            ray_lengths, attenuation=1.0
        )
        probabilities.sum().backward()

        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue(torch.isfinite(ray_lengths.grad).all())
        self.assertGreater(float(ray_lengths.grad[1].item()), 0.0)
        self.assertGreater(float(ray_lengths.grad[2].item()), 0.0)

    def test_sdf_loss_weight_has_projection_only_warmup_and_ramp(self):
        self.assertEqual(scheduled_loss_weight(1.0, 1, 100, 400), 0.0)
        self.assertEqual(scheduled_loss_weight(1.0, 100, 100, 400), 0.0)
        self.assertAlmostEqual(
            scheduled_loss_weight(1.0, 300, 100, 400), 0.5
        )
        self.assertEqual(scheduled_loss_weight(1.0, 500, 100, 400), 1.0)
        self.assertEqual(scheduled_loss_weight(1.0, 900, 100, 400), 1.0)

    def test_kornia_distance_transform_has_finite_gradient_at_zero_margins(self):
        try:
            import kornia  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Kornia unavailable")

        occupancy = torch.zeros((16, 16), requires_grad=True)
        with torch.no_grad():
            occupancy[6:10, 6:10] = 0.75
        sdf = occupancy_to_sdf_2d(occupancy, distance_epsilon=1.0e-6)
        sdf.square().mean().backward()

        self.assertTrue(torch.isfinite(sdf).all())
        self.assertTrue(torch.isfinite(occupancy.grad).all())


if __name__ == "__main__":
    unittest.main()
