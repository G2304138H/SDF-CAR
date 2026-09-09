import unittest


try:
    import torch

    from src.network.network import DensityNetwork
    from src.render.sdf_utils import ray_integral_to_binary_probability
except ModuleNotFoundError:
    torch = None
    DensityNetwork = None
    ray_integral_to_binary_probability = None


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


if __name__ == "__main__":
    unittest.main()
