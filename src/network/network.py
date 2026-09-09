import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256,
                 skips=[4], out_dim=1, use_sdf=False,
                 use_gradient_checkpointing=True, sdf_initial_bias=0.1):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.sdf_initial_bias = float(sdf_initial_bias)
        if not math.isfinite(self.sdf_initial_bias):
            raise ValueError("sdf_initial_bias must be finite.")
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU(inplace=True) for i in range(0, num_layers-1, 1)])  # Use inplace for memory
        self.activations.append(nn.Identity() if use_sdf else nn.Sigmoid())

        # Apply SDF-tuned Xavier initialization if we are on SDF mode
        if use_sdf:
            self._init_weights_for_sdf()

    def _forward_layers(self, x, input_pts, start_layer=0, end_layer=None):
        """
        Forward pass through a subset of layers for gradient checkpointing.
        """
        if end_layer is None:
            end_layer = len(self.layers)
            
        for i in range(start_layer, end_layer):
            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x

    def forward(self, x):
        # Encode input
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            # Split the network into checkpointed segments
            mid_layer = len(self.layers) // 2
            
            # First half of layers
            x = checkpoint(self._forward_layers, x, input_pts, 0, mid_layer, use_reentrant=False)
            
            # Second half of layers  
            x = checkpoint(self._forward_layers, x, input_pts, mid_layer, len(self.layers), use_reentrant=False)
            
        else:
            # Standard forward pass for inference
            x = self._forward_layers(x, input_pts, 0, len(self.layers))
        
        return x

    def _init_weights_for_sdf(self):
        """
        SDF-tuned Xavier initialization for better convergence.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # Xavier uniform initialization with SDF-specific scaling
                if i == len(self.layers) - 1:  # Final layer
                    # A zero SDF produces occupancy=0.5 everywhere.  Projecting
                    # that dense cube saturates binary silhouette rendering and
                    # kills its gradient.  Start outside the surface instead so
                    # the initial occupancy is sparse but still differentiable.
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.constant_(layer.bias, self.sdf_initial_bias)
                else:
                    # Hidden layers with standard Xavier but slightly smaller gain
                    # This helps with gradient flow in deep SDF networks
                    nn.init.xavier_uniform_(layer.weight, gain=0.8)
                    nn.init.constant_(layer.bias, 0.0)


def get_network(name):
    if name == "mlp":
        return DensityNetwork
    else:
        raise NotImplementedError(f"Network {name} not implemented")
