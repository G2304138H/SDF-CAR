import math

import numpy as np
import torch
from scipy import ndimage
try:
    from kornia.contrib import distance_transform
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print(
        "Warning: kornia not available; SciPy distance transforms are "
        "non-differentiable and cannot be used as a training loss."
    )


def require_differentiable_distance_transform():
    """Fail clearly when the configured geometric training loss cannot backprop."""
    if not KORNIA_AVAILABLE:
        raise RuntimeError(
            "The 2D SDF training loss requires Kornia's differentiable distance "
            "transform. Install it with `python -m pip install kornia`, or set "
            "train.current_loss_weights[1] to 0 to disable the SDF loss. The "
            "SciPy fallback is valid only for targets/evaluation because it "
            "detaches tensors from autograd."
        )


def scheduled_loss_weight(base_weight, epoch, warmup_epochs=0, ramp_epochs=0):
    """Return a loss weight after an optional zero-weight warm-up and ramp."""
    base_weight = float(base_weight)
    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    if not math.isfinite(base_weight) or base_weight < 0.0:
        raise ValueError("base_weight must be finite and non-negative.")
    if epoch < 0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("epoch, warmup_epochs, and ramp_epochs must be non-negative.")
    if base_weight == 0.0 or epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return base_weight
    ramp_fraction = min((epoch - warmup_epochs) / ramp_epochs, 1.0)
    return base_weight * ramp_fraction


def ray_integral_to_binary_probability(ray_integrals, attenuation=1.0):
    """Map non-negative ray lengths to silhouette probabilities in FP32.

    ``-expm1(-x)`` is equivalent to ``1-exp(-x)`` but is more accurate near
    zero.  Disabling autocast here prevents the exponential derivative from
    underflowing prematurely in FP16.
    """
    attenuation = float(attenuation)
    if not math.isfinite(attenuation) or attenuation <= 0.0:
        raise ValueError("projection attenuation must be finite and positive.")
    with torch.amp.autocast(device_type=ray_integrals.device.type, enabled=False):
        lengths = torch.clamp_min(ray_integrals.float(), 0.0)
        return -torch.expm1(-attenuation * lengths)


def sdf_to_occupancy(sdf, alpha=50.0):
    """
    Convert SDF to occupancy/density for volume rendering.
    
    Args:
        sdf: SDF values (negative inside vessel, positive outside)
        alpha: Sharpness parameter for sigmoid conversion
        
    Returns:
        occupancy: Values in [0,1] representing vessel occupancy
    """
    # Inside vessel (negative SDF) -> high occupancy (near 1)
    # Outside vessel (positive SDF) -> low occupancy (near 0)
    occupancy = torch.sigmoid(-alpha * sdf)
    return occupancy


def occupancy_to_sdf_2d(
    occupancy_2d,
    voxel_size=1.0,
    use_kornia=True,
    distance_epsilon=1.0e-6,
):
    """
    Convert 2D occupancy/projection to 2D SDF using distance transform.
    
    Args:
        occupancy_2d: 2D projection tensor [batch, height, width] or [height, width]
        voxel_size: Physical size of pixels for distance calculation
        use_kornia: Use differentiable kornia distance transform if available (default: True)
        distance_epsilon: Soft-mask bound used before Kornia's logarithmic
            transform. This prevents finite forward values with NaN gradients
            at exact-zero or exact-one projection pixels.
        
    Returns:
        sdf_2d: 2D SDF tensor (negative inside, positive outside)
    """
    original_shape = occupancy_2d.shape
    if occupancy_2d.dim() == 2:
        occupancy_2d = occupancy_2d.unsqueeze(0)  # Add batch dimension
    
    batch_size = occupancy_2d.shape[0]
    device = occupancy_2d.device
    
    # Use differentiable kornia implementation if available and requested
    if KORNIA_AVAILABLE and use_kornia:
        distance_epsilon = float(distance_epsilon)
        if (
            not math.isfinite(distance_epsilon)
            or distance_epsilon <= 0.0
            or distance_epsilon >= 0.5
        ):
            raise ValueError(
                "distance_epsilon must be finite and strictly between 0 and 0.5."
            )

        # Kornia defines DT(image) as distance to the nearest non-zero pixel.
        # Keeping this mask soft preserves the gradient from the geometric loss
        # back to the projected volume.  A hard ``> 0.5`` conversion here would
        # sever that gradient entirely. Keep the transform in FP32 since its
        # iterative exponentials are numerically fragile in mixed precision.
        # Kornia internally evaluates ``-h * log(filtered_mask)``. Exact-zero
        # detector pixels therefore create an infinite local derivative even
        # though Kornia sanitizes its forward output with ``nan_to_num``. Bound
        # both sides of the soft mask so its backward pass stays finite.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            numerical_epsilon = max(
                distance_epsilon,
                float(torch.finfo(torch.float32).eps),
            )
            soft_mask = occupancy_2d.float().clamp(
                numerical_epsilon, 1.0 - numerical_epsilon
            ).unsqueeze(1)

            # Positive outside and negative inside.  For a binary input,
            # DT(mask) is zero inside and positive outside; DT(1-mask) is the
            # converse.
            distance_to_foreground = distance_transform(soft_mask) * voxel_size
            distance_to_background = distance_transform(1.0 - soft_mask) * voxel_size
            sdf_2d = (distance_to_foreground - distance_to_background).squeeze(1)
        
        # Remove batch dimension if input was 2D
        if len(original_shape) == 2:
            sdf_2d = sdf_2d.squeeze(0)
        
        return sdf_2d
    else:
        # Fallback to scipy (non-differentiable) implementation
        sdf_2d_list = []
        
        for b in range(batch_size):
            # Convert to numpy for scipy operations
            occ_np = occupancy_2d[b].detach().cpu().numpy()
            
            # Threshold to create binary mask
            binary_mask = occ_np > 0.5
            
            # Distance transform for inside (negative distances)
            inside_dist = ndimage.distance_transform_edt(binary_mask) * voxel_size
            
            # Distance transform for outside (positive distances)  
            outside_dist = ndimage.distance_transform_edt(~binary_mask) * voxel_size
            
            # Combine: negative inside, positive outside
            sdf_2d = np.where(binary_mask, -inside_dist, outside_dist)
            
            # Convert back to tensor
            sdf_2d_tensor = torch.tensor(sdf_2d, dtype=torch.float32, device=device)
            sdf_2d_list.append(sdf_2d_tensor)
        
        result = torch.stack(sdf_2d_list, dim=0)
        
        # Remove batch dimension if input was 2D
        if len(original_shape) == 2:
            result = result.squeeze(0)
        
        return result


def sdf_3d_to_occupancy_to_sdf_2d(sdf_3d, projector_first, projector_second, 
                                  alpha=50.0, voxel_size_2d=0.278):
    """
    Complete pipeline: 3D SDF -> occupancy -> 2D projections -> 2D SDF
    
    Args:
        sdf_3d: 3D SDF prediction
        projector_first: First view projector
        projector_second: Second view projector  
        alpha: SDF to occupancy conversion sharpness
        voxel_size_2d: Pixel size for 2D SDF computation (mm)
        
    Returns:
        sdf_2d_combined: Combined 2D SDF from both views [batch, 2, height, width]
        occupancy_projections: Intermediate occupancy projections for debugging
    """
    # Convert 3D SDF to occupancy
    occupancy_3d = sdf_to_occupancy(sdf_3d, alpha=alpha)
    
    # Project to 2D occupancy
    proj_occupancy_first = projector_first.forward_project(occupancy_3d)
    proj_occupancy_second = projector_second.forward_project(occupancy_3d)
    
    # Combine projections
    occupancy_projections = torch.cat((proj_occupancy_first, proj_occupancy_second), dim=1)
    
    # Convert each projection to 2D SDF
    batch_size = occupancy_projections.shape[0]
    num_views = occupancy_projections.shape[1]
    
    sdf_2d_list = []
    for b in range(batch_size):
        view_sdf_list = []
        for v in range(num_views):
            proj = occupancy_projections[b, v]  # [height, width]
            sdf_2d = occupancy_to_sdf_2d(proj, voxel_size=voxel_size_2d)
            view_sdf_list.append(sdf_2d)
        sdf_2d_batch = torch.stack(view_sdf_list, dim=0)  # [num_views, height, width]
        sdf_2d_list.append(sdf_2d_batch)
    
    sdf_2d_combined = torch.stack(sdf_2d_list, dim=0)  # [batch, num_views, height, width]
    
    return sdf_2d_combined, occupancy_projections
