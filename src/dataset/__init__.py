"""Dataset exports.

Keep the pure-NumPy Stage-2 reader usable for data preparation on machines
without a PyTorch/CUDA environment.  The training dataset is imported lazily.
"""

from .stage2_npz import (
    Stage2ProjectionCase,
    binary_mask_dice,
    embed_roi_mask_in_reference_grid,
    extract_reference_grid_roi,
    load_stage2_projection_case,
    validate_view_indices,
)


def __getattr__(name):
    if name == "TIGREDataset":
        from .tigre import TIGREDataset

        return TIGREDataset
    raise AttributeError(name)


__all__ = [
    "Stage2ProjectionCase",
    "TIGREDataset",
    "binary_mask_dice",
    "embed_roi_mask_in_reference_grid",
    "extract_reference_grid_roi",
    "load_stage2_projection_case",
    "validate_view_indices",
]
