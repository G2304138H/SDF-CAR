#!/usr/bin/env python3
"""Run sequential per-case reconstruction from a selected directory of NPZs."""

from __future__ import annotations

import argparse
import gc
import os.path as osp
import sys
import traceback

import numpy as np
import torch

from src.batch_cases import make_case_config, resolve_projection_cases
from src.config.configloading import load_config
from train import BasicTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct selected external projection NPZ cases."
    )
    parser.add_argument("--config", required=True, help="Batch YAML configuration.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device (default: cuda; respects CUDA_VISIBLE_DEVICES).",
    )
    return parser


def _reset_case_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _validate_prediction_npz(path: str) -> None:
    if not osp.isfile(path):
        raise RuntimeError(f"Trainer completed without creating output: {path}")
    with np.load(path, allow_pickle=False) as prediction:
        required = {"vol", "spacing", "vol_axis_order"}
        missing = required.difference(prediction.files)
        if missing:
            raise RuntimeError(
                f"Prediction NPZ is missing keys {sorted(missing)}: {path}"
            )
        volume = prediction["vol"]
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise RuntimeError(
                "Prediction 'vol' must be a 3D uint8 array, got "
                f"shape={volume.shape}, dtype={volume.dtype}: {path}"
            )


def main() -> int:
    args = _parser().parse_args()
    base_config = load_config(args.config)
    cases = resolve_projection_cases(base_config.get("exp", {}))
    device = torch.device(args.device)
    seed = int(base_config.get("train", {}).get("seed", 42))

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU.")

    print(f"Selected {len(cases)} case(s): {[case.case_name for case in cases]}")
    successful: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for index, case in enumerate(cases, start=1):
        print("\n" + "=" * 72)
        print(f"Case {index}/{len(cases)}: {case.case_name}")
        print(f"Input: {case.projection_npz}")
        print("=" * 72)
        trainer = None
        try:
            case_config = make_case_config(base_config, case)
            _reset_case_seed(seed)
            trainer = BasicTrainer(case_config, device)
            trainer.start()
            output_path = osp.join(
                trainer.output_recon_dir,
                f"reconstruction_{case.case_name}.npz",
            )
            _validate_prediction_npz(output_path)
            successful.append((case.case_name, output_path))
            print(f"Completed {case.case_name}: {output_path}")
        except Exception as exc:  # Continue so one failed case does not lose the batch.
            failed.append((case.case_name, str(exc)))
            print(f"FAILED {case.case_name}: {exc}", file=sys.stderr)
            traceback.print_exc()
        finally:
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nBatch reconstruction summary")
    print(f"Successful: {len(successful)}/{len(cases)}")
    for case_name, output_path in successful:
        print(f"  {case_name}: {output_path}")
    if failed:
        print(f"Failed: {len(failed)}", file=sys.stderr)
        for case_name, error in failed:
            print(f"  {case_name}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
