"""Resolve YAML-selected external projection cases for batch reconstruction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectionCaseSpec:
    """One resolved external projection case."""

    requested_id: str
    case_name: str
    projection_npz: Path


def _case_name(case_id: Any, prefix: str, width: int) -> tuple[str, str]:
    requested_id = str(case_id).strip()
    if not requested_id:
        raise ValueError("exp.case_ids cannot contain an empty case ID.")
    if "/" in requested_id or "\\" in requested_id:
        raise ValueError(
            f"Case ID {requested_id!r} cannot contain a path separator."
        )

    normalized_prefix = str(prefix).strip().strip("_")
    if normalized_prefix and requested_id.startswith(f"{normalized_prefix}_"):
        return requested_id, requested_id

    padded_id = requested_id.zfill(width) if requested_id.isdigit() else requested_id
    case_name = (
        f"{normalized_prefix}_{padded_id}" if normalized_prefix else padded_id
    )
    return requested_id, case_name


def resolve_projection_cases(exp: Mapping[str, Any]) -> list[ProjectionCaseSpec]:
    """Resolve ``projection_npz_dir`` and ``case_ids`` into existing NPZ files.

    The default filename pattern is ``{case_name}.npz``. For example, prefix
    ``lca``, width ``4`` and case ID ``"1"`` resolve to ``lca_0001.npz``.
    ``projection_filename_pattern`` may also contain ``{case_id}``,
    ``{case_id_padded}``, or ``{case_prefix}``, allowing layouts such as
    ``{case_name}/2d.npz``.
    """

    directory_value = exp.get("projection_npz_dir")
    if not directory_value:
        raise ValueError(
            "Batch reconstruction requires exp.projection_npz_dir."
        )
    directory = Path(str(directory_value)).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Projection NPZ directory does not exist: {directory}"
        )

    case_ids = exp.get("case_ids")
    if isinstance(case_ids, (str, bytes)) or not isinstance(case_ids, (list, tuple)):
        raise TypeError("exp.case_ids must be a YAML list, for example ['1', '2'].")
    if not case_ids:
        raise ValueError("exp.case_ids must contain at least one case ID.")

    prefix = str(exp.get("case_prefix", "")).strip().strip("_")
    width = int(exp.get("case_id_width", 4))
    if width < 0:
        raise ValueError("exp.case_id_width must be non-negative.")
    filename_pattern = str(
        exp.get("projection_filename_pattern", "{case_name}.npz")
    )

    cases: list[ProjectionCaseSpec] = []
    seen_names: set[str] = set()
    for case_id in case_ids:
        requested_id, case_name = _case_name(case_id, prefix, width)
        if case_name in seen_names:
            raise ValueError(f"Duplicate resolved case name: {case_name}")
        seen_names.add(case_name)

        case_id_padded = (
            requested_id.zfill(width) if requested_id.isdigit() else requested_id
        )
        try:
            relative_name = filename_pattern.format(
                case_id=requested_id,
                case_id_padded=case_id_padded,
                case_name=case_name,
                case_prefix=prefix,
            )
        except KeyError as exc:
            raise ValueError(
                "Unknown placeholder in exp.projection_filename_pattern: "
                f"{exc.args[0]!r}."
            ) from exc

        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError(
                "exp.projection_filename_pattern must produce a path relative "
                "to exp.projection_npz_dir."
            )
        projection_npz = directory / relative_path
        if not projection_npz.is_file():
            raise FileNotFoundError(
                f"Projection NPZ for case {case_name!r} was not found: "
                f"{projection_npz}"
            )
        cases.append(
            ProjectionCaseSpec(
                requested_id=requested_id,
                case_name=case_name,
                projection_npz=projection_npz,
            )
        )

    return cases


def make_case_config(
    base_config: Mapping[str, Any], case: ProjectionCaseSpec
) -> dict[str, Any]:
    """Return an isolated single-case trainer configuration."""

    config = copy.deepcopy(dict(base_config))
    exp = config.setdefault("exp", {})
    exp["projection_npz"] = str(case.projection_npz)
    exp["current_model_id"] = case.case_name

    # Prediction-only mode deliberately has no reference-volume dependency and
    # therefore cannot compute a ground-truth metric.
    if bool(exp.get("prediction_only", False)):
        exp.pop("reference_volume_npz", None)
    return config


__all__ = [
    "ProjectionCaseSpec",
    "make_case_config",
    "resolve_projection_cases",
]
