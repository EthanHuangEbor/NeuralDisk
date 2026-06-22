from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_FEATURES: list[str] = [
    "rpm",
    "omega",
    "volume_fraction",
    "Ri",
    "Ro",
    "B",
    "h_hub",
    "h_rim",
    "r_min",
    "eta",
    "n_control_points",
    "degree",
]


FEATURE_ALIASES: dict[str, list[str]] = {
    "rpm": ["rpm", "rotation_speed_rpm", "speed_rpm", "n_rpm"],
    "omega": ["omega", "angular_velocity", "angular_speed"],
    "volume_fraction": ["volume_fraction", "vf", "f_v", "fv", "target_volume_fraction"],
    "Ri": ["Ri", "R_i", "ri", "inner_radius", "hub_inner_radius"],
    "Ro": ["Ro", "R_o", "ro", "outer_radius", "rim_outer_radius"],
    "B": ["B", "thickness", "width", "section_width", "axial_width"],
    "h_hub": ["h_hub", "hub_keep", "hub_retained_size", "hub_reserved_size"],
    "h_rim": ["h_rim", "rim_keep", "rim_retained_size", "rim_reserved_size"],
    "r_min": ["r_min", "rmin", "minimum_member_size", "filter_radius", "rf"],
    "eta": ["eta", "density_threshold", "threshold_eta"],
    "n_control_points": ["n_control_points", "num_control_points", "control_points"],
    "degree": ["degree", "nurbs_degree", "spline_degree"],
}


@dataclass(slots=True)
class DatasetBuildResult:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    sample_ids: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    index_path: Path
    dataset_path: Path


def target_names_for_control_points(n_control_points: int) -> list[str]:
    names: list[str] = []
    for idx in range(1, n_control_points + 1):
        names.extend([f"P{idx}_x", f"P{idx}_z"])
    return names
