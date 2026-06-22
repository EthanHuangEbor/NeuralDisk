from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(slots=True)
class NodeDensityTable:
    """Joined ANSYS node coordinates and node-wise topology densities."""
    node_id: np.ndarray  # shape [N], int
    xyz: np.ndarray      # shape [N, 3], working units, usually mm
    rho: np.ndarray      # shape [N], float in [0, 1]
    source_node_file: Path | None = None
    source_density_file: Path | None = None

    def validate(self) -> None:
        assert self.node_id.ndim == 1
        assert self.xyz.shape == (self.node_id.size, 3)
        assert self.rho.shape == (self.node_id.size,)
        assert np.all(np.isfinite(self.xyz))
        assert np.all(np.isfinite(self.rho))


@dataclass(slots=True)
class SectionCloud:
    """2D section-point cloud after projection and optional aggregation."""
    point_id: np.ndarray        # shape [M]
    xy: np.ndarray              # shape [M, 2], section coordinates, usually x-z in mm
    rho: np.ndarray             # shape [M]
    source_node_ids: list[list[int]] = field(default_factory=list)
    axes: tuple[str, str] = ("x", "z")


@dataclass(slots=True)
class BoundaryLoop:
    """A closed 2D boundary loop sampled as a polyline."""
    points: np.ndarray          # shape [K, 2]
    role: Literal["outer", "hole", "unknown"] = "unknown"
    component_id: int = 0
    is_closed: bool = True
    segment_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NurbsCurveSpec:
    """Serializable NURBS/B-spline curve definition."""
    degree: int
    control_points: np.ndarray  # shape [C, 2]
    weights: np.ndarray         # shape [C]
    knots: np.ndarray           # shape [C + degree + 1]
    is_closed: bool
    role: str = "unknown"
    segment_id: int = 0
    fit_kind: str = "smooth"


@dataclass(slots=True)
class FitMetrics:
    mean_error_mm: float
    max_error_mm: float
    hausdorff_error_mm: float
    area_error_ratio: float
    n_control_points: int
    n_samples: int


@dataclass(slots=True)
class FitResult:
    loop: BoundaryLoop
    curves: list[NurbsCurveSpec]
    metrics: FitMetrics
