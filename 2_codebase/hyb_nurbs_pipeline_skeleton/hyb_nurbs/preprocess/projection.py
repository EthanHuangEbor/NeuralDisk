from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np

from hyb_nurbs.schema import NodeDensityTable, SectionCloud

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def project_to_section(
    table: NodeDensityTable,
    axes: tuple[str, str] = ("x", "z"),
    thickness_axis: str = "y",
    mode: Literal["aggregate", "slice"] = "aggregate",
    slice_y: float | None = None,
    aggregate_density: Literal["max", "mean", "median"] = "max",
    rounding_digits: int = 8,
) -> SectionCloud:
    """Project 3D node-density data to a 2D section plane.

    Current HYB files have several Y layers but the NURBS section is an X-Z profile.
    In `aggregate` mode, coincident projected points are merged. Density defaults to max,
    which preserves material if any thickness-layer node is retained.

    Codex TODO:
    - implement optional y-slice filtering with tolerance;
    - expose aggregation policy;
    - keep source_node_ids for traceability.
    """
    i, j = _AXIS_INDEX[axes[0]], _AXIS_INDEX[axes[1]]
    pts = table.xyz[:, [i, j]]

    if mode == "slice":
        if slice_y is None:
            raise ValueError("slice_y is required when projection.mode='slice'")
        k = _AXIS_INDEX[thickness_axis]
        # TODO: choose tolerance based on mesh spacing; placeholder uses exact nearest layer.
        layer_values = np.unique(table.xyz[:, k])
        nearest = layer_values[np.argmin(np.abs(layer_values - slice_y))]
        mask = np.isclose(table.xyz[:, k], nearest)
        pts = pts[mask]
        rho = table.rho[mask]
        node_ids = table.node_id[mask]
    else:
        rho = table.rho
        node_ids = table.node_id

    keys = np.round(pts, rounding_digits)
    buckets: dict[tuple[float, float], list[int]] = defaultdict(list)
    for idx, key in enumerate(map(tuple, keys)):
        buckets[key].append(idx)

    out_pts = []
    out_rho = []
    out_src = []
    for inds in buckets.values():
        sub_pts = pts[inds]
        sub_rho = rho[inds]
        out_pts.append(sub_pts.mean(axis=0))
        if aggregate_density == "max":
            out_rho.append(float(np.max(sub_rho)))
        elif aggregate_density == "mean":
            out_rho.append(float(np.mean(sub_rho)))
        elif aggregate_density == "median":
            out_rho.append(float(np.median(sub_rho)))
        else:
            raise ValueError(aggregate_density)
        out_src.append([int(node_ids[ii]) for ii in inds])

    return SectionCloud(
        point_id=np.arange(len(out_pts), dtype=int),
        xy=np.asarray(out_pts, dtype=float),
        rho=np.asarray(out_rho, dtype=float),
        source_node_ids=out_src,
        axes=axes,
    )
