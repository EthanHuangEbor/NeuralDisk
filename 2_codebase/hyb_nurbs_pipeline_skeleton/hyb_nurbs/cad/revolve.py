from __future__ import annotations

import numpy as np

from hyb_nurbs.schema import NurbsCurveSpec
from hyb_nurbs.nurbs.evaluate import evaluate_curve


def revolve_curves_to_surface(
    curves: list[NurbsCurveSpec],
    *,
    axis: str = "z",
    angle_deg: float = 360.0,
    n_sections: int = 96,
):
    """Generate a revolved surface/mesh from 2D section NURBS curves.

    Codex TODO:
    - for a lightweight route, sample curves and create STL/PLY mesh;
    - for CAD route, use pythonocc-core or geomdl exchange when available;
    - keep axis convention explicit: section axes are usually X-Z, turbine axis may be Z or X depending on ANSYS model.
    """
    if axis.lower() != "z":
        raise NotImplementedError("Lightweight revolve currently supports section X-Z revolved about the Z axis")
    if not curves:
        return {"vertices": np.empty((0, 3)), "faces": np.empty((0, 3), dtype=int)}

    profile = np.vstack([evaluate_curve(curve, np.linspace(0.0, 1.0, 160, endpoint=False)) for curve in curves])
    theta = np.linspace(0.0, np.deg2rad(angle_deg), n_sections, endpoint=abs(angle_deg) >= 360.0)
    vertices = []
    for angle in theta:
        ca, sa = np.cos(angle), np.sin(angle)
        for x, z in profile:
            r = abs(x)
            vertices.append((r * ca, r * sa, z))
    vertices = np.asarray(vertices, dtype=float)

    n_profile = len(profile)
    n_ring = len(theta)
    faces: list[tuple[int, int, int]] = []
    wrap = abs(angle_deg) >= 360.0
    ring_limit = n_ring if wrap else n_ring - 1
    for i in range(ring_limit):
        ni = (i + 1) % n_ring
        for j in range(n_profile):
            nj = (j + 1) % n_profile
            a = i * n_profile + j
            b = ni * n_profile + j
            c = ni * n_profile + nj
            d = i * n_profile + nj
            faces.append((a, b, c))
            faces.append((a, c, d))
    return {"vertices": vertices, "faces": np.asarray(faces, dtype=int)}
