from __future__ import annotations

import numpy as np

from hyb_nurbs.schema import NurbsCurveSpec


def bspline_basis_matrix(u: np.ndarray, degree: int, knots: np.ndarray, n_ctrlpts: int) -> np.ndarray:
    """Evaluate B-spline basis functions with Cox-de Boor recursion."""
    u = np.asarray(u, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.size != n_ctrlpts + degree + 1:
        raise ValueError("Invalid knot vector length")
    if n_ctrlpts <= degree:
        raise ValueError("n_ctrlpts must be greater than degree")

    u_eval = np.clip(u, knots[degree], knots[-degree - 1])
    basis = np.zeros((u_eval.size, n_ctrlpts), dtype=float)
    for i in range(n_ctrlpts):
        basis[:, i] = ((knots[i] <= u_eval) & (u_eval < knots[i + 1])).astype(float)
    basis[np.isclose(u_eval, knots[-degree - 1]), :] = 0.0
    basis[np.isclose(u_eval, knots[-degree - 1]), -1] = 1.0

    for p in range(1, degree + 1):
        next_basis = np.zeros_like(basis)
        for i in range(n_ctrlpts):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1] if i + 1 < n_ctrlpts else 0.0
            if left_den > 0:
                next_basis[:, i] += (u_eval - knots[i]) / left_den * basis[:, i]
            if right_den > 0 and i + 1 < n_ctrlpts:
                next_basis[:, i] += (knots[i + p + 1] - u_eval) / right_den * basis[:, i + 1]
        basis = next_basis
    return basis


def evaluate_curve(curve: NurbsCurveSpec, u: np.ndarray) -> np.ndarray:
    """Evaluate 2D NURBS curve at parameters u."""
    ctrl = np.asarray(curve.control_points, dtype=float)
    weights = np.asarray(curve.weights, dtype=float)
    u_eval = np.asarray(u, dtype=float)
    if curve.is_closed and curve.fit_kind == "periodic":
        u_eval = np.mod(u_eval, 1.0)
        u_eval[np.isclose(u_eval, 1.0)] = 0.0
    basis = bspline_basis_matrix(u_eval, curve.degree, curve.knots, ctrl.shape[0])
    weighted = basis * weights[None, :]
    denom = weighted.sum(axis=1)
    denom = np.where(np.abs(denom) < 1e-14, 1.0, denom)
    return (weighted @ ctrl) / denom[:, None]
