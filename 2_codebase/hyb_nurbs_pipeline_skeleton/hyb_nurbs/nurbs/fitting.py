from __future__ import annotations

import numpy as np

from hyb_nurbs.boundary.postprocess import split_by_curvature_or_corners, segment_kind
from hyb_nurbs.schema import BoundaryLoop, FitResult, NurbsCurveSpec
from hyb_nurbs.nurbs.evaluate import bspline_basis_matrix
from hyb_nurbs.validation.metrics import evaluate_fit_metrics


def chord_length_parameters(points: np.ndarray, centripetal: bool = False) -> np.ndarray:
    """Compute normalized chord-length or centripetal parameters in [0, 1]."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.array([], dtype=float)
    if len(pts) == 1:
        return np.array([0.0], dtype=float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if centripetal:
        d = np.sqrt(d)
    cumulative = np.r_[0.0, np.cumsum(d)]
    total = cumulative[-1]
    if total <= 0:
        return np.linspace(0.0, 1.0, len(pts))
    return cumulative / total


def open_uniform_knot_vector(n_ctrlpts: int, degree: int) -> np.ndarray:
    """Create clamped open-uniform knot vector."""
    if n_ctrlpts <= degree:
        raise ValueError("n_ctrlpts must be greater than degree")
    knots = np.zeros(n_ctrlpts + degree + 1, dtype=float)
    knots[-degree - 1 :] = 1.0
    n_internal = n_ctrlpts - degree - 1
    if n_internal > 0:
        knots[degree + 1 : degree + 1 + n_internal] = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return knots


def fit_bspline_least_squares(
    points: np.ndarray,
    degree: int = 3,
    n_ctrlpts: int = 12,
    lambda_smooth: float = 1e-4,
    closed: bool = False,
    closed_loop_mode: str = "periodic",
    fit_kind: str | None = None,
) -> NurbsCurveSpec:
    """Fit a B-spline as a NURBS with fixed unit weights.

    Implementation notes for Codex:
    - build basis matrix N(u);
    - solve min ||N P - Q||^2 + lambda_smooth ||D2 P||^2;
    - for clamped open curves, force first/last control points near endpoints;
    - for closed loops, either use periodic basis or fit segmented clamped pieces.
    """
    pts = np.asarray(points, dtype=float)
    if closed and closed_loop_mode == "periodic":
        return _fit_periodic_bspline(pts, degree, n_ctrlpts, lambda_smooth)
    if closed and len(pts) > 1 and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    if len(pts) < degree + 1:
        raise ValueError("Not enough points to fit requested degree")

    n_ctrlpts = int(max(n_ctrlpts, degree + 1))
    knots = open_uniform_knot_vector(n_ctrlpts, degree)
    u = chord_length_parameters(pts)
    basis = bspline_basis_matrix(u, degree, knots, n_ctrlpts)

    lhs_blocks = [basis]
    rhs_blocks = [pts]
    if lambda_smooth > 0 and n_ctrlpts > 2:
        d2 = np.zeros((n_ctrlpts - 2, n_ctrlpts), dtype=float)
        for i in range(n_ctrlpts - 2):
            d2[i, i : i + 3] = (1.0, -2.0, 1.0)
        lhs_blocks.append(np.sqrt(lambda_smooth) * d2)
        rhs_blocks.append(np.zeros((d2.shape[0], 2), dtype=float))

    constraint_weight = 1e3
    endpoint_rows = np.zeros((2, n_ctrlpts), dtype=float)
    endpoint_rows[0, 0] = constraint_weight
    endpoint_rows[1, -1] = constraint_weight
    lhs_blocks.append(endpoint_rows)
    rhs_blocks.append(np.vstack([pts[0], pts[-1]]) * constraint_weight)

    lhs = np.vstack(lhs_blocks)
    rhs = np.vstack(rhs_blocks)
    control_points, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    weights = np.ones(n_ctrlpts, dtype=float)
    return NurbsCurveSpec(
        degree=degree,
        control_points=control_points,
        weights=weights,
        knots=knots,
        is_closed=closed,
        role="unknown",
        segment_id=0,
        fit_kind=fit_kind or ("closed_clamped" if closed else "smooth"),
    )


def _fit_periodic_bspline(
    points: np.ndarray,
    degree: int,
    n_ctrlpts: int,
    lambda_smooth: float,
) -> NurbsCurveSpec:
    pts = np.asarray(points, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < degree + 1:
        raise ValueError("Not enough points to fit requested degree")

    n_unique = int(max(n_ctrlpts, degree + 1))
    n_ext = n_unique + degree
    raw_knots = np.arange(n_ext + degree + 1, dtype=float)
    knots = (raw_knots - degree) / float(n_unique)
    u = chord_length_parameters(np.vstack([pts, pts[0]]))[:-1]
    basis_ext = bspline_basis_matrix(u, degree, knots, n_ext)
    basis = np.zeros((len(u), n_unique), dtype=float)
    for j in range(n_ext):
        basis[:, j % n_unique] += basis_ext[:, j]

    lhs_blocks = [basis]
    rhs_blocks = [pts]
    if lambda_smooth > 0 and n_unique > 3:
        d2 = np.zeros((n_unique, n_unique), dtype=float)
        for i in range(n_unique):
            d2[i, i] = -2.0
            d2[i, (i - 1) % n_unique] = 1.0
            d2[i, (i + 1) % n_unique] = 1.0
        lhs_blocks.append(np.sqrt(lambda_smooth) * d2)
        rhs_blocks.append(np.zeros((n_unique, 2), dtype=float))

    unique_ctrl, *_ = np.linalg.lstsq(np.vstack(lhs_blocks), np.vstack(rhs_blocks), rcond=None)
    control_points = np.vstack([unique_ctrl, unique_ctrl[:degree]])
    return NurbsCurveSpec(
        degree=degree,
        control_points=control_points,
        weights=np.ones(control_points.shape[0], dtype=float),
        knots=knots,
        is_closed=True,
        role="unknown",
        segment_id=0,
        fit_kind="periodic",
    )


def fit_segmented_loop(
    loop: BoundaryLoop,
    *,
    degree: int,
    n_ctrlpts: int,
    lambda_smooth: float,
    corner_angle_deg: float,
) -> list[NurbsCurveSpec]:
    """Fit a closed loop as C0 joined clamped pieces split at corners."""
    segments = split_by_curvature_or_corners(loop, corner_angle_deg)
    curves: list[NurbsCurveSpec] = []
    total_len = sum(_polyline_length(seg) for seg in segments) or 1.0
    for idx, segment in enumerate(segments):
        kind = segment_kind(segment)
        seg_degree = 1 if kind == "line" else degree
        seg_ctrl = max(seg_degree + 1, int(round(n_ctrlpts * _polyline_length(segment) / total_len)))
        curve = fit_bspline_least_squares(
            segment,
            degree=seg_degree,
            n_ctrlpts=seg_ctrl,
            lambda_smooth=lambda_smooth,
            closed=False,
            closed_loop_mode="clamped",
            fit_kind=kind,
        )
        curve.segment_id = idx
        curve.role = loop.role
        curves.append(curve)
    return curves


def refine_fit_until_tolerance(
    loop: BoundaryLoop,
    *,
    degree: int,
    min_ctrlpts: int,
    max_ctrlpts: int,
    initial_control_spacing_mm: float,
    lambda_smooth: float,
    tolerances: dict,
    closed_loop_mode: str = "periodic",
    corner_angle_deg: float = 135.0,
) -> FitResult:
    """Fit and increase control count until error tolerances are met or max iterations reached."""
    pts = np.asarray(loop.points, dtype=float)
    if len(pts) < degree + 2:
        raise ValueError("Loop has too few points to fit")

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg_lengths.sum())
    spacing_guess = int(np.ceil(length / max(initial_control_spacing_mm, 1e-9)))
    initial = min(max_ctrlpts, max(min_ctrlpts, spacing_guess))
    candidates = list(range(max(degree + 1, min_ctrlpts), max_ctrlpts + 1, 4))
    if initial not in candidates:
        candidates.append(initial)
        candidates.sort()
    if max_ctrlpts not in candidates:
        candidates.append(max_ctrlpts)
    if not candidates:
        candidates = [max_ctrlpts]

    best: FitResult | None = None
    first_passing: FitResult | None = None
    for n_ctrl in candidates:
        if loop.is_closed and closed_loop_mode == "segmented":
            curves = fit_segmented_loop(
                loop,
                degree=degree,
                n_ctrlpts=n_ctrl,
                lambda_smooth=lambda_smooth,
                corner_angle_deg=corner_angle_deg,
            )
        else:
            curve = fit_bspline_least_squares(
                pts,
                degree=degree,
                n_ctrlpts=n_ctrl,
                lambda_smooth=lambda_smooth,
                closed=loop.is_closed,
                closed_loop_mode=closed_loop_mode,
            )
            curve.role = loop.role
            curve.segment_id = loop.component_id
            curves = [curve]
        metrics = evaluate_fit_metrics(loop, curves, n_eval=max(400, len(pts) * 4))
        result = FitResult(loop=loop, curves=curves, metrics=metrics)
        if best is None or metrics.hausdorff_error_mm < best.metrics.hausdorff_error_mm:
            best = result
        if (
            metrics.mean_error_mm <= float(tolerances.get("mean_mm", np.inf))
            and metrics.max_error_mm <= float(tolerances.get("max_mm", np.inf))
            and metrics.area_error_ratio <= float(tolerances.get("area_ratio", np.inf))
        ):
            first_passing = result
            break

    assert best is not None
    if first_passing is None:
        return best
    return _compress_passing_fit(
        loop,
        first_passing,
        degree=degree,
        min_ctrlpts=min_ctrlpts,
        lambda_smooth=lambda_smooth,
        tolerances=tolerances,
        closed_loop_mode=closed_loop_mode,
        corner_angle_deg=corner_angle_deg,
    )


def _compress_passing_fit(
    loop: BoundaryLoop,
    result: FitResult,
    *,
    degree: int,
    min_ctrlpts: int,
    lambda_smooth: float,
    tolerances: dict,
    closed_loop_mode: str,
    corner_angle_deg: float,
) -> FitResult:
    current_n = result.metrics.n_control_points
    for n_ctrl in range(max(degree + 1, min_ctrlpts), current_n):
        if loop.is_closed and closed_loop_mode == "segmented":
            curves = fit_segmented_loop(
                loop,
                degree=degree,
                n_ctrlpts=n_ctrl,
                lambda_smooth=lambda_smooth,
                corner_angle_deg=corner_angle_deg,
            )
        else:
            curve = fit_bspline_least_squares(
                loop.points,
                degree=degree,
                n_ctrlpts=n_ctrl,
                lambda_smooth=lambda_smooth,
                closed=loop.is_closed,
                closed_loop_mode=closed_loop_mode,
            )
            curve.role = loop.role
            curve.segment_id = loop.component_id
            curves = [curve]
        metrics = evaluate_fit_metrics(loop, curves, n_eval=max(400, len(loop.points) * 4))
        if (
            metrics.mean_error_mm <= float(tolerances.get("mean_mm", np.inf))
            and metrics.max_error_mm <= float(tolerances.get("max_mm", np.inf))
            and metrics.area_error_ratio <= float(tolerances.get("area_ratio", np.inf))
        ):
            return FitResult(loop=loop, curves=curves, metrics=metrics)
    return result


def _polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
