from __future__ import annotations

import numpy as np

from hyb_nurbs.schema import BoundaryLoop, FitMetrics, NurbsCurveSpec
from hyb_nurbs.nurbs.evaluate import evaluate_curve


def polygon_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def directed_nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Nearest distances from each src point to dst point set. Use scipy KDTree."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.size == 0 or dst.size == 0:
        return np.array([], dtype=float)
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(dst).query(src, k=1)
        return np.asarray(dist, dtype=float)
    except Exception:
        diff = src[:, None, :] - dst[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def evaluate_fit_metrics(loop: BoundaryLoop, curves: list[NurbsCurveSpec], n_eval: int = 1000) -> FitMetrics:
    """Compute mean, max, Hausdorff, and area error ratio."""
    source = np.asarray(loop.points, dtype=float)
    if len(source) > 1 and np.allclose(source[0], source[-1]):
        source_open = source[:-1]
    else:
        source_open = source

    samples: list[np.ndarray] = []
    per_curve = max(16, int(np.ceil(n_eval / max(len(curves), 1))))
    for curve in curves:
        u = np.linspace(0.0, 1.0, per_curve, endpoint=not curve.is_closed)
        samples.append(evaluate_curve(curve, u))
    fitted = np.vstack(samples) if samples else np.empty((0, 2), dtype=float)
    if len(fitted) and loop.is_closed:
        fitted_closed = np.vstack([fitted, fitted[0]])
    else:
        fitted_closed = fitted

    src_to_fit = directed_nearest_distances(source_open, fitted)
    fit_to_src = directed_nearest_distances(fitted, source_open)
    mean_error = float(src_to_fit.mean()) if src_to_fit.size else float("inf")
    max_error = float(src_to_fit.max()) if src_to_fit.size else float("inf")
    hausdorff = max(
        float(src_to_fit.max()) if src_to_fit.size else float("inf"),
        float(fit_to_src.max()) if fit_to_src.size else float("inf"),
    )
    source_area = polygon_area(source)
    fit_area = polygon_area(fitted_closed)
    area_error = abs(fit_area - source_area) / source_area if source_area > 0 else float("inf")
    return FitMetrics(
        mean_error_mm=mean_error,
        max_error_mm=max_error,
        hausdorff_error_mm=hausdorff,
        area_error_ratio=float(area_error),
        n_control_points=int(sum(curve.control_points.shape[0] for curve in curves)),
        n_samples=int(len(fitted)),
    )
