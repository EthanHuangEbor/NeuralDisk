from __future__ import annotations

import numpy as np

from hyb_nurbs.schema import BoundaryLoop


def signed_area(points: np.ndarray) -> float:
    """Return signed polygon area. Positive usually means CCW."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_closed(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Append first point when loop endpoint gap exceeds tolerance."""
    p = np.asarray(points, dtype=float)
    if len(p) == 0:
        return p
    if np.linalg.norm(p[0] - p[-1]) > tolerance:
        p = np.vstack([p, p[0]])
    return p


def classify_and_filter_loops(
    loops: list[BoundaryLoop],
    min_component_area_mm2: float,
    min_hole_area_mm2: float,
) -> list[BoundaryLoop]:
    """Remove small fragments and assign outer/hole roles by area and containment."""
    prepared: list[tuple[BoundaryLoop, float, object]] = []
    for loop in loops:
        for pts in _valid_polygon_rings(loop.points):
            area = abs(signed_area(pts))
            if area <= 0.0:
                continue
            prepared.append(
                (
                    BoundaryLoop(pts, "unknown", loop.component_id, True, segment_tags=list(loop.segment_tags)),
                    area,
                    _shape_polygon(pts),
                )
            )

    prepared.sort(key=lambda item: item[1], reverse=True)
    out: list[BoundaryLoop] = []
    for loop, area, polygon in prepared:
        sample = _representative_point(polygon, loop.points)
        containing = sum(
            1
            for _, other_area, other_polygon in prepared
            if other_area > area and _shape_contains_point(other_polygon, sample)
        )
        role = "hole" if containing % 2 == 1 else "outer"
        if role == "outer" and area < min_component_area_mm2:
            continue
        if role == "hole" and area < min_hole_area_mm2:
            continue

        pts = loop.points.copy()
        current_area = signed_area(pts)
        if role == "outer" and current_area < 0:
            pts = pts[::-1]
        elif role == "hole" and current_area > 0:
            pts = pts[::-1]
        pts = ensure_closed(pts, tolerance=1e-9)
        out.append(BoundaryLoop(points=pts, role=role, component_id=len(out), is_closed=True, segment_tags=list(loop.segment_tags)))
    return out


def resample_loop_by_arclength(loop: BoundaryLoop, spacing_mm: float) -> BoundaryLoop:
    """Uniformly resample a closed loop by arc length."""
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be positive")
    pts = ensure_closed(loop.points, tolerance=1e-9)
    if len(pts) < 4:
        return loop

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-12]
    pts = pts[keep]
    pts = ensure_closed(pts, tolerance=1e-9)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return loop

    cumulative = np.r_[0.0, np.cumsum(seg)]
    n_samples = max(4, int(np.ceil(total / spacing_mm)))
    targets = np.linspace(0.0, total, n_samples, endpoint=False)
    new_pts = np.empty((n_samples, 2), dtype=float)
    for dim in range(2):
        new_pts[:, dim] = np.interp(targets, cumulative, pts[:, dim])
    new_pts = np.vstack([new_pts, new_pts[0]])
    return BoundaryLoop(points=new_pts, role=loop.role, component_id=loop.component_id, is_closed=True, segment_tags=list(loop.segment_tags))


def split_by_curvature_or_corners(loop: BoundaryLoop, corner_angle_deg: float) -> list[np.ndarray]:
    """Split a loop into curve segments at sharp corners.

    Smooth regions may be fitted with C1/C2 NURBS; true corners should be C0 joints.
    """
    pts = ensure_closed(loop.points, tolerance=1e-9)
    if len(pts) < 6:
        return [pts]
    open_pts = pts[:-1]
    prev_pts = np.roll(open_pts, 1, axis=0)
    next_pts = np.roll(open_pts, -1, axis=0)
    incoming = open_pts - prev_pts
    outgoing = next_pts - open_pts
    in_norm = np.linalg.norm(incoming, axis=1)
    out_norm = np.linalg.norm(outgoing, axis=1)
    valid = (in_norm > 1e-12) & (out_norm > 1e-12)
    cos_turn = np.ones(len(open_pts), dtype=float)
    cos_turn[valid] = np.sum(incoming[valid] * outgoing[valid], axis=1) / (in_norm[valid] * out_norm[valid])
    interior_angle = np.degrees(np.arccos(np.clip(-cos_turn, -1.0, 1.0)))
    corner_indices = np.flatnonzero(interior_angle < corner_angle_deg)
    if len(corner_indices) == 0:
        return [pts]

    segments: list[np.ndarray] = []
    for start, end in zip(corner_indices, np.roll(corner_indices, -1)):
        if end <= start:
            segment = np.vstack([open_pts[start:], open_pts[: end + 1]])
        else:
            segment = open_pts[start : end + 1]
        if len(segment) >= 2:
            segments.append(segment)
    return segments or [pts]


def segment_kind(points: np.ndarray, straight_tolerance_mm: float = 0.05) -> str:
    """Classify a segment as straight or smooth for fitting/export metadata."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return "line"
    chord = pts[-1] - pts[0]
    length = float(np.linalg.norm(chord))
    if length <= 1e-12:
        return "corner"
    distances = np.abs(np.cross(chord, pts - pts[0])) / length
    return "line" if float(np.max(distances)) <= straight_tolerance_mm else "smooth"


def _valid_polygon_rings(points: np.ndarray) -> list[np.ndarray]:
    pts = ensure_closed(points, tolerance=1e-9)
    if len(pts) < 4:
        return []
    try:
        from shapely.geometry import LineString, Polygon
        from shapely.ops import polygonize, unary_union

        polygon = Polygon(pts)
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            return [_ring_from_coords(polygon.exterior.coords)]

        linework = unary_union([LineString(pts)])
        rings = []
        for poly in polygonize(linework):
            if poly.is_valid and poly.area > 0:
                rings.append(_ring_from_coords(poly.exterior.coords))
        return rings
    except Exception:
        if _has_self_intersection(pts):
            return []
        return [pts] if abs(signed_area(pts)) > 0 else []


def _shape_polygon(points: np.ndarray) -> object:
    try:
        from shapely.geometry import Polygon

        return Polygon(points)
    except Exception:
        return points


def _representative_point(polygon: object, fallback_points: np.ndarray) -> np.ndarray:
    try:
        point = polygon.representative_point()
        return np.array([point.x, point.y], dtype=float)
    except Exception:
        return _interior_point(fallback_points)


def _shape_contains_point(polygon: object, point: np.ndarray) -> bool:
    try:
        from shapely.geometry import Point

        return bool(polygon.contains(Point(float(point[0]), float(point[1]))))
    except Exception:
        return _point_in_polygon(point, polygon)


def _ring_from_coords(coords: object) -> np.ndarray:
    return ensure_closed(np.asarray(coords, dtype=float), tolerance=1e-9)


def _interior_point(points: np.ndarray) -> np.ndarray:
    p = points[:-1] if np.allclose(points[0], points[-1]) else points
    centroid = p.mean(axis=0)
    if _point_in_polygon(centroid, points):
        return centroid
    return p[0]


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    p = ensure_closed(polygon, tolerance=1e-9)
    inside = False
    for a, b in zip(p[:-1], p[1:]):
        x1, y1 = a
        x2, y2 = b
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def _has_self_intersection(points: np.ndarray) -> bool:
    p = ensure_closed(points, tolerance=1e-9)
    n = len(p) - 1
    if n < 4:
        return False
    for i in range(n):
        a1, a2 = p[i], p[i + 1]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b1, b2 = p[j], p[j + 1]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    eps = 1e-10
    if abs(o1) < eps and _on_segment(a1, b1, a2):
        return True
    if abs(o2) < eps and _on_segment(a1, b2, a2):
        return True
    if abs(o3) < eps and _on_segment(b1, a1, b2):
        return True
    if abs(o4) < eps and _on_segment(b1, a2, b2):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool(np.all(b >= np.minimum(a, c) - 1e-10) and np.all(b <= np.maximum(a, c) + 1e-10))
