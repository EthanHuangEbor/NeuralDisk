from __future__ import annotations

import numpy as np

from hyb_nurbs.schema import BoundaryLoop, SectionCloud

TaggedSegment = tuple[np.ndarray, np.ndarray, str]


def extract_iso_contours_tri(
    cloud: SectionCloud,
    eta: float = 0.5,
    min_points: int = 20,
) -> list[BoundaryLoop]:
    """Extract iso-density boundary loops rho=eta from scattered section points.

    Recommended implementation for current node-only data:
    1. Build scipy.spatial.Delaunay triangulation on cloud.xy.
    2. For each triangle crossing eta, linearly interpolate edge intersection points.
    3. Stitch line segments into polylines with a KD-tree/tolerance.
    4. Close loops and discard short open fragments.

    Alternative implementation:
    - use matplotlib.tri.Triangulation + tricontour to get path vertices;
    - then convert paths into BoundaryLoop objects.

    Codex TODO: implement robust segment stitching and loop extraction.
    """
    from scipy.spatial import Delaunay

    xy = np.asarray(cloud.xy, dtype=float)
    rho = np.asarray(cloud.rho, dtype=float)
    if xy.shape[0] < 3:
        return []

    tri = Delaunay(xy)
    segments: list[TaggedSegment] = []
    for simplex in tri.simplices:
        pts = xy[simplex]
        vals = rho[simplex]
        intersections: list[np.ndarray] = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va = vals[a] - eta
            vb = vals[b] - eta
            if np.isclose(va, 0.0) and np.isclose(vb, 0.0):
                continue
            if (va > 0 and vb > 0) or (va < 0 and vb < 0):
                continue
            denom = vals[b] - vals[a]
            if np.isclose(denom, 0.0):
                continue
            t = (eta - vals[a]) / denom
            if -1e-12 <= t <= 1.0 + 1e-12:
                intersections.append(pts[a] + np.clip(t, 0.0, 1.0) * (pts[b] - pts[a]))

        unique: list[np.ndarray] = []
        for p in intersections:
            if not any(np.linalg.norm(p - q) < 1e-9 for q in unique):
                unique.append(p)
        if len(unique) == 2 and np.linalg.norm(unique[0] - unique[1]) > 1e-12:
            segments.append((unique[0], unique[1], "iso"))
        elif len(unique) > 2:
            center = np.mean(unique, axis=0)
            unique.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            for a, b in zip(unique[:-1], unique[1:]):
                if np.linalg.norm(a - b) > 1e-12:
                    segments.append((a, b, "iso"))

    # If the retained material reaches the section/design-domain boundary, the
    # rho=eta contour is open inside the triangulation. Close the material
    # boundary with the corresponding convex-hull edge portions.
    for ia, ib in tri.convex_hull:
        ra, rb = rho[ia], rho[ib]
        pa, pb = xy[ia], xy[ib]
        if ra >= eta and rb >= eta:
            segments.append((pa, pb, "convex_hull_closure"))
        elif (ra >= eta) != (rb >= eta):
            denom = rb - ra
            if not np.isclose(denom, 0.0):
                t = (eta - ra) / denom
                cross = pa + np.clip(t, 0.0, 1.0) * (pb - pa)
                high = pa if ra >= eta else pb
                segments.append((high, cross, "convex_hull_closure"))

    loops = _stitch_segments_to_loops(segments, min_points=min_points)
    return [
        BoundaryLoop(points=p, role="unknown", component_id=i, is_closed=True, segment_tags=tags)
        for i, (p, tags) in enumerate(loops)
    ]


def extract_iso_contours_grid(
    cloud: SectionCloud,
    eta: float = 0.5,
    grid_resolution_mm: float = 0.15,
) -> list[BoundaryLoop]:
    """Grid-interpolate density and extract contours with skimage.measure.find_contours.

    Use this when triangulation contouring is unstable. Preserve coordinate transforms
    from image indices back to section coordinates.
    """
    from scipy.interpolate import griddata
    from skimage import measure

    xy = np.asarray(cloud.xy, dtype=float)
    rho = np.asarray(cloud.rho, dtype=float)
    if xy.shape[0] < 3:
        return []

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xs = np.arange(xmin, xmax + grid_resolution_mm, grid_resolution_mm)
    ys = np.arange(ymin, ymax + grid_resolution_mm, grid_resolution_mm)
    gx, gy = np.meshgrid(xs, ys)
    values = griddata(xy, rho, (gx, gy), method="linear")
    nearest = griddata(xy, rho, (gx, gy), method="nearest")
    values = np.where(np.isfinite(values), values, nearest)

    loops: list[BoundaryLoop] = []
    for cid, contour in enumerate(measure.find_contours(values, eta)):
        if len(contour) < 20:
            continue
        rows, cols = contour[:, 0], contour[:, 1]
        pts = np.column_stack([np.interp(cols, np.arange(len(xs)), xs), np.interp(rows, np.arange(len(ys)), ys)])
        tags = ["iso"] * max(len(pts) - 1, 0)
        if np.linalg.norm(pts[0] - pts[-1]) > max(grid_resolution_mm * 2.0, 1e-9):
            pts, closure_tags = _close_polyline_along_bbox(pts, (xmin, ymin, xmax, ymax))
            tags.extend(closure_tags)
        else:
            pts[-1] = pts[0]
        loops.append(BoundaryLoop(points=pts, role="unknown", component_id=cid, is_closed=True, segment_tags=tags))
    return loops


def extract_iso_contours_mesh(
    cloud: SectionCloud,
    elements: list[list[int]],
    eta: float = 0.5,
    min_points: int = 20,
) -> list[BoundaryLoop]:
    """Extract iso contours from real ANSYS element connectivity when available.

    Elements may be triangles/quads/higher-order rows. The contour uses each
    element's projected polygon boundary and, for non-triangles, a fan
    triangulation in section space. Node ids are matched through
    ``SectionCloud.source_node_ids`` so this works after projection/aggregation.
    """
    xy = np.asarray(cloud.xy, dtype=float)
    rho = np.asarray(cloud.rho, dtype=float)
    node_to_point: dict[int, int] = {}
    for point_index, node_ids in enumerate(cloud.source_node_ids):
        for node_id in node_ids:
            node_to_point[int(node_id)] = point_index

    segments: list[TaggedSegment] = []
    for element in elements:
        point_ids = [node_to_point[n] for n in element if n in node_to_point]
        point_ids = list(dict.fromkeys(point_ids))
        if len(point_ids) < 3:
            continue
        pts = xy[point_ids]
        center = pts.mean(axis=0)
        point_ids.sort(key=lambda idx: np.arctan2(xy[idx, 1] - center[1], xy[idx, 0] - center[0]))
        triangles = [(point_ids[0], point_ids[i], point_ids[i + 1]) for i in range(1, len(point_ids) - 1)]
        for tri in triangles:
            tri_pts = xy[list(tri)]
            vals = rho[list(tri)]
            intersections: list[np.ndarray] = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                va = vals[a] - eta
                vb = vals[b] - eta
                if (va > 0 and vb > 0) or (va < 0 and vb < 0):
                    continue
                denom = vals[b] - vals[a]
                if np.isclose(denom, 0.0):
                    continue
                t = (eta - vals[a]) / denom
                if -1e-12 <= t <= 1.0 + 1e-12:
                    intersections.append(tri_pts[a] + np.clip(t, 0.0, 1.0) * (tri_pts[b] - tri_pts[a]))
            if len(intersections) == 2 and np.linalg.norm(intersections[0] - intersections[1]) > 1e-12:
                segments.append((intersections[0], intersections[1], "mesh_iso"))

    loops = _stitch_segments_to_loops(segments, min_points=min_points)
    return [
        BoundaryLoop(points=p, role="unknown", component_id=i, is_closed=True, segment_tags=tags)
        for i, (p, tags) in enumerate(loops)
    ]


def extract_alpha_shape_boundary(
    cloud: SectionCloud,
    eta: float = 0.5,
    alpha: float | None = None,
) -> list[BoundaryLoop]:
    """Fallback: compute a concave hull from points with rho >= eta.

    This is less faithful to gray-region iso-density information than tri/grid contouring.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(cloud.xy, dtype=float)[np.asarray(cloud.rho) >= eta]
    if pts.shape[0] < 3:
        return []
    hull = ConvexHull(pts)
    loop = pts[hull.vertices]
    loop = np.vstack([loop, loop[0]])
    return [BoundaryLoop(points=loop, role="unknown", component_id=0, is_closed=True, segment_tags=["alpha_shape"] * (len(loop) - 1))]


def _stitch_segments_to_loops(
    segments: list[TaggedSegment],
    *,
    min_points: int,
) -> list[tuple[np.ndarray, list[str]]]:
    if not segments:
        return []

    all_pts = np.vstack([p for a, b, _ in segments for p in (a, b)])
    diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    tol = max(diag * 1e-9, 1e-8)

    nodes: list[np.ndarray] = []
    key_to_node: dict[tuple[int, int], int] = {}
    adjacency: dict[int, set[int]] = {}
    edge_tags: dict[tuple[int, int], str] = {}

    def node_id(point: np.ndarray) -> int:
        key = tuple(np.round(point / tol).astype(int))
        if key not in key_to_node:
            key_to_node[key] = len(nodes)
            nodes.append(np.asarray(point, dtype=float))
            adjacency[key_to_node[key]] = set()
        return key_to_node[key]

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for a, b, tag in segments:
        ia = node_id(a)
        ib = node_id(b)
        if ia == ib:
            continue
        adjacency[ia].add(ib)
        adjacency[ib].add(ia)
        edge_tags[edge_key(ia, ib)] = tag

    visited_edges: set[tuple[int, int]] = set()
    loops: list[tuple[np.ndarray, list[str]]] = []

    for start in list(adjacency):
        for first in list(adjacency[start]):
            if edge_key(start, first) in visited_edges:
                continue
            path = [start, first]
            tags = [edge_tags.get(edge_key(start, first), "unknown")]
            visited_edges.add(edge_key(start, first))
            prev, curr = start, first

            while True:
                if curr == start:
                    break
                candidates = [n for n in adjacency[curr] if n != prev and edge_key(curr, n) not in visited_edges]
                if not candidates and start in adjacency[curr] and edge_key(curr, start) not in visited_edges:
                    candidates = [start]
                if not candidates:
                    break
                nxt = candidates[0]
                visited_edges.add(edge_key(curr, nxt))
                tags.append(edge_tags.get(edge_key(curr, nxt), "unknown"))
                path.append(nxt)
                prev, curr = curr, nxt

            if len(path) >= min_points and path[-1] == path[0]:
                pts = np.asarray([nodes[i] for i in path], dtype=float)
                if np.linalg.norm(pts[0] - pts[-1]) > tol:
                    pts = np.vstack([pts, pts[0]])
                loops.append((pts, tags))

    return loops


def _close_polyline_along_bbox(
    points: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, list[str]]:
    xmin, ymin, xmax, ymax = bbox
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts, []

    start = _snap_to_bbox(pts[0], bbox)
    end = _snap_to_bbox(pts[-1], bbox)
    perimeter = 2.0 * ((xmax - xmin) + (ymax - ymin))
    if perimeter <= 0:
        return np.vstack([pts, pts[0]]), ["grid_boundary_closure"]

    s0 = _bbox_abscissa(start, bbox)
    s1 = _bbox_abscissa(end, bbox)
    forward = (s0 - s1) % perimeter
    backward = (s1 - s0) % perimeter
    closure_s = [s1, s0] if forward <= backward else [s1, s0 - perimeter]
    extra: list[np.ndarray] = []
    corners_s = [0.0, xmax - xmin, xmax - xmin + ymax - ymin, 2 * (xmax - xmin) + ymax - ymin, perimeter]
    lo, hi = sorted(closure_s)
    for s in corners_s:
        shifted = s
        if shifted > max(closure_s):
            shifted -= perimeter
        if lo < shifted < hi:
            extra.append(_bbox_point(shifted % perimeter, bbox))
    if closure_s[0] > closure_s[1]:
        extra = extra[::-1]
    closed = np.vstack([pts, np.asarray(extra, dtype=float).reshape((-1, 2)) if extra else np.empty((0, 2)), pts[0]])
    return closed, ["grid_boundary_closure"] * (len(closed) - len(pts))


def _snap_to_bbox(point: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    p = np.asarray(point, dtype=float).copy()
    distances = np.array([abs(p[0] - xmin), abs(p[0] - xmax), abs(p[1] - ymin), abs(p[1] - ymax)])
    side = int(np.argmin(distances))
    if side == 0:
        p[0] = xmin
    elif side == 1:
        p[0] = xmax
    elif side == 2:
        p[1] = ymin
    else:
        p[1] = ymax
    return p


def _bbox_abscissa(point: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = bbox
    x, y = point
    w = xmax - xmin
    h = ymax - ymin
    if np.isclose(y, ymin):
        return float(x - xmin)
    if np.isclose(x, xmax):
        return float(w + y - ymin)
    if np.isclose(y, ymax):
        return float(w + h + xmax - x)
    return float(2 * w + h + ymax - y)


def _bbox_point(s: float, bbox: tuple[float, float, float, float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    if s <= w:
        return np.array([xmin + s, ymin], dtype=float)
    if s <= w + h:
        return np.array([xmax, ymin + s - w], dtype=float)
    if s <= 2 * w + h:
        return np.array([xmax - (s - w - h), ymax], dtype=float)
    return np.array([xmin, ymax - (s - 2 * w - h)], dtype=float)
