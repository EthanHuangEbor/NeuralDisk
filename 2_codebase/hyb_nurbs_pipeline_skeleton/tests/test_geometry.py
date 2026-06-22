import numpy as np

from hyb_nurbs.boundary.postprocess import ensure_closed, signed_area
from hyb_nurbs.validation.metrics import polygon_area


def test_polygon_area_square():
    p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert polygon_area(p) == 1.0
    assert signed_area(p) > 0


def test_ensure_closed():
    p = np.array([[0, 0], [1, 0], [1, 1]])
    q = ensure_closed(p, tolerance=1e-9)
    assert np.allclose(q[0], q[-1])
