from __future__ import annotations

from hyb_nurbs.ml.model import require_torch


def mse_loss(pred, target):  # type: ignore[no-untyped-def]
    torch = require_torch()
    return torch.mean((pred - target) ** 2)


def mae(pred, target):  # type: ignore[no-untyped-def]
    torch = require_torch()
    return torch.mean(torch.abs(pred - target))


def rmse(pred, target):  # type: ignore[no-untyped-def]
    torch = require_torch()
    return torch.sqrt(mse_loss(pred, target))


def smoothness_loss(pred):  # type: ignore[no-untyped-def]
    """Second-difference penalty over consecutive 2D control points."""
    torch = require_torch()
    if pred.shape[-1] < 6 or pred.shape[-1] % 2 != 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    points = pred.reshape(pred.shape[0], pred.shape[-1] // 2, 2)
    if points.shape[1] < 3:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    second_diff = points[:, 2:, :] - 2.0 * points[:, 1:-1, :] + points[:, :-2, :]
    return torch.mean(second_diff**2)


def combined_loss(pred, target, *, lambda_p: float = 1.0, lambda_s: float = 0.0):  # type: ignore[no-untyped-def]
    return float(lambda_p) * mse_loss(pred, target) + float(lambda_s) * smoothness_loss(pred)


def geometry_distance_loss(pred, target):  # type: ignore[no-untyped-def]
    """Reserved interface for curve-level distance losses.

    A robust implementation should reuse the NURBS evaluator and compare sampled
    curves, but the current single-case pipeline should not report synthetic
    geometry-loss values as formal engineering accuracy.
    """
    raise NotImplementedError("geometry_distance_loss is reserved for future NURBS curve-level training.")
