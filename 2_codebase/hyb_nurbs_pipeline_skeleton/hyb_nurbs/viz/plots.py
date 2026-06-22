from __future__ import annotations

from pathlib import Path

import numpy as np

from hyb_nurbs.schema import BoundaryLoop, FitResult, SectionCloud
from hyb_nurbs.nurbs.evaluate import evaluate_curve


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_density_cloud(cloud: SectionCloud, out_path: str | Path, eta: float) -> None:
    """Scatter plot of projected density field with threshold annotation."""
    plt = _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(cloud.xy[:, 0], cloud.xy[:, 1], c=cloud.rho, s=18, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.scatter(cloud.xy[cloud.rho >= eta, 0], cloud.xy[cloud.rho >= eta, 1], s=8, facecolors="none", edgecolors="white", linewidths=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{cloud.axes[0]} (mm)")
    ax.set_ylabel(f"{cloud.axes[1]} (mm)")
    ax.set_title(f"Projected density cloud, eta={eta:g}")
    fig.colorbar(sc, ax=ax, label="Topology density")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_boundary_overlay(cloud: SectionCloud, loops: list[BoundaryLoop], out_path: str | Path) -> None:
    """Overlay extracted boundary loops on the density cloud."""
    plt = _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(cloud.xy[:, 0], cloud.xy[:, 1], c=cloud.rho, s=14, cmap="viridis", vmin=0.0, vmax=1.0)
    for loop in loops:
        pts = np.asarray(loop.points)
        color = "crimson" if loop.role == "outer" else "orange"
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8, label=loop.role)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{cloud.axes[0]} (mm)")
    ax.set_ylabel(f"{cloud.axes[1]} (mm)")
    ax.set_title("Extracted rho=eta boundary overlay")
    fig.colorbar(sc, ax=ax, label="Topology density")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fit_overlay(results: list[FitResult], out_path: str | Path) -> None:
    """Overlay source boundary points and fitted NURBS samples."""
    plt = _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for result in results:
        pts = np.asarray(result.loop.points)
        ax.plot(pts[:, 0], pts[:, 1], color="0.65", linewidth=1.0)
        for curve in result.curves:
            samples = evaluate_curve(curve, np.linspace(0.0, 1.0, max(300, result.metrics.n_samples), endpoint=False))
            ax.plot(samples[:, 0], samples[:, 1], color="crimson", linewidth=1.8)
            ax.scatter(curve.control_points[:, 0], curve.control_points[:, 1], s=10, color="navy", alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("B-spline fit overlay")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
