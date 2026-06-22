from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export_thesis_report(outputs_root: str | Path, report_dir: str | Path) -> dict[str, Path]:
    """Export Chapter 4 metrics, eta sweep table, and thesis-ready figures."""
    outputs_root = Path(outputs_root)
    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = _collect_fit_metrics(outputs_root)
    chapter4_path = tables_dir / "chapter4_fit_metrics.csv"
    eta_path = tables_dir / "eta_sweep.csv"
    metrics.to_csv(chapter4_path, index=False)
    _eta_sweep(metrics).to_csv(eta_path, index=False)

    png_path = figures_dir / "chapter4_fit_metrics.png"
    svg_path = figures_dir / "chapter4_fit_metrics.svg"
    _plot_metrics(metrics, png_path)
    _plot_metrics(metrics, svg_path)
    return {"chapter4_table": chapter4_path, "eta_sweep_table": eta_path, "png": png_path, "svg": svg_path}


def _collect_fit_metrics(outputs_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in outputs_root.rglob("fit_metrics.csv"):
        df = pd.read_csv(path)
        run_dir = path.parent
        eta_match = re.search(r"eta(\d{3})", str(run_dir))
        df.insert(0, "run_dir", str(run_dir))
        df.insert(1, "eta", int(eta_match.group(1)) / 100.0 if eta_match else pd.NA)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No fit_metrics.csv files found below {outputs_root}")
    return pd.concat(rows, ignore_index=True)


def _eta_sweep(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "eta",
        "mean_error_mm",
        "max_error_mm",
        "hausdorff_error_mm",
        "area_error_ratio",
        "n_control_points",
    ]
    available = [c for c in columns if c in metrics.columns]
    return metrics.dropna(subset=["eta"]).groupby("eta", as_index=False)[available[1:]].mean()


def _plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    x = range(len(metrics))
    labels = [Path(v).name for v in metrics["run_dir"]]
    ax1.plot(x, metrics["mean_error_mm"], marker="o", label="Mean error (mm)", color="#1f77b4")
    ax1.plot(x, metrics["max_error_mm"], marker="s", label="Max error (mm)", color="#d62728")
    ax1.set_ylabel("Fit error (mm)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(x, metrics["n_control_points"], alpha=0.22, label="Control points", color="#2ca02c")
    ax2.set_ylabel("Control points")

    lines, names = ax1.get_legend_handles_labels()
    bars, bar_names = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, names + bar_names, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export HYB NURBS thesis tables and figures")
    parser.add_argument("--outputs-root", default="outputs", help="Directory containing pipeline run outputs")
    parser.add_argument("--report-dir", default="reports/thesis", help="Destination report directory")
    args = parser.parse_args(argv)
    export_thesis_report(args.outputs_root, args.report_dir)


if __name__ == "__main__":
    main()
