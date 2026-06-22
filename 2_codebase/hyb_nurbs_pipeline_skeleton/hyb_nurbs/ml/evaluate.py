from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_predictions(predictions_csv: str | Path, out: str | Path | None = None) -> dict[str, float]:
    frame = pd.read_csv(predictions_csv)
    target_cols = [col for col in frame.columns if col.startswith("target_")]
    pred_cols = [col for col in frame.columns if col.startswith("pred_")]
    if len(target_cols) != len(pred_cols) or not target_cols:
        raise ValueError("predictions CSV must contain matching target_XX and pred_XX columns")
    target = frame[target_cols].to_numpy(dtype=float)
    pred = frame[pred_cols].to_numpy(dtype=float)
    error = pred - target
    metrics = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
    }
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved HYB NURBS ML predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    metrics = evaluate_predictions(args.predictions, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
