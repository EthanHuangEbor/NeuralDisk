from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hyb_nurbs.ml.losses import combined_loss
from hyb_nurbs.ml.model import MLPRegressor, require_torch


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _strings(array: np.ndarray) -> list[str]:
    return [str(item) for item in array.tolist()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalization(values: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = values[idx] if idx.size else values
    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _write_loss_curve(out_path: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _eval_loss(model, Xn, yn, indices, lambda_mse: float, lambda_smooth: float):  # type: ignore[no-untyped-def]
    torch = require_torch()
    if indices.size == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = model(Xn[indices])
        loss = combined_loss(pred, yn[indices], lambda_p=lambda_mse, lambda_s=lambda_smooth)
    return float(loss.detach().cpu().item())


def train_model(
    *,
    dataset: str | Path,
    config: str | Path | None,
    out: str | Path,
    smoke_test: bool = False,
    allow_small_data: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    cfg = _load_config(config)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    payload = np.load(dataset, allow_pickle=True)
    X = payload["X"].astype(np.float32)
    y = payload["y"].astype(np.float32)
    feature_names = _strings(payload["feature_names"])
    target_names = _strings(payload["target_names"])
    sample_ids = _strings(payload["sample_ids"])
    train_idx = payload["train_idx"].astype(int)
    val_idx = payload["val_idx"].astype(int)
    test_idx = payload["test_idx"].astype(int)
    n_samples = int(X.shape[0])
    if n_samples < 10 and not (smoke_test or allow_small_data):
        raise ValueError("Refusing formal training with fewer than 10 samples. Use --smoke-test or --allow-small-data.")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_mean, x_std = _normalization(X, train_idx)
    y_mean, y_std = _normalization(y, train_idx)
    Xn_np = (X - x_mean) / x_std
    yn_np = (y - y_mean) / y_std
    Xn = torch.tensor(Xn_np, dtype=torch.float32)
    yn = torch.tensor(yn_np, dtype=torch.float32)

    hidden_dims = [int(v) for v in cfg.get("hidden_dims", [64, 128, 128, 64])]
    activation = str(cfg.get("activation", "relu"))
    dropout = float(cfg.get("dropout", 0.0))
    model = MLPRegressor(X.shape[1], y.shape[1], hidden_dims=hidden_dims, activation=activation, dropout=dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    batch_size = max(1, int(cfg.get("batch_size", 32)))
    epochs = int(cfg.get("epochs", 300))
    if smoke_test:
        epochs = min(epochs, 3)
    patience = int(cfg.get("patience", 30))
    lambda_mse = float(cfg.get("lambda_mse", 1.0))
    lambda_smooth = float(cfg.get("lambda_smooth", 0.0))

    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    train_idx_for_batches = train_idx if train_idx.size else np.arange(n_samples)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_idx_for_batches.copy()
        rng.shuffle(shuffled)
        batch_losses: list[float] = []
        for start in range(0, shuffled.size, batch_size):
            batch_idx = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(Xn[batch_idx])
            loss = combined_loss(pred, yn[batch_idx], lambda_p=lambda_mse, lambda_s=lambda_smooth)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_indices = val_idx if val_idx.size else train_idx_for_batches
        val_loss = _eval_loss(model, Xn, yn, val_indices, lambda_mse, lambda_smooth)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_norm = model(Xn).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    errors = pred - y
    eval_idx = test_idx if test_idx.size else np.arange(n_samples)
    test_errors = errors[eval_idx]
    test_mse = float(np.mean(test_errors**2))
    test_mae = float(np.mean(np.abs(test_errors)))
    test_rmse = float(np.sqrt(test_mse))
    max_abs_error = float(np.max(np.abs(test_errors)))

    warning = ""
    if smoke_test or n_samples < 10:
        warning = "Smoke test only: this run validates the training code path and does not represent engineering generalization accuracy."

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "output_dim": int(y.shape[1]),
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
            "is_smoke_test": bool(smoke_test or n_samples < 10),
            "target_format": f"{y.shape[1] // 2}_control_points_2d",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        out_dir / "model.pt",
    )

    normalizer = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "feature_names": feature_names,
        "target_names": target_names,
        "is_smoke_test": bool(smoke_test or n_samples < 10),
        "warning": warning,
    }
    (out_dir / "normalizer.json").write_text(json.dumps(normalizer, indent=2), encoding="utf-8")

    with (out_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    split_by_idx = {int(i): "train" for i in train_idx}
    split_by_idx.update({int(i): "val" for i in val_idx})
    split_by_idx.update({int(i): "test" for i in test_idx})
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "split"] + [f"target_{i:02d}" for i in range(y.shape[1])] + [
            f"pred_{i:02d}" for i in range(y.shape[1])
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample_id in enumerate(sample_ids):
            row = {"sample_id": sample_id, "split": split_by_idx.get(idx, "unused")}
            row.update({f"target_{i:02d}": float(y[idx, i]) for i in range(y.shape[1])})
            row.update({f"pred_{i:02d}": float(pred[idx, i]) for i in range(y.shape[1])})
            writer.writerow(row)

    _write_loss_curve(out_dir / "loss_curve.png", history)
    metrics = {
        "n_samples": n_samples,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "feature_names": feature_names,
        "target_dim": int(y.shape[1]),
        "train_loss": float(history[-1]["train_loss"]) if history else None,
        "val_loss": float(history[-1]["val_loss"]) if history else None,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "max_abs_error": max_abs_error,
        "is_smoke_test": bool(smoke_test or n_samples < 10),
        "warning": warning,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(dataset),
                "config": str(config) if config else None,
                "out_dir": str(out_dir),
                "is_smoke_test": bool(smoke_test or n_samples < 10),
                "warning": warning,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# HYB NURBS MLP run\n\n"
        "This directory contains a control-point surrogate training run.\n\n"
        f"- samples: {n_samples}\n"
        f"- target dimension: {y.shape[1]}\n"
        f"- smoke test: {bool(smoke_test or n_samples < 10)}\n\n"
        f"{warning}\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an MLP surrogate for HYB NURBS control points.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-small-data", action="store_true")
    args = parser.parse_args(argv)
    metrics = train_model(
        dataset=args.dataset,
        config=args.config,
        out=args.out,
        smoke_test=args.smoke_test,
        allow_small_data=args.allow_small_data,
    )
    print(
        f"trained MLP: samples={metrics['n_samples']} target_dim={metrics['target_dim']} "
        f"smoke={metrics['is_smoke_test']} out={args.out}"
    )


if __name__ == "__main__":
    main()
