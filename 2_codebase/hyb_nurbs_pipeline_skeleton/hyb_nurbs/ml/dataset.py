from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hyb_nurbs.ml.schema import (
    BASE_FEATURES,
    FEATURE_ALIASES,
    DatasetBuildResult,
    target_names_for_control_points,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(_flatten(value, child))
    else:
        out[prefix] = payload
    return out


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _find_numeric_field(payloads: list[dict[str, Any]], aliases: list[str]) -> float | None:
    normalized_aliases = {_norm_key(alias) for alias in aliases}
    flattened: dict[str, Any] = {}
    for payload in payloads:
        flattened.update(_flatten(payload))

    for key, value in flattened.items():
        key_norm = _norm_key(key)
        tail_norm = _norm_key(key.split(".")[-1])
        if key_norm in normalized_aliases or tail_norm in normalized_aliases:
            val = _as_float(value)
            if val is not None:
                return val

    for key, value in flattened.items():
        key_norm = _norm_key(key)
        if any(key_norm.endswith(alias) for alias in normalized_aliases):
            val = _as_float(value)
            if val is not None:
                return val
    return None


def _resolve_outputs_root(outputs_root: str | Path) -> Path:
    root = Path(outputs_root)
    if root.exists():
        return root
    if not root.is_absolute() and root.name == "outputs":
        sibling = Path.cwd().parent / "outputs"
        if sibling.exists():
            return sibling
    return root


def scan_output_runs(outputs_root: str | Path) -> list[Path]:
    root = _resolve_outputs_root(outputs_root)
    if not root.exists():
        raise FileNotFoundError(f"outputs root does not exist: {root}")
    runs = sorted(path.parent for path in root.rglob("nurbs_fit_results.json"))
    return list(dict.fromkeys(runs))


def _extract_control_points(nurbs_payload: Any, required_control_points: int) -> tuple[np.ndarray | None, int | None]:
    results = nurbs_payload if isinstance(nurbs_payload, list) else [nurbs_payload]
    for result in results:
        if not isinstance(result, dict):
            continue
        curves = result.get("curves") or []
        if not curves and "control_points" in result:
            curves = [result]

        combined: list[np.ndarray] = []
        for curve in curves:
            if not isinstance(curve, dict) or "control_points" not in curve:
                continue
            cp = np.asarray(curve["control_points"], dtype=float)
            if cp.ndim != 2 or cp.shape[1] != 2:
                continue
            if cp.shape[0] == required_control_points:
                return cp, int(curve.get("degree", result.get("degree", 0)) or 0)
            combined.append(cp)
        if combined:
            stacked = np.vstack(combined)
            if stacked.shape[0] == required_control_points:
                degree = 0
                if curves and isinstance(curves[0], dict):
                    degree = int(curves[0].get("degree", result.get("degree", 0)) or 0)
                return stacked, degree
    return None, None


def _sample_id_from(run_dir: Path, manifest: dict[str, Any]) -> str:
    case_id = manifest.get("case_id") or manifest.get("case", {}).get("case_id")
    run_name = manifest.get("run_name")
    if case_id and run_name:
        return f"{case_id}/{run_name}"
    if run_name:
        return str(run_name)
    return run_dir.name


def extract_feature_values(
    payloads: list[dict[str, Any]],
    *,
    extra_values: dict[str, Any] | None = None,
) -> tuple[list[float], list[float], list[str]]:
    extra_values = extra_values or {}
    values: list[float] = []
    masks: list[float] = []
    missing: list[str] = []
    augmented_payloads = payloads + [extra_values]
    for feature in BASE_FEATURES:
        val = _find_numeric_field(augmented_payloads, FEATURE_ALIASES.get(feature, [feature]))
        if val is None:
            values.append(0.0)
            masks.append(1.0)
            missing.append(feature)
        else:
            values.append(float(val))
            masks.append(0.0)
    return values, masks, missing


def _make_splits(n_samples: int, split_ratio: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_samples, dtype=int)
    if n_samples <= 0:
        return indices, indices, indices
    if n_samples < 3:
        return indices.copy(), indices.copy(), indices.copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_ratio, val_ratio, _ = split_ratio
    n_train = max(1, int(round(n_samples * train_ratio)))
    n_val = max(1, int(round(n_samples * val_ratio)))
    if n_train + n_val >= n_samples:
        n_train = max(1, n_samples - 2)
        n_val = 1
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])
    if test_idx.size == 0:
        test_idx = val_idx.copy()
    return train_idx, val_idx, test_idx


def _stats(array: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = array[idx] if idx.size else array
    mean = subset.mean(axis=0) if subset.size else np.zeros(array.shape[1], dtype=float)
    std = subset.std(axis=0) if subset.size else np.ones(array.shape[1], dtype=float)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def build_dataset(
    *,
    outputs_root: str | Path,
    out: str | Path,
    index: str | Path,
    required_control_points: int = 15,
    allow_small_data: bool = False,
    split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> DatasetBuildResult:
    run_dirs = scan_output_runs(outputs_root)
    rows: list[dict[str, Any]] = []
    X_rows: list[list[float]] = []
    y_rows: list[np.ndarray] = []
    sample_ids: list[str] = []

    for run_dir in run_dirs:
        nurbs_path = run_dir / "nurbs_fit_results.json"
        manifest_path = run_dir / "run_manifest.json"
        config_path = run_dir / "config_resolved.yaml"
        metrics_path = run_dir / "fit_metrics.csv"

        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        config = _read_yaml(config_path)
        metrics_payload: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics_payload = pd.read_csv(metrics_path).iloc[0].to_dict()
            except Exception:
                metrics_payload = {}

        nurbs_payload = _read_json(nurbs_path)
        control_points, degree = _extract_control_points(nurbs_payload, required_control_points)
        sample_id = _sample_id_from(run_dir, manifest)
        if control_points is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "run_dir": str(run_dir),
                    "used": False,
                    "skipped_reason": f"required_control_points={required_control_points} not found",
                    "missing_fields": "",
                }
            )
            continue

        extra_values = {
            "n_control_points": int(control_points.shape[0]),
            "degree": degree,
        }
        values, masks, missing = extract_feature_values(
            [config, manifest, metrics_payload],
            extra_values=extra_values,
        )
        feature_vector = values + masks
        target_vector = control_points.reshape(-1)
        row = {
            "sample_id": sample_id,
            "run_dir": str(run_dir),
            "used": True,
            "skipped_reason": "",
            "missing_fields": ";".join(missing),
        }
        for name, value in zip(BASE_FEATURES, values, strict=True):
            row[name] = value
        for name, value in zip([f"is_missing_{name}" for name in BASE_FEATURES], masks, strict=True):
            row[name] = value
        rows.append(row)
        X_rows.append(feature_vector)
        y_rows.append(target_vector)
        sample_ids.append(sample_id)

    index_path = Path(index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, index=False)

    if not X_rows:
        raise ValueError(f"No usable NURBS samples found under {outputs_root}")
    if len(X_rows) < 10 and not allow_small_data:
        raise ValueError(
            f"Only {len(X_rows)} usable sample(s) found. Add --allow-small-data for smoke/demo dataset builds."
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    feature_names = BASE_FEATURES + [f"is_missing_{name}" for name in BASE_FEATURES]
    target_names = target_names_for_control_points(required_control_points)
    train_idx, val_idx, test_idx = _make_splits(len(X_rows), split_ratio, seed)
    x_mean, x_std = _stats(X, train_idx)
    y_mean, y_std = _stats(y, train_idx)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names, dtype=object),
        target_names=np.asarray(target_names, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        required_control_points=np.asarray([required_control_points], dtype=int),
        split_ratio=np.asarray(split_ratio, dtype=float),
        is_small_data=np.asarray([len(X_rows) < 10], dtype=bool),
    )
    return DatasetBuildResult(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        sample_ids=sample_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        index_path=index_path,
        dataset_path=out_path,
    )


def _parse_split_ratio(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split ratio must be three comma-separated numbers")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("split ratio must have positive sum")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build ML datasets from HYB NURBS run outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Build an NPZ regression dataset.")
    build_parser.add_argument("--outputs-root", default="outputs")
    build_parser.add_argument("--out", default="data/processed/ml/hyb_nurbs_dataset.npz")
    build_parser.add_argument("--index", default="data/processed/ml/ml_index.csv")
    build_parser.add_argument("--required-control-points", type=int, default=15)
    build_parser.add_argument("--allow-small-data", action="store_true")
    build_parser.add_argument("--split-ratio", type=_parse_split_ratio, default=(0.70, 0.15, 0.15))
    build_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_dataset(
            outputs_root=args.outputs_root,
            out=args.out,
            index=args.index,
            required_control_points=args.required_control_points,
            allow_small_data=args.allow_small_data,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
        print(
            f"built dataset: samples={result.X.shape[0]} features={result.X.shape[1]} "
            f"targets={result.y.shape[1]} out={result.dataset_path}"
        )
        if result.X.shape[0] < 10:
            print("warning: small dataset built for smoke/demo validation only; not formal generalization training.")


if __name__ == "__main__":
    main()
