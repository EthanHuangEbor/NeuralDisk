from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hyb_nurbs.ml.dataset import _find_numeric_field
from hyb_nurbs.ml.model import MLPRegressor, require_torch
from hyb_nurbs.ml.schema import BASE_FEATURES, FEATURE_ALIASES


def _read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"input": data}


def _feature_vector(payload: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    values: dict[str, float] = {}
    masks: dict[str, float] = {}
    for feature in BASE_FEATURES:
        val = _find_numeric_field([payload], FEATURE_ALIASES.get(feature, [feature]))
        if val is None:
            values[feature] = 0.0
            masks[f"is_missing_{feature}"] = 1.0
        else:
            values[feature] = float(val)
            masks[f"is_missing_{feature}"] = 0.0
    merged = {**values, **masks}
    return np.asarray([merged.get(name, 0.0) for name in feature_names], dtype=np.float32)


def predict_control_points(
    *,
    model_path: str | Path,
    normalizer_path: str | Path,
    input_path: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    torch = require_torch()
    normalizer = _read_json(normalizer_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    model = MLPRegressor(
        int(checkpoint["input_dim"]),
        int(checkpoint["output_dim"]),
        hidden_dims=list(checkpoint.get("hidden_dims", [64, 128, 128, 64])),
        activation=str(checkpoint.get("activation", "relu")),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    payload = _read_json(input_path)
    feature_names = [str(name) for name in normalizer["feature_names"]]
    x = _feature_vector(payload, feature_names)
    x_mean = np.asarray(normalizer["x_mean"], dtype=np.float32)
    x_std = np.asarray(normalizer["x_std"], dtype=np.float32)
    y_mean = np.asarray(normalizer["y_mean"], dtype=np.float32)
    y_std = np.asarray(normalizer["y_std"], dtype=np.float32)
    xn = (x - x_mean) / x_std
    with torch.no_grad():
        pred_norm = model(torch.tensor(xn[None, :], dtype=torch.float32)).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
    control_points = pred.reshape(-1, 2).tolist()
    is_smoke = bool(checkpoint.get("is_smoke_test") or normalizer.get("is_smoke_test"))
    warning = ""
    if is_smoke:
        warning = "Model was trained in smoke-test/small-data mode and must not be used as a formal design result."
    output = {
        "input_features": {name: float(value) for name, value in zip(feature_names, x.tolist(), strict=True)},
        "predicted_control_points": control_points,
        "target_format": checkpoint.get("target_format", f"{len(control_points)}_control_points_2d"),
        "model_path": str(model_path),
        "normalizer_path": str(normalizer_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": warning,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict NURBS control points from an ML surrogate.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = predict_control_points(model_path=args.model, normalizer_path=args.normalizer, input_path=args.input, out=args.out)
    print(f"wrote prediction: control_points={len(result['predicted_control_points'])} out={args.out}")


if __name__ == "__main__":
    main()
