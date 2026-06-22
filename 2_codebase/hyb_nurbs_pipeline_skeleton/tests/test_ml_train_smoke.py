import json

import numpy as np
import yaml

from hyb_nurbs.ml.train import train_model


def test_smoke_training_writes_expected_artifacts(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    y = np.asarray([[0.0] * 30, [1.0] * 30], dtype=float)
    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        feature_names=np.asarray(["rpm", "is_missing_rpm"], dtype=object),
        target_names=np.asarray([f"target_{i}" for i in range(30)], dtype=object),
        sample_ids=np.asarray(["a", "b"], dtype=object),
        train_idx=np.asarray([0, 1], dtype=int),
        val_idx=np.asarray([0, 1], dtype=int),
        test_idx=np.asarray([0, 1], dtype=int),
    )
    cfg_path = tmp_path / "mlp.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 1,
                "hidden_dims": [8],
                "epochs": 2,
                "batch_size": 2,
                "patience": 2,
                "lambda_mse": 1.0,
                "lambda_smooth": 0.0,
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "run"
    metrics = train_model(dataset=dataset_path, config=cfg_path, out=out_dir, smoke_test=True)

    assert metrics["is_smoke_test"] is True
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "normalizer.json").exists()
    assert (out_dir / "metrics.json").exists()
    saved_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["target_dim"] == 30
