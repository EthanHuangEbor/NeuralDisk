import json
from pathlib import Path

import numpy as np

from hyb_nurbs.ml.dataset import build_dataset


def _write_dummy_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    control_points = [[float(i), float(i + 1)] for i in range(15)]
    (run_dir / "nurbs_fit_results.json").write_text(
        json.dumps(
            [
                {
                    "curves": [
                        {
                            "degree": 3,
                            "control_points": control_points,
                            "weights": [1.0] * 15,
                            "knots": list(np.linspace(0, 1, 19)),
                            "is_closed": True,
                        }
                    ],
                    "metrics": {"n_control_points": 15},
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "fit_metrics.csv").write_text("n_control_points,mean_error_mm\n15,0.1\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "dummy",
                "run_name": run_dir.name,
                "parameters": {"eta": 0.5, "degree": 3},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(
        "\n".join(
            [
                "operating:",
                "  rpm: 12000",
                "topology:",
                "  volume_fraction: 0.5",
                "geometry:",
                "  Ri: 32",
                "  Ro: 168",
                "  B: 46",
                "  h_hub: 8",
                "  h_rim: 10",
                "nurbs:",
                "  degree: 3",
            ]
        ),
        encoding="utf-8",
    )


def test_dataset_builder_generates_xy_from_minimal_run(tmp_path):
    outputs = tmp_path / "outputs"
    _write_dummy_run(outputs / "case_a" / "run_001")
    result = build_dataset(
        outputs_root=outputs,
        out=tmp_path / "data" / "dataset.npz",
        index=tmp_path / "data" / "index.csv",
        required_control_points=15,
        allow_small_data=True,
    )

    assert result.X.shape[0] == 1
    assert result.y.shape == (1, 30)
    assert "is_missing_omega" in result.feature_names
    assert result.dataset_path.exists()
    assert result.index_path.exists()
