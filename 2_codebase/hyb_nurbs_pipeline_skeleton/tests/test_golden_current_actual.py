from pathlib import Path

from hyb_nurbs.config import load_config
from hyb_nurbs.pipeline import run_pipeline


def test_current_real_data_golden_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "cases" / "hyb_current_actual.yaml", write_artifacts=False)
    assert Path(cfg["input"]["node_file"]).exists()
    assert Path(cfg["input"]["density_file"]).exists()

    cfg["export"]["out_dir"] = str(tmp_path / "golden_run")
    cfg["export"]["write_debug_csv"] = False
    cfg["export"]["write_plots"] = False
    cfg["export"]["write_apdl_macro"] = False
    cfg["export"]["write_cad"] = False

    result = run_pipeline(cfg)
    fit = result["fit_results"][0]

    assert result["table"].node_id.size == 1050
    assert result["cloud"].xy.shape[0] == 673
    assert len(result["loops"]) == 1
    assert fit.loop.role == "outer"
    assert fit.metrics.mean_error_mm < 0.36
    assert fit.metrics.max_error_mm < 1.50
    assert fit.metrics.area_error_ratio < 0.012
    assert fit.metrics.n_control_points <= 20
    assert (tmp_path / "golden_run" / "run_manifest.json").exists()
