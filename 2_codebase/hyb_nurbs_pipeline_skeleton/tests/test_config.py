from pathlib import Path

from hyb_nurbs.config import load_config


def test_case_config_merges_default_and_resolves_relative_paths():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "cases" / "hyb_current_actual.yaml", write_artifacts=False)

    assert cfg["nurbs"]["degree"] == 3
    assert Path(cfg["input"]["node_file"]).is_absolute()
    assert Path(cfg["input"]["density_file"]).is_absolute()
    assert Path(cfg["input"]["node_file"]).name == "NLIST.lis"
    assert Path(cfg["input"]["density_file"]).name == "export1.txt"
    assert cfg["export"]["run_name"] == "run_manual_eta050_tri_iso"
