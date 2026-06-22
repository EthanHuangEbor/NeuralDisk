from __future__ import annotations

import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_PATH_KEYS = {
    ("input", "node_file"),
    ("input", "density_file"),
    ("input", "element_file"),
    ("export", "out_root"),
    ("export", "out_dir"),
}


def load_config(path: str | Path, *, write_artifacts: bool = True) -> dict[str, Any]:
    """Load, merge, resolve, and optionally persist a pipeline config.

    Relative paths in the case config are resolved against the directory that
    contains that config file. Case YAML values override ``configs/default.yaml``.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    default_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    default_cfg = _read_yaml(default_path) if default_path.exists() else {}
    case_cfg = _read_yaml(config_path)
    cfg = _deep_merge(default_cfg, case_cfg)
    cfg["_config_meta"] = {
        "config_path": str(config_path),
        "default_config_path": str(default_path) if default_path.exists() else None,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }

    _derive_run_dir(cfg, config_path.parent)
    _resolve_paths(cfg, config_path.parent)

    if write_artifacts:
        write_config_artifacts(cfg)
    return cfg


def write_config_artifacts(config: dict[str, Any]) -> None:
    """Write config_resolved.yaml and an initial run_manifest.json."""
    out_dir = Path(config["export"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = _jsonable(config)
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = build_run_manifest(config, status="configured")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_run_manifest(
    config: dict[str, Any],
    *,
    status: str,
    counts: dict[str, Any] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the run manifest payload used before and after pipeline execution."""
    case = config.get("case", {})
    export = config.get("export", {})
    boundary = config.get("boundary", {})
    nurbs = config.get("nurbs", {})
    projection = config.get("projection", {})
    return {
        "case_id": case.get("case_id", "default"),
        "run_name": export.get("run_name", Path(export.get("out_dir", "run")).name),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "node_file": config.get("input", {}).get("node_file"),
            "density_file": config.get("input", {}).get("density_file"),
            "element_file": config.get("input", {}).get("element_file"),
        },
        "counts": counts or {},
        "parameters": {
            "eta": config.get("threshold", {}).get("eta"),
            "projection": {
                "axes": projection.get("axes"),
                "mode": projection.get("mode"),
                "aggregate_density": projection.get("aggregate_density"),
            },
            "boundary_method": boundary.get("method"),
            "degree": nurbs.get("degree"),
            "closed_loop_mode": nurbs.get("closed_loop_mode"),
            "weights_mode": nurbs.get("weights_mode"),
        },
        "metrics": metrics or [],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": config.get("_config_meta", {}),
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _derive_run_dir(config: dict[str, Any], base_dir: Path) -> None:
    export = config.setdefault("export", {})
    case_id = config.get("case", {}).get("case_id", "default")
    eta = float(config.get("threshold", {}).get("eta", 0.5))
    method = config.get("boundary", {}).get("method", "tri_iso")
    run_name = export.get("run_name")
    if not run_name or run_name == "auto":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{stamp}_eta{int(round(eta * 100)):03d}_{method}"
        export["run_name"] = run_name
    if not export.get("out_dir"):
        out_root = Path(export.get("out_root", "outputs"))
        if not out_root.is_absolute():
            out_root = (base_dir / out_root).resolve()
        export["out_dir"] = str(out_root / case_id / run_name)


def _resolve_paths(config: dict[str, Any], base_dir: Path) -> None:
    for section, key in _PATH_KEYS:
        if section not in config or key not in config[section] or config[section][key] in (None, ""):
            continue
        raw = Path(str(config[section][key])).expanduser()
        config[section][key] = str(raw if raw.is_absolute() else (base_dir / raw).resolve())


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    return obj
