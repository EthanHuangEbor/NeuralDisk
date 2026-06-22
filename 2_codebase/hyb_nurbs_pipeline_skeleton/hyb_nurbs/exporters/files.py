from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hyb_nurbs.schema import BoundaryLoop, FitResult, NodeDensityTable, NurbsCurveSpec, SectionCloud
from hyb_nurbs.nurbs.evaluate import evaluate_curve


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_debug_csv(out_dir: str | Path, table: NodeDensityTable, cloud: SectionCloud) -> None:
    """Write merged node-density and projected-section CSV files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = pd.DataFrame(
        {
            "node_id": table.node_id,
            "x_mm": table.xyz[:, 0],
            "y_mm": table.xyz[:, 1],
            "z_mm": table.xyz[:, 2],
            "rho": table.rho,
        }
    )
    merged.to_csv(out_dir / "merged_node_density.csv", index=False)

    section = pd.DataFrame(
        {
            "point_id": cloud.point_id,
            f"{cloud.axes[0]}_mm": cloud.xy[:, 0],
            f"{cloud.axes[1]}_mm": cloud.xy[:, 1],
            "rho": cloud.rho,
            "source_node_ids": [";".join(map(str, ids)) for ids in cloud.source_node_ids],
        }
    )
    section.to_csv(out_dir / "section_cloud.csv", index=False)


def write_apdl_polyline_macro(path: str | Path, curves: list[NurbsCurveSpec]) -> None:
    """Write an APDL macro template using sampled keypoints/lines.

    The sampled polyline is the geometry fallback. The rest of the macro is a
    ready-to-edit area/revolve/mesh/load/solve/postprocess scaffold.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "! HYB NURBS sampled section plus APDL analysis template",
        "! Coordinates are in mm in the section plane.",
        "! STL exports from this pipeline are visualization previews only.",
        "! Prefer STEP/IGES/NURBS CAD exchange for production geometry.",
        "/CLEAR",
        "/FILNAME,hyb_nurbs_disk",
        "/PREP7",
        "ET,1,PLANE183",
        "KEYOPT,1,3,1",
        "MPTEMP,1,0",
        "MPDATA,EX,1,,2.06E5",
        "MPDATA,PRXY,1,,0.30",
    ]
    kp = 1
    first_kp = kp
    prev_kp: int | None = None
    for curve in curves:
        samples = evaluate_curve(curve, np.linspace(0.0, 1.0, 100, endpoint=False))
        for x, z in samples:
            lines.append(f"K,{kp},{x:.9g},0,{z:.9g}")
            if prev_kp is not None:
                lines.append(f"L,{prev_kp},{kp}")
            prev_kp = kp
            kp += 1
    if prev_kp is not None and prev_kp != first_kp:
        lines.append(f"L,{prev_kp},{first_kp}")
    lines.extend(
        [
            "AL,ALL",
            "! Optional axisymmetric area solve:",
            "MSHKEY,1",
            "ESIZE,1.0",
            "AMESH,ALL",
            "D,ALL,UY,0",
            "! Apply realistic constraints and loads for the thesis case here.",
            "! SFL,ALL,PRES,<pressure>",
            "/SOLU",
            "ANTYPE,STATIC",
            "SOLVE",
            "FINISH",
            "/POST1",
            "SET,LAST",
            "PLNSOL,S,EQV",
            "FINISH",
            "! Optional 3D revolve route:",
            "! /PREP7",
            "! K,900001,0,0,-1",
            "! K,900002,0,0,1",
            "! VROTAT,ALL,,,,,,900001,900002,360,96",
            "! ET,2,SOLID186",
            "! VMESH,ALL",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cad_exchange_stub(path: str | Path, curves: list[NurbsCurveSpec], *, format_name: str) -> None:
    """Write a CAD exchange interface file or a documented fallback stub.

    Full STEP/IGES NURBS entities require a CAD kernel such as pythonocc-core.
    This function keeps the export contract stable and records sampled curve
    points so downstream CAD implementation has deterministic input.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format_name.upper()
    samples = [
        {
            "degree": curve.degree,
            "fit_kind": curve.fit_kind,
            "role": curve.role,
            "segment_id": curve.segment_id,
            "points": evaluate_curve(curve, np.linspace(0.0, 1.0, 64, endpoint=not curve.is_closed)).tolist(),
        }
        for curve in curves
    ]
    if fmt == "STEP":
        text = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('HYB NURBS CAD exchange interface; sampled fallback only'),'2;1');",
            f"FILE_NAME('{path.name}','',('hyb_nurbs'),('hyb_nurbs'),'hyb_nurbs','hyb_nurbs','');",
            "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));",
            "ENDSEC;",
            "DATA;",
            f"/* {json.dumps(_to_jsonable(samples), ensure_ascii=False)} */",
            "ENDSEC;",
            "END-ISO-10303-21;",
        ]
    else:
        text = [
            "                                                                        S      1",
            "1H,,1H;,6HHYB_NU,10HNURBS_STUB,32,38,6,308,15,1.0,1,4HMM,1,0.01,15H20260508.000000;G      1",
            f"{json.dumps(_to_jsonable(samples), ensure_ascii=False)}",
            "S      1G      1D      0P      0                                        T      1",
        ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_ascii_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray, solid_name: str = "hyb_revolved") -> None:
    """Write a lightweight ASCII STL mesh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    lines = [f"solid {solid_name}"]
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        lines.append(f"  facet normal {normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}")
        lines.append("    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
