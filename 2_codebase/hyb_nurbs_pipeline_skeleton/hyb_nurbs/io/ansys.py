from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from hyb_nurbs.schema import NodeDensityTable

_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?"
_NODE_LINE_RE = re.compile(rf"^\s*(\d+)\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s*$")
_DENSITY_LINE_RE = re.compile(rf"^\s*(\d+)\s+({_FLOAT_RE})\s*$")
_INT_LINE_RE = re.compile(r"^\s*(?:\d+\s+){2,}\d+\s*$")


def parse_nlist(path: str | Path) -> pd.DataFrame:
    """Parse ANSYS NLIST.lis into columns: node_id, x, y, z.

    Must ignore repeated ANSYS headers such as `NODE X Y Z`.
    """
    rows: list[tuple[int, float, float, float]] = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = _NODE_LINE_RE.match(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    df = pd.DataFrame(rows, columns=["node_id", "x", "y", "z"])
    if df.empty:
        raise ValueError(f"No node rows parsed from {path}")
    if df["node_id"].duplicated().any():
        raise ValueError(f"Duplicate node ids in {path}")
    return df.sort_values("node_id").reset_index(drop=True)


def parse_density(path: str | Path) -> pd.DataFrame:
    """Parse export1.txt into columns: node_id, rho."""
    rows: list[tuple[int, float]] = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = _DENSITY_LINE_RE.match(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    df = pd.DataFrame(rows, columns=["node_id", "rho"])
    if df.empty:
        raise ValueError(f"No density rows parsed from {path}")
    if df["node_id"].duplicated().any():
        raise ValueError(f"Duplicate node ids in {path}")
    return df.sort_values("node_id").reset_index(drop=True)


def parse_element_connectivity(path: str | Path) -> list[list[int]]:
    """Parse a simple ANSYS element connectivity listing.

    The parser is intentionally permissive: it keeps integer-only rows and treats
    the first integer as the element id and the remaining integers as node ids.
    For richer ELIST formats, export a clean element-id/node-id table before use.
    """
    elements: list[list[int]] = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not _INT_LINE_RE.match(line):
            continue
        ints = [int(v) for v in line.split()]
        if len(ints) >= 4:
            nodes = list(dict.fromkeys(ints[1:]))
            if len(nodes) >= 3:
                elements.append(nodes)
    if not elements:
        raise ValueError(f"No element connectivity rows parsed from {path}")
    return elements


def load_node_density(
    node_file: str | Path,
    density_file: str | Path,
    *,
    scale_to_mm: bool = True,
) -> NodeDensityTable:
    """Load, join, validate, and optionally scale coordinates from meters to millimeters."""
    nodes = parse_nlist(node_file)
    dens = parse_density(density_file)
    merged = nodes.merge(dens, on="node_id", how="inner", validate="one_to_one")
    if len(merged) != len(nodes) or len(merged) != len(dens):
        missing_dens = set(nodes.node_id) - set(dens.node_id)
        missing_nodes = set(dens.node_id) - set(nodes.node_id)
        raise ValueError(f"Node-density mismatch: missing_density={missing_dens}, missing_nodes={missing_nodes}")
    xyz = merged[["x", "y", "z"]].to_numpy(dtype=float)
    if scale_to_mm:
        xyz = xyz * 1000.0
    table = NodeDensityTable(
        node_id=merged["node_id"].to_numpy(dtype=int),
        xyz=xyz,
        rho=merged["rho"].to_numpy(dtype=float),
        source_node_file=Path(node_file),
        source_density_file=Path(density_file),
    )
    table.validate()
    return table
