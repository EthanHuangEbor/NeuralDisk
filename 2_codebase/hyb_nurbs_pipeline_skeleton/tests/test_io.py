from pathlib import Path

from hyb_nurbs.io.ansys import load_node_density, parse_density, parse_nlist


def test_parse_current_files_when_available():
    root = Path(__file__).resolve().parents[1]
    candidates = [
        (
            root / "data" / "raw" / "hyb_current_actual" / "NLIST.lis",
            root / "data" / "raw" / "hyb_current_actual" / "export1.txt",
        ),
        (root / "NLIST.lis", root / "export1.txt"),
        (root.parent / "NLIST.lis", root.parent / "export1.txt"),
    ]
    node_file, density_file = next(
        ((node, density) for node, density in candidates if node.exists() and density.exists()),
        candidates[0],
    )
    assert node_file.exists(), f"Current real node file is required for golden tests: {node_file}"
    assert density_file.exists(), f"Current real density file is required for golden tests: {density_file}"
    nodes = parse_nlist(node_file)
    dens = parse_density(density_file)
    assert len(nodes) == 1050
    assert len(dens) == 1050
    table = load_node_density(node_file, density_file, scale_to_mm=True)
    assert table.node_id.size == 1050
    assert table.node_id[0] == 1
    assert table.node_id[-1] == 1050
