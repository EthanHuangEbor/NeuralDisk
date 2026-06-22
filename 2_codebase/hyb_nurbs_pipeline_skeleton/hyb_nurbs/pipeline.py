from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hyb_nurbs.boundary.iso import (
    extract_alpha_shape_boundary,
    extract_iso_contours_grid,
    extract_iso_contours_mesh,
    extract_iso_contours_tri,
)
from hyb_nurbs.boundary.postprocess import classify_and_filter_loops, resample_loop_by_arclength
from hyb_nurbs.cad.revolve import revolve_curves_to_surface
from hyb_nurbs.config import build_run_manifest
from hyb_nurbs.exporters.files import (
    write_apdl_polyline_macro,
    write_ascii_stl,
    write_cad_exchange_stub,
    write_debug_csv,
    write_json,
)
from hyb_nurbs.io.ansys import load_node_density, parse_element_connectivity
from hyb_nurbs.nurbs.fitting import refine_fit_until_tolerance
from hyb_nurbs.preprocess.projection import project_to_section
from hyb_nurbs.viz.plots import plot_boundary_overlay, plot_density_cloud, plot_fit_overlay


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Execute full node-density -> boundary -> NURBS reconstruction pipeline."""
    out_dir = Path(config["export"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_node_density(
        config["input"]["node_file"],
        config["input"]["density_file"],
        scale_to_mm=config["units"].get("auto_scale_to_mm", True),
    )

    cloud = project_to_section(
        table,
        axes=tuple(config["projection"].get("axes", ["x", "z"])),
        thickness_axis=config["projection"].get("thickness_axis", "y"),
        mode=config["projection"].get("mode", "aggregate"),
        slice_y=config["projection"].get("slice_y"),
        aggregate_density=config["projection"].get("aggregate_density", "max"),
    )

    eta = float(config["threshold"]["eta"])
    method = config["boundary"].get("method", "tri_iso")
    element_file = config.get("input", {}).get("element_file")
    if method == "mesh_iso" or (method == "auto" and element_file):
        if not element_file:
            raise ValueError("boundary.method='mesh_iso' requires input.element_file")
        loops = extract_iso_contours_mesh(cloud, parse_element_connectivity(element_file), eta=eta)
    elif method == "tri_iso" or method == "auto":
        loops = extract_iso_contours_tri(cloud, eta=eta)
    elif method == "grid_iso":
        loops = extract_iso_contours_grid(cloud, eta=eta, grid_resolution_mm=config["boundary"]["grid_resolution_mm"])
    elif method == "alpha_shape":
        loops = extract_alpha_shape_boundary(cloud, eta=eta)
    else:
        raise ValueError(f"Unknown boundary method: {method}")

    loops = classify_and_filter_loops(
        loops,
        min_component_area_mm2=float(config["threshold"]["min_component_area_mm2"]),
        min_hole_area_mm2=float(config["threshold"]["min_hole_area_mm2"]),
    )
    loops = [resample_loop_by_arclength(loop, config["boundary"]["resample_spacing_mm"]) for loop in loops]

    nurbs_cfg = config["nurbs"]
    results = [
        refine_fit_until_tolerance(
            loop,
            degree=int(nurbs_cfg["degree"]),
            min_ctrlpts=int(nurbs_cfg["min_ctrlpts"]),
            max_ctrlpts=int(nurbs_cfg["max_ctrlpts"]),
            initial_control_spacing_mm=float(nurbs_cfg["initial_control_spacing_mm"]),
            lambda_smooth=float(nurbs_cfg["lambda_smooth"]),
            tolerances=nurbs_cfg["fit_tolerance"],
            closed_loop_mode=nurbs_cfg.get("closed_loop_mode", "periodic"),
            corner_angle_deg=float(config["boundary"].get("corner_angle_deg", 135.0)),
        )
        for loop in loops
    ]

    if config["export"].get("write_debug_csv", True):
        write_debug_csv(out_dir, table, cloud)
    if config["export"].get("write_boundary_json", True):
        write_json(out_dir / "boundary_loops.json", loops)
    if config["export"].get("write_nurbs_json", True):
        write_json(out_dir / "nurbs_fit_results.json", results)
    if results:
        pd.DataFrame(
            [
                {
                    "component_id": result.loop.component_id,
                    "role": result.loop.role,
                    "mean_error_mm": result.metrics.mean_error_mm,
                    "max_error_mm": result.metrics.max_error_mm,
                    "hausdorff_error_mm": result.metrics.hausdorff_error_mm,
                    "area_error_ratio": result.metrics.area_error_ratio,
                    "n_control_points": result.metrics.n_control_points,
                    "n_samples": result.metrics.n_samples,
                }
                for result in results
            ]
        ).to_csv(out_dir / "fit_metrics.csv", index=False)
    if config["export"].get("write_plots", True):
        plot_density_cloud(cloud, out_dir / "density_cloud.png", eta=eta)
        plot_boundary_overlay(cloud, loops, out_dir / "boundary_overlay.png")
        plot_fit_overlay(results, out_dir / "nurbs_fit_overlay.png")
    if config["export"].get("write_apdl_macro", False):
        curves = [curve for result in results for curve in result.curves]
        write_apdl_polyline_macro(out_dir / "sampled_section_for_apdl.mac", curves)
    if config["export"].get("write_step", config["export"].get("write_cad", False)):
        curves = [curve for result in results for curve in result.curves]
        write_cad_exchange_stub(out_dir / "section_curves.step", curves, format_name="STEP")
    if config["export"].get("write_iges", config["export"].get("write_cad", False)):
        curves = [curve for result in results for curve in result.curves]
        write_cad_exchange_stub(out_dir / "section_curves.iges", curves, format_name="IGES")
    if (
        config["export"].get("write_stl_preview", config["export"].get("write_cad", False))
        and config.get("revolve", {}).get("enabled", False)
    ):
        curves = [curve for result in results for curve in result.curves]
        mesh = revolve_curves_to_surface(
            curves,
            axis=config["revolve"].get("axis", "z"),
            angle_deg=float(config["revolve"].get("angle_deg", 360.0)),
            n_sections=int(config["revolve"].get("n_sections", 96)),
        )
        write_ascii_stl(out_dir / "revolved_surface_preview.stl", mesh["vertices"], mesh["faces"])

    counts = {
        "nodes": int(table.node_id.size),
        "section_points": int(cloud.xy.shape[0]),
        "boundary_loops": len(loops),
        "fit_results": len(results),
    }
    metrics = [
        {
            "component_id": result.loop.component_id,
            "role": result.loop.role,
            "mean_error_mm": result.metrics.mean_error_mm,
            "max_error_mm": result.metrics.max_error_mm,
            "hausdorff_error_mm": result.metrics.hausdorff_error_mm,
            "area_error_ratio": result.metrics.area_error_ratio,
            "n_control_points": result.metrics.n_control_points,
        }
        for result in results
    ]
    write_json(out_dir / "run_manifest.json", build_run_manifest(config, status="ok", counts=counts, metrics=metrics))

    return {"table": table, "cloud": cloud, "loops": loops, "fit_results": results, "out_dir": out_dir}
