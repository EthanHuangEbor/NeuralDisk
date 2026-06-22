# HYB NURBS File Layout and Algorithm Interfaces

This document defines the stable storage layout, naming rules, and module interfaces for the NeuralDisk topology-density-to-NURBS pipeline.

The current real ANSYS exports are archived inside this repo:

- `data/raw/hyb_current_actual/NLIST.lis`
- `data/raw/hyb_current_actual/export1.txt`

Future exported cases should follow the same case-based layout so experiments stay reproducible.

## 1. Repository Boundaries

```text
NeuralDisk/
  hyb_nurbs/        # source package
  configs/          # defaults, case configs, ML configs
  data/             # raw/interim/processed data
  outputs/          # generated run artifacts; ignored by git
  reports/          # curated figures/tables for public docs
  docs/             # human documentation
  tests/            # automated tests
```

Rules:

- Raw ANSYS exports live under `data/raw/<case_id>/`.
- Generated files live under `outputs/` and should be reproducible from raw inputs plus config.
- Curated figures/tables for papers or slides live under `reports/`.
- Do not mix generated artifacts into `hyb_nurbs/`, `configs/`, or `tests/`.
- Keep private forms, identity images, credentials, archives, and large model checkpoints out of the public repo.

## 2. Canonical Data Layout

```text
data/
  raw/
    hyb_current_actual/
      NLIST.lis
      export1.txt
      manifest.yaml
      notes.md
    hyb_0002_eta_sweep/
      NLIST.lis
      export1.txt
      manifest.yaml
      notes.md
    _template/
      manifest.yaml
      notes.md

  interim/
    README.md

  processed/
    README.md
    ml/
      hyb_nurbs_dataset.npz
      ml_index.csv
```

`manifest.yaml` captures provenance:

```yaml
case_id: hyb_current_actual
source: ansys
node_file: NLIST.lis
density_file: export1.txt
node_count_expected: 1050
density_count_expected: 1050
length_unit: m
working_length_unit: mm
density_field: topology_density
exported_at: null
imported_at: 2026-06-17
public_release: true
notes: Current real HYB export.
```

Naming rules:

- New case IDs use `hyb_####_short_name`.
- Raw file names keep the ANSYS/export names: `NLIST.lis`, `export1.txt`.
- If multiple density exports exist, use clear names such as `density_topology.txt` and record the selected file in `manifest.yaml`.
- Use ASCII in machine-read paths. Chinese is fine in human docs.

## 3. Run Output Layout

Each run gets an immutable run directory:

```text
outputs/
  <case_id>/
    run_YYYYMMDD_HHMMSS_eta050_tri_iso/
      config_resolved.yaml
      run_manifest.json
      merged_node_density.csv
      section_cloud.csv
      boundary_loops.json
      nurbs_fit_results.json
      fit_metrics.csv
      density_cloud.png
      boundary_overlay.png
      nurbs_fit_overlay.png
```

Generated file naming:

- CSV tables use snake_case.
- JSON geometry/spec files use snake_case.
- Images describe the visible layer, for example `boundary_overlay.png`.
- CAD/FEA exports should include the target format, for example `sampled_section_for_apdl.mac`.

## 4. Pipeline Stage Contracts

The pipeline remains stage-based:

```text
ANSYS files
-> NodeDensityTable
-> SectionCloud
-> list[BoundaryLoop]
-> list[BoundaryLoop] after cleanup/resampling
-> list[FitResult]
-> exported files
```

### IO

Module: `hyb_nurbs.io.ansys`

- `parse_nlist(path)`: returns `node_id, x, y, z`.
- `parse_density(path)`: returns `node_id, rho`.
- `load_node_density(node_file, density_file, scale_to_mm=True)`: validates matching IDs and returns coordinates in working units.

Requirements:

- `node_id` is unique in both files.
- Node and density IDs match exactly.
- The current baseline parses as 1050 nodes and 1050 densities.

### Projection

Module: `hyb_nurbs.preprocess.projection`

- Default section axes: `("x", "z")`.
- Default thickness axis: `y`.
- Default mode: `aggregate`.
- Default density aggregation: `max`.
- Preserve source node traceability when aggregating projected points.

### Boundary Extraction

Module: `hyb_nurbs.boundary.iso`

- Primary method: `tri_iso`.
- Fallback methods: `grid_iso`, `alpha_shape`.
- Do not fit all `rho >= eta` nodes directly.
- Close retained-region boundaries before fitting.

### Boundary Postprocess

Module: `hyb_nurbs.boundary.postprocess`

- Ensure loops are closed.
- Classify outer loops and holes.
- Filter tiny fragments.
- Resample by arc length before fitting.

### NURBS/B-Spline Fitting

Modules: `hyb_nurbs.nurbs.evaluate`, `hyb_nurbs.nurbs.fitting`

- Default degree: 3.
- Default weights: fixed `1.0`.
- Fit in millimeters.
- Refine control-point count until mean, max, and area error tolerances pass or `max_ctrlpts` is reached.

### Validation

Module: `hyb_nurbs.validation.metrics`

Track:

- `mean_error_mm`
- `max_error_mm`
- `hausdorff_error_mm`
- `area_error_ratio`
- `n_control_points`
- `n_samples`

### Export

Module: `hyb_nurbs.exporters.files`

- Write debug CSV files.
- Serialize loops, NURBS results, and manifests as JSON.
- Keep CAD/FEA outputs disabled unless boundary and fit validation pass.

## 5. Config Design

Reusable configs:

```text
configs/
  default.yaml
  cases/
    hyb_current_actual.yaml
    hyb_0002_eta_sweep.yaml
  ml/
    mlp_baseline.yaml
```

Case configs point to raw data and output roots relative to the YAML file:

```yaml
case:
  case_id: hyb_current_actual

input:
  node_file: ../../data/raw/hyb_current_actual/NLIST.lis
  density_file: ../../data/raw/hyb_current_actual/export1.txt

export:
  out_root: ../../outputs
  run_name: auto
```

## 6. Run Manifest

Each run writes `run_manifest.json` with:

- `case_id`
- `run_name`
- `input_files`
- `counts`
- `parameters`
- `metrics`
- `environment`
- `status`

The manifest is the link between a generated output folder, raw data, config, and validation metrics.

## 7. Next Data Tasks

1. Add each new ANSYS export as a `data/raw/<case_id>/` folder.
2. Fill the manifest and notes before running the pipeline.
3. Add a matching `configs/cases/<case_id>.yaml`.
4. Run `python -m hyb_nurbs.cli configs/cases/<case_id>.yaml`.
5. Promote selected final figures/tables to `reports/`.
