# Outputs

Pipeline and ML runs write generated artifacts here. Output folders are ignored by git by default.

Recommended run layout:

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

Promote only selected figures/tables to `reports/` when they are needed for papers, slides, or public documentation.
