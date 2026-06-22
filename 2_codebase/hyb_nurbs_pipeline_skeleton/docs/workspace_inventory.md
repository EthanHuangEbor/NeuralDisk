# Workspace Inventory

This repo lives inside a larger local workspace with two major folders:

```text
D:/Project/1-NetrualDisk/
  Before/   # Historical Fengru Cup template and earlier report materials
  Latest/   # Current project deliverables, raw exports, slides, reports, and repo
```

## Public Repo Boundary

The public GitHub repo should be `Latest/hyb_nurbs_pipeline_skeleton/`.

It contains:

- Python source under `hyb_nurbs/`.
- Case configs under `configs/`.
- Public-release-safe raw baseline data under `data/raw/hyb_current_actual/`.
- Documentation under `docs/`.
- Tests under `tests/`.

## Local-Only Materials

The outer workspace also contains presentations, PDFs, archive `.zip` files, competition packages, and personal identity images/forms. Those files are useful locally but should not be committed to the public repo unless each file has been reviewed for public release.

Recommended handling:

- Keep historical templates in `Before/`.
- Keep current slides, submissions, and private competition materials outside the Git repo.
- Move any future private repo-adjacent files into an ignored `private/` folder.
- Promote only sanitized figures/tables into `reports/`.
