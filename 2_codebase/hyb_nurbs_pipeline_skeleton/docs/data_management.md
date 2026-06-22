# Data Management Policy

This repo separates source code, raw data, generated outputs, and public-facing reports so new data can be added without mixing provenance or leaking private files.

## Folder Roles

```text
data/raw/         Immutable input cases. Commit only public-release-safe exports.
data/interim/     Temporary conversions and caches. Ignored by git.
data/processed/   Generated ML/table datasets. Mostly ignored by git.
outputs/          Pipeline and ML run artifacts. Ignored by git.
reports/          Curated figures/tables selected for papers or slides.
docs/             Human documentation and file-management rules.
```

## Raw Case Checklist

Before adding a new `data/raw/<case_id>/` folder:

1. Use a stable case ID such as `hyb_0002_eta045_refined_mesh`.
2. Include `NLIST.lis`, `export1.txt`, `manifest.yaml`, and `notes.md`.
3. Record units, expected node/density counts, source software, export date, and public-release status.
4. Keep filenames ASCII for machine-read data paths.
5. Never modify raw files after the case is committed. Add a new case folder for revised exports.

## Config Checklist

Each public raw case should have a matching config under `configs/cases/`.

Use paths relative to the config file:

```yaml
input:
  node_file: ../../data/raw/<case_id>/NLIST.lis
  density_file: ../../data/raw/<case_id>/export1.txt

export:
  out_root: ../../outputs
  run_name: auto
```

## Output Promotion

Generated output directories are not versioned. When an output is important for a manuscript, presentation, or release:

- Copy only the final selected PNG/CSV/JSON into `reports/figures/` or `reports/tables/`.
- Note the originating `case_id`, `run_name`, and commit hash in the report caption or notes.
- Do not promote exploratory files that cannot be regenerated.

## Privacy and Public Release

Keep the public repo free of:

- Identity documents, signatures, phone numbers, addresses, and student IDs.
- Competition forms containing personal data.
- Credentials, API keys, tokens, and private notes.
- Large archives, model checkpoints, and one-off generated outputs.

Use a local ignored `private/` folder or an external drive for those materials.
