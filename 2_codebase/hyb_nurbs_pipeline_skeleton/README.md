# NeuralDisk

NeuralDisk is a turbine-disk section reconstruction and surrogate-modeling repo. It turns ANSYS topology-density exports into a cleaned 2D boundary, fits cubic NURBS/B-spline curves, validates the fit, and prepares data for later neural surrogate training.

The repository is organized around reproducible data cases: raw exports stay under `data/raw/`, generated artifacts stay under `outputs/`, and only curated human-facing material goes under `reports/`.

## Repository Layout

```text
NeuralDisk/
  hyb_nurbs/                  # Python package: IO, projection, boundary, NURBS, CAD, ML
  configs/
    default.yaml              # Shared pipeline defaults
    cases/                    # One YAML per data case/run family
  data/
    raw/                      # Versioned ANSYS/input cases
    interim/                  # Disposable intermediate data, ignored by git
    processed/                # Derived ML/table datasets, generated and mostly ignored
  outputs/                    # Pipeline/ML run outputs, ignored by git except README
  reports/                    # Curated figures/tables for papers and slides
  docs/                       # Interfaces, data rules, usage notes, workspace inventory
  tests/                      # Regression and smoke tests
```

## Current Case

The baseline real export is stored at:

- `data/raw/hyb_current_actual/NLIST.lis`
- `data/raw/hyb_current_actual/export1.txt`
- `data/raw/hyb_current_actual/manifest.yaml`
- `configs/cases/hyb_current_actual.yaml`

This case contains 1050 ANSYS nodes and 1050 topology-density values. It is suitable for pipeline regression tests and smoke ML integration, not for claiming neural-network generalization.

## Add a New Data Case

1. Copy `data/raw/_template/` to `data/raw/hyb_####_short_name/`.
2. Place raw ANSYS exports in that folder. Keep canonical names `NLIST.lis` and `export1.txt` unless the manifest records a different filename.
3. Fill in `manifest.yaml` with source, unit, expected counts, author/date, and notes.
4. Add a matching config under `configs/cases/`, using paths relative to that YAML file:

   ```yaml
   case:
     case_id: hyb_0002_short_name

   input:
     node_file: ../../data/raw/hyb_0002_short_name/NLIST.lis
     density_file: ../../data/raw/hyb_0002_short_name/export1.txt

   export:
     out_root: ../../outputs
     run_name: auto
   ```

5. Run the pipeline and inspect the generated manifest, metrics, and overlays.

## Run

```powershell
python -m pip install -e ".[test]"
python -m hyb_nurbs.cli configs/cases/hyb_current_actual.yaml
python -m pytest
```

For ML smoke usage:

```powershell
python -m hyb_nurbs.ml.dataset build --outputs-root outputs --allow-small-data
python -m hyb_nurbs.ml.train --dataset data/processed/ml/hyb_nurbs_dataset.npz --config configs/ml/mlp_baseline.yaml --out outputs/ml/mlp_baseline --smoke-test
```

## Public Repo Hygiene

Do not commit identity documents, competition forms containing personal data, `.zip` archives, private notes, credentials, model checkpoints, or large generated outputs. Keep those in a local `private/` or external archive. The `.gitignore` is set up to reduce accidental uploads, but review `git status` before every push.

See `docs/data_management.md` for the full file-management policy.
