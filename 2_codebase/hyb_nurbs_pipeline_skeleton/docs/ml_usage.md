# ML surrogate usage

The neural network module is a surrogate interface for rapid turbine-disk section screening. It does not replace ANSYS topology optimization or finite element verification. The current single real case is suitable for smoke testing the code path only; it must not be reported as formal generalization accuracy.

## Dataset Format

The dataset builder writes a compressed NPZ file with:

- `X`: numeric input features plus missing-value masks.
- `y`: flattened NURBS/B-spline control points.
- `feature_names`, `target_names`, `sample_ids`.
- `train_idx`, `val_idx`, `test_idx`.
- `x_mean`, `x_std`, `y_mean`, `y_std` normalization statistics.

For 15 control points the target vector is:

```text
z = [Px1, Pz1, Px2, Pz2, ..., Px15, Pz15]
```

Missing numeric inputs are filled with `0` and recorded through `is_missing_<field>` mask features and the `missing_fields` column in `ml_index.csv`.

## Build a Dataset

```powershell
python -m hyb_nurbs.ml.dataset build `
  --outputs-root outputs `
  --out data/processed/ml/hyb_nurbs_dataset.npz `
  --index data/processed/ml/ml_index.csv `
  --required-control-points 15 `
  --allow-small-data
```

If the skeleton package is used from `Latest/hyb_nurbs_pipeline_skeleton`, the builder also accepts the existing sibling output folder:

```powershell
python -m hyb_nurbs.ml.dataset build --outputs-root ..\outputs --allow-small-data
```

## Train

```powershell
python -m hyb_nurbs.ml.train `
  --dataset data/processed/ml/hyb_nurbs_dataset.npz `
  --config configs/ml/mlp_baseline.yaml `
  --out outputs/ml/mlp_baseline `
  --smoke-test
```

Training with fewer than 10 samples is refused unless `--smoke-test` or `--allow-small-data` is provided.

## Predict

```powershell
python -m hyb_nurbs.ml.predict `
  --model outputs/ml/mlp_baseline/model.pt `
  --normalizer outputs/ml/mlp_baseline/normalizer.json `
  --input configs/ml/example_input.json `
  --out outputs/ml/mlp_baseline/predicted_control_points.json
```

Prediction JSON contains the input feature vector, predicted control points, model paths, creation time, and a warning if the model came from smoke-test training.

## Current Limitation

With only one real ANSYS/NURBS sample, the ML output validates software integration only. It cannot prove engineering accuracy, topology preservation, or generalization. Formal training needs batch ANSYS samples with consistent inputs, NURBS labels, and downstream FEA metrics.

## Batch ANSYS Integration

Future APDL/Workbench automation should produce one run folder per sample containing:

- `nurbs_fit_results.json`
- `fit_metrics.csv`
- `run_manifest.json`
- `config_resolved.yaml`

The dataset builder will scan these folders and turn each valid 15-control-point result into one supervised sample.
