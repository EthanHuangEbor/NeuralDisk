# Data Directory

`data/` contains inputs and derived datasets used by the NeuralDisk pipeline.

```text
data/
  raw/        # Versioned, immutable input cases from ANSYS or experiments
  interim/    # Temporary conversion/cache files; ignored by git
  processed/  # Generated training tables, NPZ files, and feature indexes
```

Rules:

- Treat every `data/raw/<case_id>/` folder as immutable after it is added.
- Put one `manifest.yaml` and one `notes.md` in every raw case folder.
- Generated artifacts belong in `outputs/` or `data/processed/`, not in `data/raw/`.
- Sensitive or personal materials must stay outside this repo or under a local ignored `private/` folder.
