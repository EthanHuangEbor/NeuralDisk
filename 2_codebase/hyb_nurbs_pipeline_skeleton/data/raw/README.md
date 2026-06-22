# Raw Data

Raw cases are the stable source of truth for reproducible pipeline runs.

Expected structure:

```text
data/raw/
  hyb_0001_example/
    NLIST.lis
    export1.txt
    manifest.yaml
    notes.md
```

Case IDs should use `hyb_####_short_name` for new batches. The existing `hyb_current_actual` case is the current baseline export retained for regression testing.
