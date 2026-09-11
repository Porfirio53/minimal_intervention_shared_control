# THOR data

Download THOR from its official Zenodo record and place the 13 ordinary 3-D TSV files in
`raw/` (do not use the `_6D.tsv` files for this first pipeline). Raw and processed data are
ignored by Git. Convert each Qualisys wide table to normalized tracks with:

```bash
.venv/bin/python scripts/preprocess_thor.py \
  datasets/thor/raw/Exp_1_run_1.tsv \
  datasets/thor/processed/Exp_1_run_1.tsv
```

The converter averages valid `Helmet_N` markers, converts millimetres to metres, and
starts a new track segment after missing observations or a timestamp gap. The normalized
output is readable by `datasets.thor.load_thor_tsv`. Use the first 0.5-1.0 seconds to
estimate velocity and retain future samples as ground truth; do not leak future samples
into the predictor.
