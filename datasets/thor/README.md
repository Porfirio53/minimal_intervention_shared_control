# THOR data

Download THOR from its official Zenodo record and place TSV files in `raw/`. Raw and
processed data are ignored by Git. `datasets.thor.load_thor_tsv` reads a track table and
accepts release-specific column names. Use the first 0.5-1.0 seconds to estimate velocity
and retain future samples as ground truth; do not leak future samples into the predictor.
