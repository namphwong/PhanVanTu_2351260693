# Dataset files

Raw API responses and processed datasets are generated locally and are not
committed to this Git repository. Run the collector and preparation scripts
from the repository root. The public Kaggle package is assembled from these
outputs by `scripts/publish_kaggle.py`.

- `raw/<city>/<year>.json`: original Open-Meteo ERA5 response plus request and
  retrieval metadata.
- `raw/SHA256SUMS.txt`: checksums of raw response files.
- `processed/hourly_clean.csv`: local hourly records, derived cyclic features,
  rain label and split assignment.
- `processed/windows_24h.parquet`: flattened 24 × 11 float32 model inputs,
  timestamps, split and label. Adaptation split labels are null.
- `processed/report_data_quality.json`: counts, gaps, missingness, split sizes
  and SHA256 values.

See `DATA_CARD.md` in this directory for provenance and limitations.
