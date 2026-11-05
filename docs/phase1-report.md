# Phase 1 execution report

## Status

Local data collection, preparation, validation, data card, and Kaggle package
builder are ready. Phase 1 remains open until a verified Kaggle account is
authenticated and the public dataset upload is checked. No Kaggle token or
credential is stored in the repository.

## Data source decision

The planned ERA5-Land request was probed at both coordinates. It returned
values for temperature, relative humidity, and dew point, but nulls for the
pressure, cloud, wind, and precipitation fields required by the schema. A
request using the same variables without fixing ERA5-Land returned all fields.
To preserve one source and one complete schema for both cities, the pipeline
uses fixed ERA5 reanalysis. The data card records this scope change and its
resolution.

## Collection and quality results

- Raw API files: six JSON responses, one per city and year (2023-2025), with
  fixed ERA5 model, request parameters, retrieval time, and SHA256 checksums.
- Hourly rows: 26,304 for Hanoi and 26,304 for Hai Phong.
- Duplicate timestamps: zero for both cities.
- Missing hourly timestamps: zero for both cities; the report records an empty
  gap list for each.
- Missing rates: zero across the eight requested weather variables for both
  cities.
- Processed windows: 43,728 total: 13,104 source train, 2,184 source
  validation, 2,184 source test, 17,520 unlabeled target adaptation, and 8,736
  target final test.
- Target adaptation labels are null in both processed CSV and Parquet outputs.
  Raw precipitation is retained for provenance as required by the plan; model
  adaptation must not derive or consume target labels.
- SHA256 `hourly_clean.csv`:
  `23bc6667a82bef161635d88a46c7efb30b4e084bbcbb4d294487fc81e95b6a46`.
- SHA256 `windows_24h.parquet`:
  `f9e00c906fe9b1215ae8e99e7e5f18ee6680a0e154a2a7f06f1c9fcc74280d0a`.
- Parquet stores each 24 × 11 feature vector as `list<float>` and records
  feature order in the quality report.

## Verification

- `python -m pytest -q --basetemp=.pytest-tmp`: 11 passed.
- `ruff check scripts tests pyproject.toml`: passed.
- `git diff --check`: passed.
- All six raw file checksums and all three processed-file checksums were
  independently rechecked.
- Kaggle package assembly is covered by a test and includes raw, processed,
  data card, license, citation, checksums, quality report, and generated
  `dataset-metadata.json`.

## Remaining gate

The Kaggle CLI is installed, but `kaggle datasets list --mine --csv` reports
that the CLI is not authenticated. The installed PyPI CLI version does not
expose `kaggle auth login`. Configure a Kaggle API key or access token in the
user-local Kaggle configuration (never in the repo), provide the verified
Kaggle username, then run the package publisher. The remaining steps are to
create the public dataset and verify its resulting URL anonymously.
