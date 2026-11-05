# Phase 1 execution report

## Status

Phase 1 is complete. The dataset is published publicly on Kaggle at
https://www.kaggle.com/datasets/phantus/vietnam-hourly-rain-next-hour-domain-adaptation.
It was created under the `phantus` account with public visibility and the
Attribution 4.0 International (CC BY 4.0) license. Kaggle confirmed successful
creation, and its dataset page displays the uploaded raw and processed
directories, data card, license, and citation. The ZIP upload was extracted by
Kaggle into its 15-file directory structure.

The Kaggle browser session was used for publication because the local CLI was
not authenticated. No Kaggle API token or credential was added to the
repository. The dataset page includes a summary describing the source, period,
domain adaptation splits, and withheld target adaptation labels.

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
- `git diff --check`: passed before Kaggle publication.
- All six raw file checksums and all three processed-file checksums were
  independently rechecked before upload.
- Kaggle package assembly includes raw, processed, data card, license,
  citation, checksums, quality report, and generated `dataset-metadata.json`.
- Kaggle confirmed successful public dataset creation. The dataset page shows
  the uploaded directory tree and 15-file summary.

The Kaggle page was verified in the authenticated owner session. An anonymous
browser check was not available in this session.
