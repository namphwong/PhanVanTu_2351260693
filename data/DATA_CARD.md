# Vietnam Hourly Rain Next Hour Domain Adaptation

## Summary

This is a self-collected and processed research dataset for binary next-hour
rain classification and a Hanoi-to-Hai-Phong domain adaptation experiment. It
contains hourly ERA5 reanalysis for 2023-01-01 00:00 through 2025-12-31
23:00 in Asia/Bangkok time. It is a research/demo dataset, not an official
weather forecast or a record of station observations.

## Provenance and license

Weather values are retrieved from the Open-Meteo Historical Weather API using
the Copernicus ERA5 model at 0.25° spatial resolution. A short availability
probe found that explicit ERA5-Land requests returned null for pressure, cloud,
wind, and precipitation fields used by the fixed schema; ERA5 returned these
fields for both locations and was therefore selected as one consistent source.
Open-Meteo states that API data is provided
under CC BY 4.0. Attribution: “Weather data by Open-Meteo.com, based on
Copernicus ERA5.” Source: https://open-meteo.com/en/docs/historical-weather-api
and https://open-meteo.com/en/terms. The raw responses and derived dataset are
released under CC BY 4.0. ERA5-Land source citation is provided in
`CITATION.cff`.

## Locations and collection

| Domain | Latitude | Longitude | Role |
| --- | ---: | ---: | --- |
| Hanoi | 21.0278 | 105.8342 | Source |
| Hai Phong | 20.8449 | 106.6881 | Target |

The collector requests each location and calendar year separately with a
fixed ERA5 model selection. It saves the API JSON response, non-secret
request parameters, retrieval timestamp, and SHA256 checksum. Re-run with
`python scripts/collect_open_meteo.py --config configs/rain_hanoi_haiphong.yaml`.

## Variables

Raw hourly fields are temperature_2m (°C), relative_humidity_2m (%),
dew_point_2m (°C), pressure_msl (hPa), cloud_cover (%), wind_speed_10m
(km/h), wind_direction_10m (degrees), precipitation (mm), and timestamp.
Processed model inputs encode wind direction and hour-of-day as sine/cosine;
each 24-hour window has 11 ordered features. See
`report_data_quality.json` for the authoritative feature order and observed
missingness.

The label `rain_next_hour` is 1 when precipitation in the next hourly record is
at least 0.1 mm and 0 otherwise. A label is omitted if that next timestamp is
not available. Processed labels are withheld for the Hai Phong adaptation
period in both hourly and window files. The raw API files are retained for
provenance and include precipitation; do not derive or use target adaptation
labels during model adaptation.

## Splits

- Hanoi source train: 2023-01-01 through 2024-06-30.
- Hanoi source validation: 2024-07-01 through 2024-09-30.
- Hanoi source test: 2024-10-01 through 2024-12-31.
- Hai Phong unlabeled adaptation: 2023-01-01 through 2024-12-31. Window labels
  are null in the model windows file.
- Hai Phong final test: 2025-01-01 through 2025-12-31.

Windows are created only when all input timestamps and their next-hour target
timestamp belong to the same split. The first and last boundary windows are
therefore excluded. Scalers must be fit on Hanoi source training data only.
Training code must not use Hai Phong labels during adaptation.

## Quality and limits

`report_data_quality.json` records observed row counts, duplicate timestamps,
missing-hour gaps, longest gap, missing rates, label counts by split and file
checksums. Missing values are not silently imputed; windows containing a
non-finite model feature are excluded. ERA5 is a gridded reanalysis at
approximately 0.25° resolution, so its values are estimates and can differ
from local station measurements, particularly near coasts. The dataset is
small geographically and should not be generalized to all of Vietnam without
additional validation.

## Reproduction

```powershell
python scripts/collect_open_meteo.py --config configs/rain_hanoi_haiphong.yaml
python scripts/prepare_dataset.py --config configs/rain_hanoi_haiphong.yaml
```
