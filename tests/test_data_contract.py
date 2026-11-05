"""Tests for hourly data contracts, labels, windows, boundaries, and checksums."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from scripts.collect_open_meteo import build_request
from scripts.collect_open_meteo import sha256_file as raw_sha256
from scripts.prepare_dataset import FEATURE_COLUMNS, load_raw_city, prepare_city
from scripts.publish_kaggle import build_package


def _fixture_frame(timestamps: list[str], city: str = "hanoi") -> pd.DataFrame:
    size = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).tz_localize("Asia/Bangkok"),
            "city": city,
            "temperature_2m": [25.0] * size,
            "relative_humidity_2m": [80.0] * size,
            "dew_point_2m": [21.0] * size,
            "pressure_msl": [1008.0] * size,
            "cloud_cover": [50.0] * size,
            "wind_speed_10m": [8.0] * size,
            "wind_direction_10m": [90.0] * size,
            "precipitation": [0.0, 0.0, 0.2] + [0.0] * max(0, size - 3),
        }
    )


def _split_config() -> dict:
    return {
        "source_train": ["2023-01-01", "2023-01-01"],
        "source_validation": ["2023-01-02", "2023-01-02"],
        "source_test": ["2023-01-03", "2023-01-03"],
        "target_adaptation_unlabeled": ["2023-01-01", "2023-01-02"],
        "target_final_test": ["2023-01-03", "2023-01-03"],
    }


def test_label_is_next_hour_and_features_are_finite():
    timestamps = (
        pd.date_range("2023-01-01 00:00", periods=5, freq="h").strftime("%Y-%m-%d %H:%M").tolist()
    )
    frame = _fixture_frame(timestamps)
    _, windows, quality = prepare_city(frame, _split_config(), 0.1, 2)

    assert windows.iloc[0]["rain_next_hour"] == 1
    values = windows.iloc[0]["features"]
    assert len(values) == 2 * len(FEATURE_COLUMNS)
    assert all(pd.notna(value) for value in values)
    assert quality["duplicate_timestamps_removed"] == 0


def test_windows_never_cross_split_boundary():
    timestamps = [
        "2023-01-01 22:00",
        "2023-01-01 23:00",
        "2023-01-02 00:00",
        "2023-01-02 01:00",
        "2023-01-02 02:00",
    ]
    hourly, windows, _ = prepare_city(_fixture_frame(timestamps), _split_config(), 0.1, 2)

    assert len(windows) == 1
    assert windows.iloc[0]["split"] == "source_validation"
    assert windows.iloc[0]["window_start_timestamp"].startswith("2023-01-02T00:00")
    assert pd.isna(hourly.loc[hourly.timestamp.dt.day.eq(1), "rain_next_hour"].iloc[-1])


def test_target_adaptation_labels_are_hidden():
    timestamps = (
        pd.date_range("2023-01-01 00:00", periods=5, freq="h").strftime("%Y-%m-%d %H:%M").tolist()
    )
    hourly, windows, _ = prepare_city(
        _fixture_frame(timestamps, "hai_phong"), _split_config(), 0.1, 2
    )

    assert len(windows) == 3
    assert windows["rain_next_hour"].isna().all()
    assert hourly["rain_next_hour"].isna().all()


def test_loader_localizes_timezone_and_filters_full_end_date(tmp_path):
    response = {
        "hourly": {
            "time": [
                "2023-01-01T00:00",
                "2023-01-01T01:00",
                "2023-01-02T23:00",
                "2023-01-03T00:00",
            ],
            "temperature_2m": [20, 20, 20, 20],
            "relative_humidity_2m": [80, 80, 80, 80],
            "dew_point_2m": [18, 18, 18, 18],
            "pressure_msl": [1000, 1000, 1000, 1000],
            "cloud_cover": [30, 30, 30, 30],
            "wind_speed_10m": [5, 5, 5, 5],
            "wind_direction_10m": [180, 180, 180, 180],
            "precipitation": [0, 0, 0, 0],
        }
    }
    raw = tmp_path / "hanoi"
    raw.mkdir()
    (raw / "2023.json").write_text(json.dumps(response), encoding="utf-8")

    frame = load_raw_city(tmp_path, "hanoi", "2023-01-01", "2023-01-02", "Asia/Bangkok")

    assert len(frame) == 3
    assert str(frame.iloc[0]["timestamp"].tz) == "Asia/Bangkok"
    assert frame.iloc[-1]["timestamp"].hour == 23


def test_raw_sha256_matches_standard_library(tmp_path):
    target = tmp_path / "raw.json"
    target.write_bytes(b"fixture")

    assert raw_sha256(target) == hashlib.sha256(b"fixture").hexdigest()


def test_collector_pins_model_timezone_and_yearly_dates():
    from urllib.parse import parse_qs, urlparse

    url = build_request({"latitude": 21.0, "longitude": 105.8}, 2024, "Asia/Bangkok", "era5")
    params = parse_qs(urlparse(url).query)

    assert params["models"] == ["era5"]
    assert params["start_date"] == ["2024-01-01"]
    assert params["end_date"] == ["2024-12-31"]
    assert params["timezone"] == ["Asia/Bangkok"]


def test_duplicate_hour_is_removed_and_counted():
    timestamps = [
        "2023-01-01 00:00",
        "2023-01-01 01:00",
        "2023-01-01 01:00",
        "2023-01-01 02:00",
    ]
    hourly, _, quality = prepare_city(_fixture_frame(timestamps), _split_config(), 0.1, 2)

    assert len(hourly) == 3
    assert quality["duplicate_timestamps_removed"] == 1
    assert not hourly["timestamp"].duplicated().any()


def test_quality_report_records_each_missing_hour_gap():
    timestamps = ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00"]
    hourly, windows, quality = prepare_city(_fixture_frame(timestamps), _split_config(), 0.1, 1)

    assert quality["missing_hour_count"] == 1
    assert quality["longest_gap_hours"] == 1
    assert quality["gaps"][0]["start"].startswith("2023-01-01T02:00")
    assert windows["target_timestamp"].str.endswith("01:00:00+07:00").all()
    assert pd.isna(hourly.loc[hourly.timestamp.dt.hour.eq(1), "rain_next_hour"].iloc[0])


def test_kaggle_package_contains_metadata_and_dataset_licenses(tmp_path):
    root = tmp_path / "repo"
    for folder in (root / "data" / "raw", root / "data" / "processed"):
        folder.mkdir(parents=True)
    (root / "data" / "raw" / "raw.json").write_text("{}", encoding="utf-8")
    (root / "data" / "processed" / "windows.parquet").write_bytes(b"parquet")
    for name in ("DATA_CARD.md", "LICENSE", "CITATION.cff"):
        (root / "data" / name).write_text("fixture", encoding="utf-8")

    package = build_package(root, tmp_path / "dist", "dataset-review")
    metadata = json.loads((package / "dataset-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "dataset-review/vietnam-hourly-rain-next-hour-domain-adaptation"
    assert (package / "raw" / "raw.json").exists()
    assert (package / "processed" / "windows.parquet").exists()
    assert (package / "CITATION.cff").exists()
