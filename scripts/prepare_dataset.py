"""Validate raw hourly archives and produce clean hourly and window datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

RAW_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]
FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "precipitation",
    "hour_sin",
    "hour_cos",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_city(raw_dir: Path, city: str, start: str, end: str, timezone: str) -> pd.DataFrame:
    pieces = []
    for path in sorted((raw_dir / city).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hourly = payload["hourly"]
        missing = set(RAW_COLUMNS) - set(hourly)
        if missing:
            raise ValueError(f"{path}: missing API columns {sorted(missing)}")
        frame = pd.DataFrame({name: hourly[name] for name in RAW_COLUMNS})
        frame["timestamp"] = pd.to_datetime(hourly["time"], errors="raise")
        frame["city"] = city
        pieces.append(frame)
    if not pieces:
        raise FileNotFoundError(f"No raw hourly JSON found for {city} under {raw_dir}")

    result = pd.concat(pieces, ignore_index=True)
    for column in RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["timestamp"] = result["timestamp"].dt.tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    )
    result = result.sort_values("timestamp").reset_index(drop=True)
    start_bound = pd.Timestamp(start, tz=timezone)
    end_bound = pd.Timestamp(f"{end} 23:00:00", tz=timezone)
    result = result.loc[result["timestamp"].between(start_bound, end_bound)].copy()
    return result


def assign_split(timestamp: pd.Timestamp, split_config: dict, city: str) -> str | None:
    if city == "hanoi":
        names = ("source_train", "source_validation", "source_test")
    else:
        names = ("target_adaptation_unlabeled", "target_final_test")
    day = timestamp.date()
    for name in names:
        first, last = (pd.Timestamp(v).date() for v in split_config[name])
        if first <= day <= last:
            return name
    return None


def prepare_city(
    frame: pd.DataFrame, splits: dict, threshold_mm: float, window_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    if duplicate_count:
        frame = frame.drop_duplicates("timestamp", keep="first")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    expected = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max(), freq="h")
    missing_times = expected.difference(pd.DatetimeIndex(frame["timestamp"]))
    gaps = []
    if len(missing_times):
        gap_group = (missing_times.to_series().diff() != np.timedelta64(1, "h")).cumsum()
        for _, group in missing_times.to_series().groupby(gap_group):
            gaps.append(
                {
                    "start": group.iloc[0].isoformat(),
                    "end": group.iloc[-1].isoformat(),
                    "missing_hours": int(len(group)),
                }
            )

    frame["wind_direction_sin"] = np.sin(np.deg2rad(frame["wind_direction_10m"]))
    frame["wind_direction_cos"] = np.cos(np.deg2rad(frame["wind_direction_10m"]))
    hour = frame["timestamp"].dt.hour
    frame["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    frame["split"] = frame.apply(
        lambda row: assign_split(row["timestamp"], splits, row["city"]), axis=1
    )
    next_timestamp = frame["timestamp"].shift(-1)
    next_split = frame["split"].shift(-1)
    adjacent_hour = (next_timestamp - frame["timestamp"]) == np.timedelta64(1, "h")
    same_split = frame["split"].eq(next_split)
    next_precipitation_present = frame["precipitation"].shift(-1).notna()
    frame["rain_next_hour"] = (frame["precipitation"].shift(-1) >= threshold_mm).astype("Int64")
    valid_label = adjacent_hour & same_split & next_precipitation_present
    frame.loc[~valid_label, "rain_next_hour"] = pd.NA
    frame.loc[frame["split"] == "target_adaptation_unlabeled", "rain_next_hour"] = pd.NA

    windows = []
    for end_idx in range(window_hours - 1, len(frame) - 1):
        window = frame.iloc[end_idx - window_hours + 1 : end_idx + 1]
        label_row = frame.iloc[end_idx + 1]
        split = frame.iloc[end_idx]["split"]
        end_timestamp = frame.iloc[end_idx]["timestamp"]
        if (label_row["timestamp"] - end_timestamp) != np.timedelta64(1, "h"):
            continue
        # Do not allow any input/label timestamp to cross a time-split boundary.
        if split is None or window["split"].isna().any() or not window["split"].eq(split).all():
            continue
        if label_row["split"] != split:
            continue
        if not window["timestamp"].diff().dropna().eq(np.timedelta64(1, "h")).all():
            continue
        values = window[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            continue
        is_unlabeled_adaptation = split == "target_adaptation_unlabeled"
        if not is_unlabeled_adaptation and pd.isna(frame.iloc[end_idx]["rain_next_hour"]):
            continue
        windows.append(
            {
                "city": frame.iloc[end_idx]["city"],
                "split": split,
                "window_start_timestamp": window.iloc[0]["timestamp"].isoformat(),
                "window_end_timestamp": window.iloc[-1]["timestamp"].isoformat(),
                "target_timestamp": label_row["timestamp"].isoformat(),
                "rain_next_hour": pd.NA
                if is_unlabeled_adaptation
                else int(frame.iloc[end_idx]["rain_next_hour"]),
                "features": values.reshape(-1).tolist(),
            }
        )
    window_frame = pd.DataFrame(windows)
    split_summary = {}
    if not window_frame.empty:
        for split, subset in window_frame.groupby("split"):
            labels = subset["rain_next_hour"].dropna()
            split_summary[split] = {
                "samples": int(len(subset)),
                "labeled_samples": int(len(labels)),
                "class_counts": {
                    str(key): int(value) for key, value in labels.value_counts().items()
                },
            }
    quality = {
        "city": frame["city"].iloc[0],
        "rows": int(len(frame)),
        "timestamp_start": frame["timestamp"].min().isoformat(),
        "timestamp_end": frame["timestamp"].max().isoformat(),
        "duplicate_timestamps_removed": duplicate_count,
        "missing_hour_count": int(len(missing_times)),
        "longest_gap_hours": max((gap["missing_hours"] for gap in gaps), default=0),
        "gaps": gaps,
        "missing_rate_by_variable": {
            name: float(frame[name].isna().mean()) for name in RAW_COLUMNS
        },
        "windows_by_split": split_summary,
    }
    return frame, window_frame, quality


def prepare(config_path: Path, raw_dir: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    all_hourly = []
    all_windows = []
    reports = {}
    for city in data_config["cities"]:
        raw = load_raw_city(
            raw_dir,
            city,
            data_config["start"][:10],
            data_config["end"][:10],
            data_config["timezone"],
        )
        hourly, windows, quality = prepare_city(
            raw,
            config["splits"],
            float(data_config["precipitation_threshold_mm"]),
            int(data_config["window_hours"]),
        )
        all_hourly.append(hourly)
        all_windows.append(windows)
        reports[city] = quality

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = output_dir / "hourly_clean.csv"
    windows_path = output_dir / "windows_24h.parquet"
    pd.concat(all_hourly, ignore_index=True).to_csv(hourly_path, index=False)
    windows = pd.concat(all_windows, ignore_index=True)
    features = pa.array(windows["features"].tolist(), type=pa.list_(pa.float32()))
    table = pa.Table.from_pandas(windows.drop(columns=["features"]), preserve_index=False)
    table = table.append_column("features", features)
    pq.write_table(table, windows_path)
    report = {
        "dataset": "Vietnam Hourly Rain Next Hour Domain Adaptation",
        "timezone": data_config["timezone"],
        "frequency": "hourly",
        "window_hours": int(data_config["window_hours"]),
        "feature_order": FEATURE_COLUMNS,
        "threshold_mm": float(data_config["precipitation_threshold_mm"]),
        "cities": reports,
        "processed_files": {
            "hourly_clean.csv": {
                "rows": int(sum(len(f) for f in all_hourly)),
                "sha256": sha256_file(hourly_path),
            },
            "windows_24h.parquet": {"rows": int(len(windows)), "sha256": sha256_file(windows_path)},
        },
    }
    (output_dir / "report_data_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksums = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rain_hanoi_haiphong.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_dir = args.raw_dir or Path(config["data"]["raw_dir"])
    output_dir = args.output_dir or Path(config["data"]["processed_dir"])
    report = prepare(args.config, raw_dir, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
