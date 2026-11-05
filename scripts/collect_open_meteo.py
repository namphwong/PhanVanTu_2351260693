"""Collect yearly Open-Meteo ERA5 hourly archives for configured cities."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, retries: int = 5, timeout: int = 90) -> dict:
    for attempt in range(retries):
        request = Request(url, headers={"User-Agent": "slarda-rain-domain-adaptation/0.1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if "hourly" not in payload or "time" not in payload["hourly"]:
                raise ValueError(f"Open-Meteo response missing hourly data: {payload}")
            return payload
        except HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
        except (URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("Request retry loop exited unexpectedly")


def build_request(coordinates: dict, year: int, timezone: str, model: str) -> str:
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
        "models": model,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }
    return f"{API_URL}?{urlencode(params)}"


def collect(config_path: Path, output_dir: Path) -> list[Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    first_year = date.fromisoformat(data_config["start"][:10]).year
    last_year = date.fromisoformat(data_config["end"][:10]).year
    timezone = data_config["timezone"]
    model = data_config["model"]
    created: list[Path] = []
    checksums: list[str] = []

    for city, coordinates in data_config["cities"].items():
        for year in range(first_year, last_year + 1):
            request_url = build_request(coordinates, year, timezone, model)
            payload = request_json(request_url)
            times = payload["hourly"]["time"]
            expected_hours = 8784 if year % 4 == 0 else 8760
            if len(times) != expected_hours:
                raise ValueError(
                    f"{city} {year}: got {len(times)} hours, expected {expected_hours}"
                )
            for name, values in payload["hourly"].items():
                if len(values) != len(times):
                    raise ValueError(f"{city} {year}: hourly field {name} has a different length")

            payload["collection"] = {
                "city": city,
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "requested_year": year,
                "timezone": timezone,
                "model": model.upper(),
                "api_url_without_parameters": API_URL,
                "request_parameters": {
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "hourly": HOURLY_VARIABLES,
                    "timezone": timezone,
                    "models": model,
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                    "timeformat": "iso8601",
                },
                "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            target = output_dir / city / f"{year}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            digest = sha256_file(target)
            checksums.append(f"{digest}  {target.relative_to(output_dir).as_posix()}")
            created.append(target)

    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rain_hanoi_haiphong.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or Path(config["data"]["raw_dir"])
    created = collect(args.config, output)
    print(f"Downloaded {len(created)} raw files to {output}")


if __name__ == "__main__":
    main()
