"""Package the Phase 1 dataset and optionally create a public Kaggle dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

DATASET_SLUG = "vietnam-hourly-rain-next-hour-domain-adaptation"


def build_package(repo_root: Path, output_dir: Path, username: str) -> Path:
    destination = output_dir / "kaggle"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ("raw", "processed"):
        source = repo_root / "data" / name
        if not source.exists() or not any(source.rglob("*")):
            raise FileNotFoundError(f"Missing {source}; collect and prepare data before packaging")
        shutil.copytree(source, destination / name)
    for name in ("DATA_CARD.md", "LICENSE", "CITATION.cff"):
        shutil.copy2(repo_root / "data" / name, destination / name)
    metadata = {
        "title": "Vietnam Hourly Rain Next Hour Domain Adaptation",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (destination / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle account username; never a token")
    parser.add_argument(
        "--package-only", action="store_true", help="Build package without uploading"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    package = build_package(args.root, args.root / "dist", args.username)
    if args.package_only:
        print(f"Kaggle package prepared at {package}")
        return
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(package),
            "--public",
            "--quiet",
            "--keep-tabular",
            "--dir-mode",
            "zip",
        ],
        check=True,
    )
    subprocess.run(
        ["kaggle", "datasets", "status", f"{args.username}/{DATASET_SLUG}"],
        check=True,
    )
    print(f"Dataset URL: https://www.kaggle.com/datasets/{args.username}/{DATASET_SLUG}")


if __name__ == "__main__":
    main()
