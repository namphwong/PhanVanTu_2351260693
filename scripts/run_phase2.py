"""Run Phase 2 baselines and 3-seed CNN/SLARDA evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from slarda_rain.phase2 import TrainingConfig, run_phase2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/rain_hanoi_haiphong.yaml"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "artifacts/metrics")
    parser.add_argument("--checkpoints-dir", type=Path, default=REPO_ROOT / "artifacts/checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config_document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = config_document["model"]
    if model_config.get("input_channels") != 11 or model_config.get("num_classes") != 2:
        raise ValueError("Phase 2 expects 11 feature channels and two rain classes")
    domain_ids = config_document["data"].get("domain_ids", {})
    if domain_ids != {"a": "hanoi", "b": "hai_phong"}:
        raise ValueError("Phase 2 domain IDs must be a=Hanoi source and b=Hai Phong target")
    phase2_values = config_document.get("phase2", {})
    valid_fields = {field.name for field in fields(TrainingConfig)}
    config = TrainingConfig(
        **{key: value for key, value in phase2_values.items() if key in valid_fields}
    )
    if args.epochs is not None:
        config = TrainingConfig(**{**config.__dict__, "epochs": args.epochs})
    if args.batch_size is not None:
        config = TrainingConfig(**{**config.__dict__, "batch_size": args.batch_size})
    seeds = args.seeds or config_document["reproducibility"]["seeds"]
    data_path = (
        args.data or REPO_ROOT / config_document["data"]["processed_dir"] / "windows_24h.parquet"
    )
    result = run_phase2(
        repo_root=REPO_ROOT,
        data_path=data_path.resolve(),
        metrics_dir=(REPO_ROOT / args.metrics_dir).resolve()
        if not args.metrics_dir.is_absolute()
        else args.metrics_dir,
        checkpoints_dir=(REPO_ROOT / args.checkpoints_dir).resolve()
        if not args.checkpoints_dir.is_absolute()
        else args.checkpoints_dir,
        seeds=seeds,
        config=config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
