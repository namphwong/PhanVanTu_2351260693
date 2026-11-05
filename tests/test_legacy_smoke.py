"""Synthetic smoke tests for the original SLARDA data and CNN interfaces."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _legacy_import_paths() -> None:
    legacy_root = str((Path(__file__).parents[1] / "Autorgressive_Adaptation").resolve())
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    # args.py parses argv at import time; isolate imports from pytest's flags.
    sys.argv[:] = ["legacy-smoke"]


def test_legacy_dataloader_loads_synthetic_domain(tmp_path):
    _legacy_import_paths()
    loader_module = importlib.import_module("dataloader.dataloader")
    for split in ("train", "val", "test"):
        torch.save(
            {"samples": torch.randn(4, 24, 3), "labels": torch.tensor([0, 1, 0, 1])},
            tmp_path / f"{split}_a.pt",
        )

    configs = SimpleNamespace(apply_transform=False, batch_size=2)
    train_loader, valid_loader, test_loader = loader_module.data_generator(
        str(tmp_path), "a", configs
    )
    x, y = next(iter(train_loader))
    assert x.shape == (2, 3, 24)
    assert y.shape == (2,)
    assert len(valid_loader) > 0
    assert len(test_loader) > 0


def test_cnn_opp_har_sl_forward_smoke():
    _legacy_import_paths()
    models_module = importlib.import_module("models.models")
    configs = SimpleNamespace(
        input_channels=3,
        out_channels=16,
        feat_dim=16,
        kernel_size=8,
        dropout=0.1,
        num_classes=2,
    )
    model = models_module.CNN_Opp_HAR_SL(configs).eval()
    with torch.no_grad():
        logits, (vec_features, seq_features) = model(torch.randn(2, 3, 64))
    assert logits.shape == (2, 2)
    assert vec_features.shape == (2, 16)
    assert seq_features.shape[0:2] == (2, 16)
