"""Phase 2 baselines and CNN-based SLARDA training for hourly rain data."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COUNT = 11
WINDOW_HOURS = 24
PRECIPITATION_INDEX = 8


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 3e-4
    patience: int = 3
    adaptation_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4
    pseudo_label_weight: float = 0.1
    confidence_threshold: float = 0.9
    teacher_momentum: float = 0.996
    adversarial_weight: float = 1.0
    num_threads: int = 4


class RainCNN(nn.Module):
    """Small 1D CNN that returns class logits and ordered temporal features."""

    def __init__(self, input_channels: int = FEATURE_COUNT, hidden_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        temporal = self.encoder(inputs)
        pooled = torch.cat((temporal.mean(dim=-1), temporal.amax(dim=-1)), dim=1)
        return self.classifier(pooled), (pooled, temporal)


class AutoregressiveDomainDiscriminator(nn.Module):
    """GRU discriminator over the CNN's temporal feature sequence."""

    def __init__(self, feature_channels: int = 64, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(feature_channels, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, channel_first_features: torch.Tensor) -> torch.Tensor:
        # CNN outputs [batch, channel, time]; the GRU consumes [batch, time, channel].
        sequence, _ = self.gru(channel_first_features.transpose(1, 2))
        return self.classifier(sequence[:, -1, :]).squeeze(-1)


def set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def experiment_source_sha256(repo_root: Path) -> str:
    files = [
        repo_root / "scripts/run_phase2.py",
        repo_root / "configs/rain_hanoi_haiphong.yaml",
        repo_root / "pyproject.toml",
        *sorted((repo_root / "src/slarda_rain").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_partition(path: Path, split: str) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    table = pq.read_table(path, filters=[("split", "=", split)])
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"No rows found for split {split!r} in {path}")
    values = np.asarray(frame["features"].tolist(), dtype=np.float32)
    if values.shape[1] != WINDOW_HOURS * FEATURE_COUNT:
        raise ValueError(f"Expected 24 x 11 features for {split}, received shape {values.shape}")
    values = values.reshape(-1, WINDOW_HOURS, FEATURE_COUNT)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite model features found in {split}")
    labels: np.ndarray | None = None
    if split != "target_adaptation_unlabeled":
        label_values = pd.to_numeric(frame["rain_next_hour"], errors="coerce")
        if label_values.isna().any():
            raise ValueError(f"Missing labels in supervised split {split}")
        labels = label_values.to_numpy(dtype=np.int64)
    elif frame["rain_next_hour"].notna().any():
        raise ValueError("Target adaptation labels must remain withheld")
    return values, labels, frame


def make_scaled_arrays(scaler: StandardScaler, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = scaler.transform(values.reshape(-1, FEATURE_COUNT)).reshape(values.shape)
    # Network inputs use [sample, feature, time]. Keep untouched original values for B0.
    return transformed.transpose(0, 2, 1).copy(), transformed.reshape(len(transformed), -1).copy()


def make_loader(
    inputs: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    x = torch.from_numpy(inputs.astype(np.float32, copy=False))
    tensors: tuple[torch.Tensor, ...]
    if labels is None:
        tensors = (x,)
    else:
        tensors = (x, torch.from_numpy(labels.astype(np.int64, copy=False)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        generator=generator,
        num_workers=0,
    )


def metric_bundle(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    roc_auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "rain_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "rain_recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "rain_precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": roc_auc,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [
        (metric_bundle(labels, probabilities, float(threshold))["macro_f1"], float(threshold))
        for threshold in candidates
    ]
    best_score = max(score for score, _ in scored)
    threshold = min(
        (threshold for score, threshold in scored if score == best_score),
        key=lambda t: abs(t - 0.5),
    )
    return threshold, metric_bundle(labels, probabilities, threshold)


def predict_probabilities(
    model: nn.Module, inputs: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    loader = make_loader(inputs, None, batch_size, False, 0)
    with torch.inference_mode():
        for (batch,) in loader:
            logits, _ = model(batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def train_supervised(
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed, config.num_threads)
    model = RainCNN().to(device)
    class_counts = np.bincount(train_y, minlength=2).astype(np.float32)
    weights = len(train_y) / (2.0 * np.maximum(class_counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = make_loader(train_x, train_y, config.batch_size, True, seed)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite B2 loss at seed {seed}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_probabilities = predict_probabilities(model, valid_x, config.batch_size, device)
        threshold, valid_metrics = select_threshold(valid_y, valid_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": valid_metrics["macro_f1"],
                "threshold": threshold,
            }
        )
        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"B2 did not produce a checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "seed": seed, "epoch": best_epoch, "threshold": best_threshold},
        checkpoint_path,
    )
    valid_probabilities = predict_probabilities(model, valid_x, config.batch_size, device)
    valid_metrics = metric_bundle(valid_y, valid_probabilities, best_threshold)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation_metrics": valid_metrics,
        "history": history,
    }


def train_slarda(
    seed: int,
    source_checkpoint: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    target_adapt_x: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed + 100_000, config.num_threads)
    source_checkpoint_data = torch.load(source_checkpoint, map_location=device, weights_only=True)
    source_model = RainCNN().to(device)
    source_model.load_state_dict(source_checkpoint_data["model_state"])
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)

    target_model = RainCNN().to(device)
    target_model.load_state_dict(source_checkpoint_data["model_state"])
    teacher_model = copy.deepcopy(target_model).to(device).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    discriminator = AutoregressiveDomainDiscriminator().to(device)
    domain_loss = nn.BCEWithLogitsLoss()
    pseudo_loss = nn.CrossEntropyLoss()
    optimizer_target = torch.optim.AdamW(
        target_model.parameters(),
        lr=config.adaptation_learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_discriminator = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.discriminator_learning_rate,
        weight_decay=config.weight_decay,
    )
    src_loader = make_loader(train_x, train_y, config.batch_size, True, seed + 1)
    tgt_loader = make_loader(target_adapt_x, None, config.batch_size, True, seed + 2)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        target_model.train()
        discriminator.train()
        total_losses: list[float] = []
        discriminator_losses: list[float] = []
        pseudo_counts: list[int] = []
        for (source_batch, source_labels), (target_batch,) in zip(src_loader, tgt_loader):
            source_batch = source_batch.to(device)
            source_labels = source_labels.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                _, (_, source_sequence) = source_model(source_batch)
            _, (_, target_sequence_detached) = target_model(target_batch)

            optimizer_discriminator.zero_grad(set_to_none=True)
            source_domain_logits = discriminator(source_sequence.detach())
            target_domain_logits = discriminator(target_sequence_detached.detach())
            domain_logits = torch.cat((source_domain_logits, target_domain_logits))
            domain_labels = torch.cat(
                (torch.ones_like(source_domain_logits), torch.zeros_like(target_domain_logits))
            )
            disc_loss = domain_loss(domain_logits, domain_labels)
            disc_loss.backward()
            optimizer_discriminator.step()

            optimizer_target.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            _, (_, target_sequence) = target_model(target_batch)
            target_source_logits = discriminator(target_sequence)
            target_as_source = torch.ones_like(target_source_logits)
            adversarial = domain_loss(target_source_logits, target_as_source)
            with torch.no_grad():
                teacher_logits, _ = teacher_model(target_batch)
                teacher_probabilities = torch.softmax(teacher_logits, dim=1)
                confidence, pseudo_labels = teacher_probabilities.max(dim=1)
                confident = confidence >= config.confidence_threshold
            if confident.any():
                target_logits, _ = target_model(target_batch[confident])
                pseudo = pseudo_loss(target_logits, pseudo_labels[confident])
                pseudo_counts.append(int(confident.sum().item()))
            else:
                pseudo = adversarial.new_zeros(())
                pseudo_counts.append(0)
            total_loss = (
                config.adversarial_weight * adversarial + config.pseudo_label_weight * pseudo
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite B3 loss at seed {seed}, epoch {epoch}")
            total_loss.backward()
            optimizer_target.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            with torch.no_grad():
                for teacher_parameter, target_parameter in zip(
                    teacher_model.parameters(), target_model.parameters()
                ):
                    teacher_parameter.mul_(config.teacher_momentum).add_(
                        target_parameter, alpha=1.0 - config.teacher_momentum
                    )
                for teacher_buffer, target_buffer in zip(
                    teacher_model.buffers(), target_model.buffers()
                ):
                    if teacher_buffer.is_floating_point():
                        teacher_buffer.mul_(config.teacher_momentum).add_(
                            target_buffer, alpha=1.0 - config.teacher_momentum
                        )
                    else:
                        teacher_buffer.copy_(target_buffer)
            total_losses.append(float(total_loss.detach().cpu()))
            discriminator_losses.append(float(disc_loss.detach().cpu()))

        valid_probabilities = predict_probabilities(
            target_model, valid_x, config.batch_size, device
        )
        threshold, valid_metrics = select_threshold(valid_y, valid_probabilities)
        history.append(
            {
                "epoch": epoch,
                "target_loss": float(np.mean(total_losses)),
                "discriminator_loss": float(np.mean(discriminator_losses)),
                "mean_confident_target_samples_per_batch": float(np.mean(pseudo_counts))
                if pseudo_counts
                else 0.0,
                "validation_macro_f1": valid_metrics["macro_f1"],
                "threshold": threshold,
            }
        )
        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_epoch = epoch
            best_threshold = threshold
            best_state = copy.deepcopy(target_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"B3 did not produce a checkpoint for seed {seed}")
    target_model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "seed": seed, "epoch": best_epoch, "threshold": best_threshold},
        checkpoint_path,
    )
    valid_probabilities = predict_probabilities(target_model, valid_x, config.batch_size, device)
    valid_metrics = metric_bundle(valid_y, valid_probabilities, best_threshold)
    return {
        "model": target_model,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation_metrics": valid_metrics,
        "history": history,
    }


def evaluate_target(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = predict_probabilities(model, test_x, batch_size, device)
    return metric_bundle(test_y, probabilities, threshold), probabilities


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def render_confusion_matrices(path: Path, matrices: dict[str, np.ndarray]) -> None:
    width, height = 1040, 310
    panel_width = 250
    colors = ("#edf2f7", "#90cdf4", "#4299e1", "#2b6cb0", "#1a365d")

    def svg_text(
        x: float,
        y: float,
        text: str,
        font_size: int,
        anchor: str = "middle",
        transform: str | None = None,
        bold: bool = False,
    ) -> str:
        transform_attr = f' transform="{transform}"' if transform else ""
        weight_attr = ' font-weight="bold"' if bold else ""
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}"'
            f'{transform_attr} font-family="sans-serif" font-size="{font_size}"'
            f"{weight_attr}>{html.escape(text)}</text>"
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(520, 30, "Target final test confusion matrices (2025)", 20, bold=True),
    ]
    for index, (model_name, matrix) in enumerate(matrices.items()):
        x0 = 20 + index * 255
        y0 = 60
        maximum = max(1, int(matrix.max()))
        elements.append(svg_text(x0 + panel_width / 2, y0, model_name, 16, bold=True))
        elements.append(svg_text(x0 + 112, y0 + 25, "Predicted class", 11))
        rotation = f"rotate(-90 {x0 + 12} {y0 + 104})"
        elements.append(svg_text(x0 + 12, y0 + 104, "Actual class", 11, transform=rotation))
        for row in range(2):
            for column in range(2):
                value = int(matrix[row, column])
                color_index = min(4, int(4 * value / maximum))
                cell_x, cell_y = x0 + 70 + column * 76, y0 + 45 + row * 72
                elements.append(
                    f'<rect x="{cell_x}" y="{cell_y}" width="72" height="68" '
                    f'fill="{colors[color_index]}" stroke="white"/>'
                )
                elements.append(svg_text(cell_x + 36, cell_y + 38, str(value), 17))
        for class_index, label in enumerate(("No rain (0)", "Rain (1)")):
            elements.append(svg_text(x0 + 57, y0 + 82 + class_index * 72, label, 10, anchor="end"))
            elements.append(svg_text(x0 + 107 + class_index * 76, y0 + 195, label, 10))
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = [
        "model",
        "seed",
        "partition",
        "threshold",
        "macro_f1",
        "rain_f1",
        "rain_recall",
        "rain_precision",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "n_samples",
        "positive_rate",
    ]
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        bundle = row["metrics"]
        tn, fp, fn, tp = np.asarray(bundle["confusion_matrix"]).ravel().tolist()
        flat_rows.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "partition": row["partition"],
                **{
                    key: bundle.get(key)
                    for key in columns
                    if key in bundle and key != "confusion_matrix"
                },
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    frame = pd.DataFrame(flat_rows)
    target_rows = frame[frame["partition"] == "target_final_test"]
    numeric = [
        "macro_f1",
        "rain_f1",
        "rain_recall",
        "rain_precision",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
    ]
    aggregate_rows = []
    for model_name, group in target_rows.groupby("model"):
        aggregate_rows.append(
            {
                "model": model_name,
                "seed": "mean",
                "partition": "target_final_test",
                **{metric: float(group[metric].mean()) for metric in numeric},
                "tn": None,
                "fp": None,
                "fn": None,
                "tp": None,
                "n_samples": int(group["n_samples"].iloc[0]),
                "positive_rate": float(group["positive_rate"].mean()),
            }
        )
        if len(group) > 1:
            aggregate_rows.append(
                {
                    "model": model_name,
                    "seed": "std",
                    "partition": "target_final_test",
                    **{metric: float(group[metric].std(ddof=1)) for metric in numeric},
                    "tn": None,
                    "fp": None,
                    "fn": None,
                    "tp": None,
                    "n_samples": None,
                    "positive_rate": None,
                }
            )
    summary = pd.concat((frame, pd.DataFrame(aggregate_rows)), ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.reindex(columns=columns).to_csv(output_path, index=False, float_format="%.6f")


def run_phase2(
    repo_root: Path,
    data_path: Path,
    metrics_dir: Path,
    checkpoints_dir: Path,
    seeds: list[int],
    config: TrainingConfig,
) -> dict[str, Any]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_train_raw, source_train_y, _ = load_partition(data_path, "source_train")
    source_valid_raw, source_valid_y, _ = load_partition(data_path, "source_validation")
    target_adapt_raw, _, _ = load_partition(data_path, "target_adaptation_unlabeled")
    assert source_train_y is not None and source_valid_y is not None

    scaler = StandardScaler()
    scaler.fit(source_train_raw.reshape(-1, FEATURE_COUNT))
    train_x, train_flat = make_scaled_arrays(scaler, source_train_raw)
    valid_x, valid_flat = make_scaled_arrays(scaler, source_valid_raw)
    adapt_x, _ = make_scaled_arrays(scaler, target_adapt_raw)
    np.savez_compressed(
        metrics_dir / "source_train_scaler.npz", mean=scaler.mean_, scale=scaler.scale_
    )

    majority = int(np.bincount(source_train_y, minlength=2).argmax())
    majority_validation = metric_bundle(
        source_valid_y, np.full(len(source_valid_y), float(majority)), 0.5
    )
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    logistic.fit(train_flat, source_train_y)
    b1_valid_probs = logistic.predict_proba(valid_flat)[:, 1]
    b1_threshold, b1_valid_metrics = select_threshold(source_valid_y, b1_valid_probs)
    if b1_valid_metrics["macro_f1"] <= majority_validation["macro_f1"]:
        raise RuntimeError(
            "B1 logistic regression failed the source-validation majority baseline gate: "
            f"B1={b1_valid_metrics['macro_f1']:.4f}, "
            f"majority={majority_validation['macro_f1']:.4f}. "
            "Inspect label construction and temporal splits before running target evaluation."
        )
    logistic_path = checkpoints_dir / "B1_logistic.joblib"
    logistic_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(logistic, logistic_path)

    run_metadata: dict[str, Any] = {
        "experiment": "Phase 2 B0-B3 hourly rain domain adaptation",
        "dataset": "Vietnam Hourly Rain Next Hour Domain Adaptation",
        "dataset_sha256": next(
            line.split()[0]
            for line in (repo_root / "data/processed/SHA256SUMS.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.endswith("windows_24h.parquet")
        ),
        "git_sha": git_sha(repo_root),
        "experiment_source_sha256": experiment_source_sha256(repo_root),
        "device": str(device),
        "feature_count": FEATURE_COUNT,
        "window_hours": WINDOW_HOURS,
        "num_classes": 2,
        "domain_ids": {"a": "hanoi", "b": "hai_phong"},
        "fit_scaler_on": "source_train_only",
        "config": asdict(config),
        "seeds": seeds,
        "target_test_access": (
            "deferred until checkpoints and thresholds were selected on source validation"
        ),
    }
    write_json(metrics_dir / "experiment_config.json", run_metadata)

    logistic_run = {
        "model": "B1_logistic",
        "seed": None,
        "partition": "source_validation",
        "metrics": b1_valid_metrics,
    }
    write_json(
        metrics_dir / "runs/B1_logistic.json",
        {
            **run_metadata,
            "model": "B1_logistic",
            "validation_threshold": b1_threshold,
            "validation_metrics": b1_valid_metrics,
            "majority_validation_macro_f1": majority_validation["macro_f1"],
            "checkpoint": str(logistic_path.relative_to(repo_root)),
        },
    )

    trained: dict[str, dict[int, dict[str, Any]]] = {"B2_cnn_source_only": {}, "B3_cnn_slarda": {}}
    for seed in seeds:
        b2_checkpoint = checkpoints_dir / f"B2_seed_{seed}.pt"
        b2_started = time.perf_counter()
        b2 = train_supervised(
            seed, train_x, source_train_y, valid_x, source_valid_y, config, b2_checkpoint, device
        )
        trained["B2_cnn_source_only"][seed] = b2
        b2_elapsed = time.perf_counter() - b2_started
        write_json(
            metrics_dir / f"runs/B2_seed_{seed}.json",
            {
                **run_metadata,
                "model": "B2_cnn_source_only",
                "seed": seed,
                "best_epoch": b2["best_epoch"],
                "training_seconds": b2_elapsed,
                "training_history": b2["history"],
                "validation_metrics": b2["validation_metrics"],
                "checkpoint": str(b2_checkpoint.relative_to(repo_root)),
            },
        )

        b3_checkpoint = checkpoints_dir / f"B3_seed_{seed}.pt"
        b3_started = time.perf_counter()
        b3 = train_slarda(
            seed,
            b2_checkpoint,
            train_x,
            source_train_y,
            valid_x,
            source_valid_y,
            adapt_x,
            config,
            b3_checkpoint,
            device,
        )
        trained["B3_cnn_slarda"][seed] = b3
        b3_elapsed = time.perf_counter() - b3_started
        write_json(
            metrics_dir / f"runs/B3_seed_{seed}.json",
            {
                **run_metadata,
                "model": "B3_cnn_slarda",
                "seed": seed,
                "best_epoch": b3["best_epoch"],
                "training_seconds": b3_elapsed,
                "training_history": b3["history"],
                "validation_metrics": b3["validation_metrics"],
                "checkpoint": str(b3_checkpoint.relative_to(repo_root)),
                "source_initialization_checkpoint": str(b2_checkpoint.relative_to(repo_root)),
                "target_adaptation_labels_used": False,
            },
        )

    # Target test labels are loaded only after all runs, checkpoints, and thresholds are fixed.
    target_test_raw, target_test_y, target_test_frame = load_partition(
        data_path, "target_final_test"
    )
    assert target_test_y is not None
    target_test_x, _ = make_scaled_arrays(scaler, target_test_raw)
    rows: list[dict[str, Any]] = [logistic_run]
    matrices: dict[str, np.ndarray] = {}

    # B0 persistence: predict that rain continues if it was raining in the last input hour.
    b0_predictions = (target_test_raw[:, -1, PRECIPITATION_INDEX] >= 0.1).astype(np.float32)
    b0_metrics = metric_bundle(target_test_y, b0_predictions, 0.5)
    rows.append(
        {
            "model": "B0_persistence",
            "seed": None,
            "partition": "target_final_test",
            "metrics": b0_metrics,
        }
    )
    matrices["B0 Persistence"] = np.asarray(b0_metrics["confusion_matrix"])
    write_json(
        metrics_dir / "runs/B0_persistence.json",
        {
            **run_metadata,
            "model": "B0_persistence",
            "partition": "target_final_test",
            "definition": (
                "rain next hour iff precipitation in the final observed input hour is >= 0.1 mm"
            ),
            "metrics": b0_metrics,
        },
    )

    logistic_test_probs = logistic.predict_proba(
        scaler.transform(target_test_raw.reshape(-1, FEATURE_COUNT))
        .reshape(target_test_raw.shape)
        .reshape(len(target_test_raw), -1)
    )[:, 1]
    b1_test_metrics = metric_bundle(target_test_y, logistic_test_probs, b1_threshold)
    rows.append(
        {
            "model": "B1_logistic",
            "seed": None,
            "partition": "target_final_test",
            "metrics": b1_test_metrics,
        }
    )
    matrices["B1 Logistic"] = np.asarray(b1_test_metrics["confusion_matrix"])
    write_json(
        metrics_dir / "runs/B1_logistic.json",
        {
            **run_metadata,
            "model": "B1_logistic",
            "validation_threshold": b1_threshold,
            "validation_metrics": b1_valid_metrics,
            "target_final_test_metrics": b1_test_metrics,
            "majority_validation_macro_f1": majority_validation["macro_f1"],
            "checkpoint": str(logistic_path.relative_to(repo_root)),
        },
    )

    for model_name, runs in trained.items():
        representative_matrix: np.ndarray | None = None
        representative_score = -1.0
        representative_seed = -1
        for seed, result in runs.items():
            target_metrics, probabilities = evaluate_target(
                result["model"],
                target_test_x,
                target_test_y,
                result["threshold"],
                config.batch_size,
                device,
            )
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "partition": "target_final_test",
                    "metrics": target_metrics,
                }
            )
            validation_score = float(result["validation_metrics"]["macro_f1"])
            if validation_score > representative_score:
                representative_score = validation_score
                representative_matrix = np.asarray(
                    target_metrics["confusion_matrix"], dtype=np.int64
                )
                representative_seed = seed
            run_path = (
                metrics_dir
                / f"runs/{'B2' if model_name.startswith('B2') else 'B3'}_seed_{seed}.json"
            )
            run_result = json.loads(run_path.read_text(encoding="utf-8"))
            run_result["target_final_test_metrics"] = target_metrics
            run_result["target_prediction_probability_sha256"] = hashlib.sha256(
                probabilities.astype("<f4").tobytes()
            ).hexdigest()
            write_json(run_path, run_result)
        label = "B2 CNN source-only" if model_name.startswith("B2") else "B3 CNN + SLARDA"
        matrices[f"{label} (seed {representative_seed})"] = representative_matrix

    write_summary(rows, metrics_dir / "baseline_summary.csv")
    render_confusion_matrices(metrics_dir / "confusion_matrices.svg", matrices)
    result = {
        **run_metadata,
        "target_final_test_split": "Hai Phong 2025 only",
        "target_test_samples": int(len(target_test_frame)),
        "majority_source_validation_macro_f1": majority_validation["macro_f1"],
        "B1_source_validation_macro_f1": b1_valid_metrics["macro_f1"],
        "summary_csv": str((metrics_dir / "baseline_summary.csv").relative_to(repo_root)),
        "confusion_matrices": str((metrics_dir / "confusion_matrices.svg").relative_to(repo_root)),
    }
    write_json(metrics_dir / "phase2_run.json", result)
    return result
