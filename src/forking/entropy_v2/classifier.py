from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from forking.entropy_v2.features import EntropyWindowFeatures


DEFAULT_CLASSIFIER_FRONTIER_CAPS = [0.01, 0.05, 0.075, 0.10, 0.125]
DEFAULT_CLASSIFIER_FEATURE_MODE = "entropy_window"
DEFAULT_CLASSIFIER_HIDDEN_DIMS = [128, 64, 32]
DEFAULT_CLASSIFIER_ACTIVATION = "gelu"
CLASSIFIER_FEATURE_MODES = ("entropy_window", "combined")


@dataclass(frozen=True)
class EntropyClassifierRecord:
    reward: float
    completion_len: int
    features: list[EntropyWindowFeatures]

    @property
    def success(self) -> bool:
        return self.reward > 0


@dataclass(frozen=True)
class ClassifierParams:
    version: int
    feature_mode: str
    input_dim: int
    hidden_dims: list[int]
    activation: str
    state_dict: dict[str, Any]
    feature_mean: list[float]
    feature_std: list[float]
    threshold: float
    max_success_trigger_rate: float


@dataclass(frozen=True)
class ClassifierFrontierPoint:
    cap: float
    threshold: float
    success_trigger_rate: float
    failure_trigger_rate: float
    youden_j: float


@dataclass(frozen=True)
class ClassifierTrainingResult:
    params: ClassifierParams
    frontier: list[ClassifierFrontierPoint]
    train_loss: float
    train_elapsed_s: float
    train_windows_success: int
    train_windows_failure: int
    train_records_success: int
    train_records_failure: int


class EntropyFailureClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or list(DEFAULT_CLASSIFIER_HIDDEN_DIMS)
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _expand_classifier_examples(
    success_records: list[EntropyClassifierRecord],
    failure_records: list[EntropyClassifierRecord],
    *,
    feature_mode: str,
) -> tuple[list[list[float]], list[float], list[float], int, int]:
    if feature_mode not in CLASSIFIER_FEATURE_MODES:
        raise ValueError(f"Unsupported classifier feature mode: {feature_mode}")
    X: list[list[float]] = []
    y: list[float] = []
    sample_weight: list[float] = []

    def add_records(
        records: list[EntropyClassifierRecord],
        label: float,
        class_weight: float,
    ) -> int:
        window_count = 0
        if not records:
            return 0

        completion_weight = class_weight / len(records)
        for record in records:
            if not record.features:
                continue

            weight = completion_weight / len(record.features)
            for window in record.features:
                if feature_mode == "entropy_window":
                    X.append(window.entropy_window_vector())
                elif feature_mode == "combined":
                    X.append(window.combined_vector())
                else:
                    raise ValueError(f"Unsupported classifier feature mode: {feature_mode}")
                y.append(label)
                sample_weight.append(weight)
                window_count += 1
        return window_count

    success_windows = add_records(success_records, label=0.0, class_weight=0.5)
    failure_windows = add_records(failure_records, label=1.0, class_weight=0.5)
    return X, y, sample_weight, success_windows, failure_windows


def _weighted_feature_stats(
    X: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = sample_weight / sample_weight.sum().clamp_min(1e-12)
    mean = (X * weight[:, None]).sum(dim=0)
    centered = X - mean
    var = ((centered * centered) * weight[:, None]).sum(dim=0)
    std = torch.sqrt(var.clamp_min(1e-12))
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def _fit_entropy_classifier(
    X_rows: list[list[float]],
    y_rows: list[float],
    weight_rows: list[float],
    *,
    train_steps: int,
    learning_rate: float,
    l2: float,
    hidden_dims: list[int],
) -> tuple[EntropyFailureClassifier, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    if not X_rows:
        raise ValueError("Cannot train classifier without feature rows")

    X = torch.tensor(X_rows, dtype=torch.float32, device="cpu")
    y = torch.tensor(y_rows, dtype=torch.float32, device="cpu")
    sample_weight = torch.tensor(weight_rows, dtype=torch.float32, device="cpu")
    sample_weight = sample_weight / sample_weight.sum().clamp_min(1e-12)

    feature_mean, feature_std = _weighted_feature_stats(X, sample_weight)
    X_norm = (X - feature_mean) / feature_std
    input_dim = int(X_norm.shape[1])

    model = EntropyFailureClassifier(input_dim=input_dim, hidden_dims=hidden_dims).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    final_loss = 0.0
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_norm)
        per_example_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        )
        loss = (per_example_loss * sample_weight).sum()
        if l2 > 0.0:
            l2_penalty = sum((parameter ** 2).sum() for parameter in model.parameters())
            loss = loss + l2 * l2_penalty
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_norm)).detach().cpu()

    return model, feature_mean.detach().cpu(), feature_std.detach().cpu(), final_loss, probabilities


def _record_scores_from_probabilities(
    success_records: list[EntropyClassifierRecord],
    failure_records: list[EntropyClassifierRecord],
    probabilities: torch.Tensor,
) -> tuple[list[float], list[float]]:
    success_scores: list[float] = []
    failure_scores: list[float] = []
    offset = 0

    for record in success_records:
        n = len(record.features)
        if n == 0:
            continue
        record_probs = probabilities[offset : offset + n]
        success_scores.append(float(record_probs.max().item()))
        offset += n

    for record in failure_records:
        n = len(record.features)
        if n == 0:
            continue
        record_probs = probabilities[offset : offset + n]
        failure_scores.append(float(record_probs.max().item()))
        offset += n

    return success_scores, failure_scores


def _trigger_rate_from_scores(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    return sum(score >= threshold for score in scores) / len(scores)


def _select_classifier_threshold(
    success_scores: list[float],
    failure_scores: list[float],
    *,
    max_success_trigger_rate: float,
    frontier_caps: list[float],
) -> tuple[float, list[ClassifierFrontierPoint]]:
    candidates = sorted(set(success_scores + failure_scores), reverse=True)
    if not candidates:
        return 1.0, []

    candidates = [max(candidates) + 1e-6, *candidates]

    frontier: list[ClassifierFrontierPoint] = []
    selected_threshold: float | None = None
    fallback_threshold = candidates[0]
    fallback_key: tuple[float, float] | None = None

    for cap in frontier_caps:
        best_for_cap: ClassifierFrontierPoint | None = None
        for threshold in candidates:
            success_rate = _trigger_rate_from_scores(success_scores, threshold)
            failure_rate = _trigger_rate_from_scores(failure_scores, threshold)
            point = ClassifierFrontierPoint(
                cap=cap,
                threshold=threshold,
                success_trigger_rate=success_rate,
                failure_trigger_rate=failure_rate,
                youden_j=failure_rate - success_rate,
            )

            fallback_candidate_key = (-success_rate, failure_rate)
            if fallback_key is None or fallback_candidate_key > fallback_key:
                fallback_key = fallback_candidate_key
                fallback_threshold = threshold

            if success_rate > cap:
                continue
            if best_for_cap is None or (
                failure_rate,
                -success_rate,
                threshold,
            ) > (
                best_for_cap.failure_trigger_rate,
                -best_for_cap.success_trigger_rate,
                best_for_cap.threshold,
            ):
                best_for_cap = point

        if best_for_cap is not None:
            frontier.append(best_for_cap)
            if cap == max_success_trigger_rate:
                selected_threshold = best_for_cap.threshold

    return selected_threshold if selected_threshold is not None else fallback_threshold, frontier


def _state_dict_to_lists(model: nn.Module) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().tolist()
        for name, tensor in model.state_dict().items()
    }


def _train_classifier_snapshot(
    *,
    success_records: list[EntropyClassifierRecord],
    failure_records: list[EntropyClassifierRecord],
    version: int,
    train_steps: int,
    learning_rate: float,
    l2: float,
    feature_mode: str,
    hidden_dims: list[int],
    max_success_trigger_rate: float,
    frontier_caps: list[float],
) -> ClassifierTrainingResult:
    start = time.perf_counter()
    X_rows, y_rows, weight_rows, success_windows, failure_windows = _expand_classifier_examples(
        success_records,
        failure_records,
        feature_mode=feature_mode,
    )
    model, feature_mean, feature_std, train_loss, probabilities = _fit_entropy_classifier(
        X_rows,
        y_rows,
        weight_rows,
        train_steps=train_steps,
        learning_rate=learning_rate,
        l2=l2,
        hidden_dims=hidden_dims,
    )
    input_dim = len(X_rows[0])

    success_scores, failure_scores = _record_scores_from_probabilities(
        success_records,
        failure_records,
        probabilities,
    )
    threshold, frontier = _select_classifier_threshold(
        success_scores,
        failure_scores,
        max_success_trigger_rate=max_success_trigger_rate,
        frontier_caps=frontier_caps,
    )

    params = ClassifierParams(
        version=version,
        feature_mode=feature_mode,
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        activation=DEFAULT_CLASSIFIER_ACTIVATION,
        state_dict=_state_dict_to_lists(model),
        feature_mean=[float(value) for value in feature_mean.tolist()],
        feature_std=[float(value) for value in feature_std.tolist()],
        threshold=float(threshold),
        max_success_trigger_rate=float(max_success_trigger_rate),
    )

    return ClassifierTrainingResult(
        params=params,
        frontier=frontier,
        train_loss=train_loss,
        train_elapsed_s=time.perf_counter() - start,
        train_windows_success=success_windows,
        train_windows_failure=failure_windows,
        train_records_success=len(success_records),
        train_records_failure=len(failure_records),
    )
