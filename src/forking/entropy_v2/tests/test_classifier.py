from __future__ import annotations

import math
import time

from forking.entropy_v2.classifier import (
    DEFAULT_CLASSIFIER_FRONTIER_CAPS,
    DEFAULT_CLASSIFIER_HIDDEN_DIMS,
    EntropyClassifierRecord,
    _train_classifier_snapshot,
)
from forking.entropy_v2.features import EntropyWindowFeatures


RECORDS_PER_CLASS = 2048
COMPLETION_LEN = 1024
CHUNK_SIZE = 64
WINDOWS_PER_COMPLETION = COMPLETION_LEN // CHUNK_SIZE
TRAIN_STEPS = 1000
LEARNING_RATE = 0.01
L2 = 0.0
MAX_SUCCESS_TRIGGER_RATE = 0.10
FEATURE_MODE = "combined"


def _make_window(
    *,
    record_idx: int,
    window_idx: int,
    failure: bool,
) -> EntropyWindowFeatures:
    phase = record_idx * 0.017 + window_idx * 0.13
    base = 0.25 + (0.18 if failure else 0.0) + window_idx * 0.002
    entropy_window = [
        base
        + 0.08 * math.sin(phase + token_idx * 0.19)
        + (0.05 * token_idx / max(CHUNK_SIZE - 1, 1) if failure else 0.0)
        for token_idx in range(CHUNK_SIZE)
    ]
    entropy = entropy_window[-1]
    deltas = [0.0] + [
        entropy_window[i] - entropy_window[i - 1]
        for i in range(1, len(entropy_window))
    ]
    vix = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
    drift = sum(deltas) / len(deltas)
    up_vix = math.sqrt(sum(max(delta, 0.0) ** 2 for delta in deltas) / len(deltas))
    down_vix = math.sqrt(sum(min(delta, 0.0) ** 2 for delta in deltas) / len(deltas))
    drawup = entropy - min(entropy_window)
    drawdown = max(entropy_window) - entropy

    return EntropyWindowFeatures(
        entropy_window=entropy_window,
        entropy=entropy,
        vix=vix,
        drift=drift,
        up_vix=up_vix,
        down_vix=down_vix,
        drawup=drawup,
        drawdown=drawdown,
        token_idx_norm=(window_idx + 1) / WINDOWS_PER_COMPLETION,
        vix_max_so_far=vix + 0.001 * window_idx,
        drawup_max_so_far=drawup + 0.001 * window_idx,
        entropy_max_so_far=max(entropy_window),
        entropy_min_so_far=min(entropy_window),
    )


def _make_record(record_idx: int, *, failure: bool) -> EntropyClassifierRecord:
    return EntropyClassifierRecord(
        reward=0.0 if failure else 1.0,
        completion_len=COMPLETION_LEN,
        features=[
            _make_window(record_idx=record_idx, window_idx=window_idx, failure=failure)
            for window_idx in range(WINDOWS_PER_COMPLETION)
        ],
    )


def test_train_classifier_snapshot_full_default_scale() -> None:
    success_records = [
        _make_record(record_idx=idx, failure=False)
        for idx in range(RECORDS_PER_CLASS)
    ]
    failure_records = [
        _make_record(record_idx=idx, failure=True)
        for idx in range(RECORDS_PER_CLASS)
    ]

    start = time.perf_counter()
    result = _train_classifier_snapshot(
        success_records=success_records,
        failure_records=failure_records,
        version=1,
        train_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        l2=L2,
        feature_mode=FEATURE_MODE,
        hidden_dims=list(DEFAULT_CLASSIFIER_HIDDEN_DIMS),
        max_success_trigger_rate=MAX_SUCCESS_TRIGGER_RATE,
        frontier_caps=list(DEFAULT_CLASSIFIER_FRONTIER_CAPS),
    )
    elapsed_s = time.perf_counter() - start

    assert result.params.version == 1
    assert result.params.feature_mode == FEATURE_MODE
    assert result.params.input_dim == 76
    assert result.params.hidden_dims == DEFAULT_CLASSIFIER_HIDDEN_DIMS
    assert len(result.params.feature_mean) == result.params.input_dim
    assert len(result.params.feature_std) == result.params.input_dim
    assert result.train_records_success == RECORDS_PER_CLASS
    assert result.train_records_failure == RECORDS_PER_CLASS
    assert result.train_windows_success == RECORDS_PER_CLASS * WINDOWS_PER_COMPLETION
    assert result.train_windows_failure == RECORDS_PER_CLASS * WINDOWS_PER_COMPLETION
    assert result.frontier
    assert math.isfinite(result.train_loss)
    assert math.isfinite(result.params.threshold)
    assert "net.0.weight" in result.params.state_dict

    print(
        "full_scale_classifier_train "
        f"elapsed_s={elapsed_s:.3f} "
        f"reported_elapsed_s={result.train_elapsed_s:.3f} "
        f"loss={result.train_loss:.6f} "
        f"threshold={result.params.threshold:.6f}"
    )
