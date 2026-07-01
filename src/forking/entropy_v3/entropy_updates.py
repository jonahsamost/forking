from __future__ import annotations

import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from copy import deepcopy
from statistics import mean
from threading import Lock
from typing import Any, Iterable

from forking.entropy_v2.classifier import (
    DEFAULT_CLASSIFIER_HIDDEN_DIMS,
    DEFAULT_CLASSIFIER_FEATURE_MODE,
    EntropyClassifierRecord,
)
from forking.entropy_v2.classifier_manager import EntropyClassifierManager
from forking.entropy_v2.features import (
    vix_metadata_from_topk_logprobs,
    window_feature_rows_from_topk_logprobs,
)
from forking.entropy_v2.models import (
    DRAWUP_VALUE_KEY,
    DRIFT_VALUE_KEY,
    ENTROPY_XARGS_INTERVENE_KEY,
    INTERVENTION_IMPROVEMENT_RATE_KEY,
    INTERVENTIONS_IMPROVED_KEY,
    INTERVENTIONS_USED_KEY,
    REQUESTED_INTERVENE_KEY,
    SPLIT_INDICES_KEY,
    TOPK_KEY,
    TOPK_LOGPROBS_KEY,
    TOKEN_IDX_KEY,
    VIX_VALUE_KEY,
    VIX_METADATA_KEY,
)


def _module_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _module_logger(__name__)


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(mean(vals)) if vals else 0.0


def _rate(values: list[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


class EntropyUpdateTracker:
    def __init__(
        self,
        max_records: int,
        max_success_trigger_rate: float = 0.05,
        threshold_chunk_size: int = 64,
        classifier_update_interval: int = 64,
        classifier_min_success_records: int = 32,
        classifier_min_failure_records: int = 32,
        classifier_train_steps: int = 1000,
        classifier_learning_rate: float = 0.01,
        classifier_l2: float = 0.0,
        classifier_feature_mode: str = DEFAULT_CLASSIFIER_FEATURE_MODE,
        classifier_hidden_dims: list[int] | None = None,
        classifier_frontier_caps: list[float] | None = None,
        classifier_update_url: str | None = None,
        classifier_update_timeout_s: float = 10.0,
    ):
        self.threshold_chunk_size = threshold_chunk_size
        self.success_classifier_records: deque[EntropyClassifierRecord] = deque(maxlen=max_records)
        self.failure_classifier_records: deque[EntropyClassifierRecord] = deque(maxlen=max_records)
        self.update_executor = ThreadPoolExecutor(max_workers=1)
        self._metrics_lock = Lock()
        self.latest_classifier_metrics: dict[str, float] = {}
        self.update_jobs_submitted = 0
        self.update_jobs_completed = 0
        self.last_update_error: str | None = None
        self._cumulative_treatment_successes = 0
        self._cumulative_treatment_total = 0
        self._cumulative_control_successes = 0
        self._cumulative_control_total = 0
        self.classifier_manager = EntropyClassifierManager(
            chunk_size=threshold_chunk_size,
            update_interval=classifier_update_interval,
            min_success_records=classifier_min_success_records,
            min_failure_records=classifier_min_failure_records,
            train_steps=classifier_train_steps,
            learning_rate=classifier_learning_rate,
            l2=classifier_l2,
            feature_mode=classifier_feature_mode,
            hidden_dims=classifier_hidden_dims or list(DEFAULT_CLASSIFIER_HIDDEN_DIMS),
            max_success_trigger_rate=max_success_trigger_rate,
            frontier_caps=classifier_frontier_caps,
            update_url=classifier_update_url,
            update_timeout_s=classifier_update_timeout_s,
        )

    def request_xargs(self, sample_idx: int) -> dict[str, Any]:
        # intervene on every-other completion in a group
        intervene = sample_idx % 2 == 1 and self.classifier_manager.version > 0
        return {
            ENTROPY_XARGS_INTERVENE_KEY: int(intervene),
            "sample_idx": sample_idx,
        }

    def initial_metadata(self) -> dict[str, Any]:
        return {
            REQUESTED_INTERVENE_KEY: False,
            INTERVENTIONS_USED_KEY: 0,
            INTERVENTIONS_IMPROVED_KEY: 0,
            INTERVENTION_IMPROVEMENT_RATE_KEY: 0.0,
            SPLIT_INDICES_KEY: [],
            VIX_METADATA_KEY: [],
            TOPK_LOGPROBS_KEY: [],
        }

    @staticmethod
    def _topk_logprobs_from_metadata(metadata: dict[str, Any]) -> list[list[float]]:
        rows = metadata.get(TOPK_LOGPROBS_KEY) or []
        return [
            [float(value) for value in row]
            for row in rows
            if isinstance(row, list)
        ]

    def _vix_metadata_from_turn(self, turn_entropy: dict[str, Any]) -> list[dict[str, Any]]:
        windows = turn_entropy.get(VIX_METADATA_KEY) or []
        if windows:
            return [dict(window) for window in windows]

        topk_logprobs = self._topk_logprobs_from_metadata(turn_entropy)
        if not topk_logprobs:
            return []
        return [
            dict(window)
            for window in vix_metadata_from_topk_logprobs(
                topk_logprobs,
                self.threshold_chunk_size,
            )
        ]

    def merge_entropy_metadata(
        self,
        aggregate: dict[str, Any],
        turn_entropy: dict[str, Any],
        token_offset: int,
        requested_intervene: bool,
    ) -> None:
        aggregate[REQUESTED_INTERVENE_KEY] = bool(
            aggregate.get(REQUESTED_INTERVENE_KEY, False) or requested_intervene
        )
        aggregate[INTERVENTIONS_USED_KEY] = int(aggregate.get(INTERVENTIONS_USED_KEY, 0)) + int(
            turn_entropy.get(INTERVENTIONS_USED_KEY, 0) or 0
        )
        aggregate[INTERVENTIONS_IMPROVED_KEY] = int(aggregate.get(INTERVENTIONS_IMPROVED_KEY, 0)) + int(
            turn_entropy.get(INTERVENTIONS_IMPROVED_KEY, 0) or 0
        )
        aggregate[INTERVENTION_IMPROVEMENT_RATE_KEY] = (
            aggregate[INTERVENTIONS_IMPROVED_KEY] / aggregate[INTERVENTIONS_USED_KEY]
            if aggregate[INTERVENTIONS_USED_KEY]
            else 0.0
        )
        aggregate.setdefault(SPLIT_INDICES_KEY, []).extend(
            token_offset + int(idx) for idx in (turn_entropy.get(SPLIT_INDICES_KEY) or [])
        )
        aggregate.setdefault(TOPK_LOGPROBS_KEY, []).extend(
            self._topk_logprobs_from_metadata(turn_entropy)
        )
        if TOPK_KEY in turn_entropy:
            aggregate[TOPK_KEY] = turn_entropy[TOPK_KEY]
        aggregate.setdefault(VIX_METADATA_KEY, [])
        for window in self._vix_metadata_from_turn(turn_entropy):
            shifted = dict(window)
            shifted[TOKEN_IDX_KEY] = token_offset + int(shifted.get(TOKEN_IDX_KEY, 0))
            aggregate[VIX_METADATA_KEY].append(shifted)

    def log_generation_metadata(
        self,
        *,
        sample_idx: int,
        turn_entropy: dict[str, Any],
        requested_intervene: bool,
        token_offset: int,
        turn_len: int,
    ) -> None:
        windows = self._vix_metadata_from_turn(turn_entropy)
        vix_mean = _safe_mean(float(w.get(VIX_VALUE_KEY, 0.0)) for w in windows)
        drawup_mean = _safe_mean(float(w.get(DRAWUP_VALUE_KEY, 0.0)) for w in windows)
        drift_mean = _safe_mean(float(w.get(DRIFT_VALUE_KEY, 0.0)) for w in windows)
        interventions_used = int(turn_entropy.get(INTERVENTIONS_USED_KEY, 0) or 0)
        split_indices = turn_entropy.get(SPLIT_INDICES_KEY) or []

        if requested_intervene:
            return

    def update_from_scored_group(
        self,
        rewards: list[float],
        completion_lengths: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> dict[str, float]:
        group_metrics = self._group_metrics(rewards, completion_lengths, entropy_metadata)

        rewards_snapshot = [float(reward) for reward in rewards]
        completion_lengths_snapshot = [int(length) for length in completion_lengths]
        entropy_metadata_snapshot = deepcopy(entropy_metadata)

        with self._metrics_lock:
            self.update_jobs_submitted += 1
        future = self.update_executor.submit(
            self._process_scored_group,
            rewards_snapshot,
            completion_lengths_snapshot,
            entropy_metadata_snapshot,
        )
        future.add_done_callback(self._on_scored_group_processed)

        with self._metrics_lock:
            pending = self.update_jobs_submitted - self.update_jobs_completed
            metrics = dict(self.latest_classifier_metrics)
            metrics.update(group_metrics)
            metrics.update(
                {
                    "entropy/update_background_pending": float(pending),
                    "entropy/update_background_failed": float(self.last_update_error is not None),
                }
            )
            return metrics

    def _on_scored_group_processed(self, future: Future[None]) -> None:
        try:
            future.result()
        except Exception as error:
            logger.exception("Entropy scored-group background update failed")
            with self._metrics_lock:
                self.last_update_error = repr(error)
                self.update_jobs_completed += 1
            return

        with self._metrics_lock:
            self.last_update_error = None
            self.update_jobs_completed += 1

    def _process_scored_group(
        self,
        rewards: list[float],
        completion_lengths: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> None:
        self.classifier_manager.maybe_install_completed()

        records_added = 0
        for reward, completion_len, metadata in zip(rewards, completion_lengths, entropy_metadata, strict=True):
            if metadata.get(REQUESTED_INTERVENE_KEY, False):
                continue
            classifier_record = self._classifier_record_from_metadata(
                float(reward),
                completion_len,
                metadata,
            )
            if classifier_record is not None:
                self._add_classifier_record(classifier_record)
            records_added += 1
        if records_added:
            self.classifier_manager.maybe_enqueue_training(
                success_records=list(self.success_classifier_records),
                failure_records=list(self.failure_classifier_records),
            )

        self.classifier_manager.maybe_install_completed()

        with self._metrics_lock:
            self.latest_classifier_metrics = self.classifier_manager.metrics(
                success_records=list(self.success_classifier_records),
                failure_records=list(self.failure_classifier_records),
            )

    def _add_classifier_record(self, record: EntropyClassifierRecord) -> None:
        if record.success:
            self.success_classifier_records.append(record)
        else:
            self.failure_classifier_records.append(record)
        self.classifier_manager.note_record_added()

    def _classifier_record_from_metadata(
        self,
        reward: float,
        completion_len: int,
        metadata: dict[str, Any],
    ) -> EntropyClassifierRecord | None:
        topk_logprobs = self._topk_logprobs_from_metadata(metadata)
        if not topk_logprobs:
            return None
        features = window_feature_rows_from_topk_logprobs(
            topk_logprobs,
            self.threshold_chunk_size,
        )
        if not features:
            return None
        return EntropyClassifierRecord(
            reward=reward,
            completion_len=completion_len,
            features=features,
        )

    def _group_metrics(
        self,
        rewards: list[float],
        completion_lengths: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> dict[str, float]:
        requested = [bool(m.get(REQUESTED_INTERVENE_KEY, False)) for m in entropy_metadata]
        interventions = [float(m.get(INTERVENTIONS_USED_KEY, 0.0) or 0.0) for m in entropy_metadata]
        interventions_improved = [
            float(m.get(INTERVENTIONS_IMPROVED_KEY, 0.0) or 0.0) for m in entropy_metadata
        ]
        split_indices = [
            float(idx)
            for metadata in entropy_metadata
            for idx in (metadata.get(SPLIT_INDICES_KEY) or [])
        ]
        normalized_splits = [
            float(idx) / max(float(completion_len), 1.0)
            for metadata, completion_len in zip(entropy_metadata, completion_lengths, strict=True)
            for idx in (metadata.get(SPLIT_INDICES_KEY) or [])
        ]

        control_indices = [i for i, is_treatment in enumerate(requested) if not is_treatment]
        treatment_indices = [i for i, is_treatment in enumerate(requested) if is_treatment]
        intervened_indices = [i for i, count in enumerate(interventions) if count > 0]
        non_intervened_indices = [i for i, count in enumerate(interventions) if count == 0]
        total_interventions = sum(interventions)
        total_interventions_improved = sum(interventions_improved)
        treatment_interventions = sum(interventions[i] for i in treatment_indices)
        treatment_interventions_improved = sum(interventions_improved[i] for i in treatment_indices)

        metrics = {
            "entropy/intervention_rate": _safe_mean(requested),
            "entropy/avg_interventions": _safe_mean(interventions),
            "entropy/max_interventions": max(interventions) if interventions else 0.0,
            "entropy/avg_interventions_improved": _safe_mean(interventions_improved),
            "entropy/intervention_improvement_rate": (
                total_interventions_improved / total_interventions
                if total_interventions
                else 0.0
            ),
            "entropy/avg_split_index": _safe_mean(split_indices),
            "entropy/avg_normalized_split_index": _safe_mean(normalized_splits),
            "entropy/split_count": float(len(split_indices)),
            "entropy/control_success_rate": self._success_rate(rewards, control_indices),
            "entropy/treatment_success_rate": self._success_rate(rewards, treatment_indices),
            "entropy/intervened_success_rate": self._success_rate(rewards, intervened_indices),
            "entropy/non_intervened_success_rate": self._success_rate(rewards, non_intervened_indices),
            "entropy/intervened_reward_mean": self._reward_mean(rewards, intervened_indices),
            "entropy/non_intervened_reward_mean": self._reward_mean(rewards, non_intervened_indices),
            "entropy/control_reward_mean": self._reward_mean(rewards, control_indices),
            "entropy/treatment_reward_mean": self._reward_mean(rewards, treatment_indices),
            "entropy/treatment_interventions_used_mean": _safe_mean(interventions[i] for i in treatment_indices),
            "entropy/treatment_interventions_improved_mean": _safe_mean(
                interventions_improved[i] for i in treatment_indices
            ),
            "entropy/treatment_intervention_improvement_rate": (
                treatment_interventions_improved / treatment_interventions
                if treatment_interventions
                else 0.0
            ),
            "entropy/treatment_noop_rate": _rate([interventions[i] == 0 for i in treatment_indices]),
        }
        metrics["entropy/treatment_minus_control_success_rate"] = (
            metrics["entropy/treatment_success_rate"] - metrics["entropy/control_success_rate"]
        )

        if treatment_indices:
            treatment_successes = sum(1 for i in treatment_indices if float(rewards[i]) > 0)
            control_successes = sum(1 for i in control_indices if float(rewards[i]) > 0)
            self._cumulative_treatment_successes += treatment_successes
            self._cumulative_treatment_total += len(treatment_indices)
            self._cumulative_control_successes += control_successes
            self._cumulative_control_total += len(control_indices)
        cum_treatment_rate = (
            self._cumulative_treatment_successes / self._cumulative_treatment_total
            if self._cumulative_treatment_total else 0.0
        )
        cum_control_rate = (
            self._cumulative_control_successes / self._cumulative_control_total
            if self._cumulative_control_total else 0.0
        )
        metrics["entropy/cumulative_treatment_success_rate"] = cum_treatment_rate
        metrics["entropy/cumulative_control_success_rate"] = cum_control_rate
        metrics["entropy/cumulative_treatment_minus_control"] = cum_treatment_rate - cum_control_rate
        metrics["entropy/cumulative_treatment_total"] = float(self._cumulative_treatment_total)
        metrics["entropy/cumulative_control_total"] = float(self._cumulative_control_total)

        metrics.update(self._vix_group_metrics("control", control_indices, entropy_metadata))
        metrics.update(self._vix_group_metrics("treatment", treatment_indices, entropy_metadata))
        return metrics

    @staticmethod
    def _success_rate(rewards: list[float], indices: list[int]) -> float:
        return _rate([float(rewards[i]) > 0 for i in indices])

    @staticmethod
    def _reward_mean(rewards: list[float], indices: list[int]) -> float:
        return _safe_mean(float(rewards[i]) for i in indices)

    @staticmethod
    def _vix_group_metrics(
        prefix: str,
        indices: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> dict[str, float]:
        vix_values: list[float] = []
        drawup_values: list[float] = []
        drift_values: list[float] = []
        for i in indices:
            for window in entropy_metadata[i].get(VIX_METADATA_KEY) or []:
                vix_values.append(float(window.get(VIX_VALUE_KEY, 0.0)))
                drawup_values.append(float(window.get(DRAWUP_VALUE_KEY, 0.0)))
                drift_values.append(float(window.get(DRIFT_VALUE_KEY, 0.0)))
        return {
            f"entropy/{prefix}_vix_mean": _safe_mean(vix_values),
            f"entropy/{prefix}_drawup_mean": _safe_mean(drawup_values),
            f"entropy/{prefix}_drift_mean": _safe_mean(drift_values),
        }
