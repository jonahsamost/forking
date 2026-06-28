from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from forking.entropy.models import (
    DRAWUP_VALUE_KEY,
    DRIFT_VALUE_KEY,
    ENTROPY_XARGS_INTERVENE_KEY,
    INTERVENTIONS_USED_KEY,
    REQUESTED_INTERVENE_KEY,
    SPLIT_INDICES_KEY,
    TOKEN_IDX_KEY,
    VIX_VALUE_KEY,
    VIX_METADATA_KEY,
)


logger = logging.getLogger(__name__)


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(mean(vals)) if vals else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _rate(values: list[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


@dataclass(frozen=True)
class VixThresholds:
    tau_vix: float
    tau_drift: float
    tau_drawup: float

    def as_vllm_xargs(self, intervene: bool) -> dict[str, float | int]:
        return {
            ENTROPY_XARGS_INTERVENE_KEY: int(intervene),
            "tau_vix": self.tau_vix,
            "tau_drift": self.tau_drift,
            "tau_drawup": self.tau_drawup,
        }


@dataclass(frozen=True)
class VixCalibrationRecord:
    reward: float
    completion_len: int
    vix_max: float
    drift_max: float
    drawup_max: float

    @property
    def success(self) -> bool:
        return self.reward > 0


class VixThresholdCalibrator:
    def __init__(
        self,
        max_records: int,
        bootstrap_records: int,
        update_interval: int,
        calibration_ema: float,
        min_successes: int = 8,
        min_failures: int = 8,
    ):
        self.success_records: deque[VixCalibrationRecord] = deque(maxlen=max_records)
        self.failure_records: deque[VixCalibrationRecord] = deque(maxlen=max_records)
        self.bootstrap_records = bootstrap_records
        self.update_interval = update_interval
        self.calibration_ema = calibration_ema
        self.min_successes = min_successes
        self.min_failures = min_failures
        self.thresholds: VixThresholds | None = None
        self.new_records_since_update = 0
        self.update_count = 0
        self.last_fit_method = "none"
        self.last_youden_j = 0.0
        self.last_trigger_rate_success = 0.0
        self.last_trigger_rate_failure = 0.0
        self._logged_min_successes = False
        self._logged_min_failures = False
        self._logged_labeled_calibration_ready = False

    @property
    def ready(self) -> bool:
        return self.thresholds is not None

    def add_completion(self, record: VixCalibrationRecord) -> None:
        if record.success:
            self.success_records.append(record)
        else:
            self.failure_records.append(record)
        self.new_records_since_update += 1
        self._log_buffer_milestones()

    def _log_buffer_milestones(self) -> None:
        success_count = len(self.success_records)
        failure_count = len(self.failure_records)
        if not self._logged_min_successes and success_count >= self.min_successes:
            logger.info(
                "VIX calibration has enough successful controls: "
                f"{success_count}/{self.min_successes}"
            )
            self._logged_min_successes = True
        if not self._logged_min_failures and failure_count >= self.min_failures:
            logger.info(
                "VIX calibration has enough failed controls: "
                f"{failure_count}/{self.min_failures}"
            )
            self._logged_min_failures = True
        if (
            not self._logged_labeled_calibration_ready
            and success_count >= self.min_successes
            and failure_count >= self.min_failures
        ):
            logger.info(
                "VIX labeled calibration is ready: "
                f"successes={success_count}, failures={failure_count}"
            )
            self._logged_labeled_calibration_ready = True

    def maybe_update(self) -> VixThresholds | None:
        if self._num_records < self.bootstrap_records:
            return self.thresholds
        if self.thresholds is not None and self.new_records_since_update < self.update_interval:
            return self.thresholds

        fitted = self._fit_thresholds()
        if fitted is None:
            self.last_fit_method = "skipped"
            return self.thresholds
        if self.thresholds is None:
            self.thresholds = fitted
        else:
            ema = self.calibration_ema
            self.thresholds = VixThresholds(
                tau_vix=ema * self.thresholds.tau_vix + (1.0 - ema) * fitted.tau_vix,
                tau_drift=ema * self.thresholds.tau_drift + (1.0 - ema) * fitted.tau_drift,
                tau_drawup=ema * self.thresholds.tau_drawup + (1.0 - ema) * fitted.tau_drawup,
            )
        self.new_records_since_update = 0
        self.update_count += 1
        self._update_trigger_metrics(self.thresholds)
        return self.thresholds

    @property
    def _num_records(self) -> int:
        return len(self.success_records) + len(self.failure_records)

    def _records(self) -> list[VixCalibrationRecord]:
        return [*self.success_records, *self.failure_records]

    def _fit_thresholds(self) -> VixThresholds | None:
        records = self._records()
        successes = list(self.success_records)
        failures = list(self.failure_records)
        if len(successes) >= self.min_successes and len(failures) >= self.min_failures:
            self.last_fit_method = "separation"
            return self._fit_by_labeled_separation(records, successes, failures)
        if len(successes) < self.min_successes:
            self.last_fit_method = "skipped"
            return None
        self.last_fit_method = "success_quantile"
        return VixThresholds(
            tau_vix=_quantile([r.vix_max for r in successes], 0.95),
            tau_drift=_quantile([r.drift_max for r in successes], 0.95),
            tau_drawup=_quantile([r.drawup_max for r in successes], 0.95),
        )

    def _fit_by_labeled_separation(
        self,
        records: list[VixCalibrationRecord],
        successes: list[VixCalibrationRecord],
        failures: list[VixCalibrationRecord],
    ) -> VixThresholds:
        quantiles = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        vix_candidates = sorted({ _quantile([r.vix_max for r in records], q) for q in quantiles })
        drift_candidates = sorted({ _quantile([r.drift_max for r in records], q) for q in quantiles })
        drawup_candidates = sorted({ _quantile([r.drawup_max for r in records], q) for q in quantiles })

        best = VixThresholds(vix_candidates[0], drift_candidates[0], drawup_candidates[0])
        best_score = float("-inf")
        best_success_rate = 0.0
        best_failure_rate = 0.0
        for tau_vix in vix_candidates:
            for tau_drift in drift_candidates:
                for tau_drawup in drawup_candidates:
                    thresholds = VixThresholds(tau_vix, tau_drift, tau_drawup)
                    success_rate = self._trigger_rate(successes, thresholds)
                    failure_rate = self._trigger_rate(failures, thresholds)
                    score = failure_rate - success_rate
                    if score > best_score:
                        best_score = score
                        best = thresholds
                        best_success_rate = success_rate
                        best_failure_rate = failure_rate
        self.last_youden_j = best_score
        self.last_trigger_rate_success = best_success_rate
        self.last_trigger_rate_failure = best_failure_rate
        return best

    def _update_trigger_metrics(self, thresholds: VixThresholds) -> None:
        successes = list(self.success_records)
        failures = list(self.failure_records)
        self.last_trigger_rate_success = self._trigger_rate(successes, thresholds)
        self.last_trigger_rate_failure = self._trigger_rate(failures, thresholds)
        self.last_youden_j = self.last_trigger_rate_failure - self.last_trigger_rate_success

    @staticmethod
    def _trigger(record: VixCalibrationRecord, thresholds: VixThresholds) -> bool:
        '''
        1. Entropy rebounds upward from a local low (i.e. confidence -> uncertainty)
        2. Entropy is locally volatile and trending upward
        '''
        return (
            record.drawup_max > thresholds.tau_drawup
            or (
                record.vix_max > thresholds.tau_vix
                and record.drift_max > thresholds.tau_drift
            )
        )

    def _trigger_rate(self, records: list[VixCalibrationRecord], thresholds: VixThresholds) -> float:
        return _rate([self._trigger(record, thresholds) for record in records])

    def metrics(self) -> dict[str, float]:
        successes = list(self.success_records)
        failures = list(self.failure_records)
        metrics = {
            "entropy/calibration_ready": float(self.ready),
            "entropy/calibration_buffer_size": float(self._num_records),
            "entropy/calibration_success_count": float(len(successes)),
            "entropy/calibration_failure_count": float(len(failures)),
            "entropy/calibration_updates": float(self.update_count),
            "entropy/calibration_trigger_rate_success": self.last_trigger_rate_success,
            "entropy/calibration_trigger_rate_failure": self.last_trigger_rate_failure,
            "entropy/calibration_youden_j": self.last_youden_j,
            "entropy/calibration_fit_separation": float(self.last_fit_method == "separation"),
            "entropy/calibration_fit_success_quantile": float(self.last_fit_method == "success_quantile"),
            "entropy/calibration_fit_skipped": float(self.last_fit_method == "skipped"),
            "entropy/control_success_vix_max_mean": _safe_mean(r.vix_max for r in successes),
            "entropy/control_failure_vix_max_mean": _safe_mean(r.vix_max for r in failures),
            "entropy/control_success_drawup_max_mean": _safe_mean(r.drawup_max for r in successes),
            "entropy/control_failure_drawup_max_mean": _safe_mean(r.drawup_max for r in failures),
            "entropy/control_success_drift_max_mean": _safe_mean(r.drift_max for r in successes),
            "entropy/control_failure_drift_max_mean": _safe_mean(r.drift_max for r in failures),
        }
        metrics["entropy/control_vix_gap"] = (
            metrics["entropy/control_failure_vix_max_mean"] - metrics["entropy/control_success_vix_max_mean"]
        )
        metrics["entropy/control_drawup_gap"] = (
            metrics["entropy/control_failure_drawup_max_mean"] - metrics["entropy/control_success_drawup_max_mean"]
        )
        metrics["entropy/control_drift_gap"] = (
            metrics["entropy/control_failure_drift_max_mean"] - metrics["entropy/control_success_drift_max_mean"]
        )
        if self.thresholds is not None:
            metrics.update(
                {
                    "entropy/threshold_vix": self.thresholds.tau_vix,
                    "entropy/threshold_drift": self.thresholds.tau_drift,
                    "entropy/threshold_drawup": self.thresholds.tau_drawup,
                }
            )
        return metrics


class EntropyUpdateTracker:
    def __init__(
        self,
        bootstrap_records: int,
        max_records: int | None = None,
        update_interval: int = 32,
        calibration_ema: float = 0.9,
    ):
        self.calibrator = VixThresholdCalibrator(
            max_records=max_records or max(bootstrap_records * 8, 1024),
            bootstrap_records=bootstrap_records,
            update_interval=update_interval,
            calibration_ema=calibration_ema,
        )

    def request_xargs(self, sample_idx: int) -> dict[str, Any]:
        # intervene on every-other completion in a group
        intervene = self.calibrator.ready and (sample_idx % 2 == 1)
        if not intervene or self.calibrator.thresholds is None:
            return {ENTROPY_XARGS_INTERVENE_KEY: 0, "sample_idx": sample_idx}
        return {
            **self.calibrator.thresholds.as_vllm_xargs(intervene=True),
            "sample_idx": sample_idx,
        }

    def initial_metadata(self) -> dict[str, Any]:
        return {
            REQUESTED_INTERVENE_KEY: False,
            INTERVENTIONS_USED_KEY: 0,
            SPLIT_INDICES_KEY: [],
            VIX_METADATA_KEY: [],
        }

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
        aggregate.setdefault(SPLIT_INDICES_KEY, []).extend(
            token_offset + int(idx) for idx in (turn_entropy.get(SPLIT_INDICES_KEY) or [])
        )
        aggregate.setdefault(VIX_METADATA_KEY, [])
        for window in turn_entropy.get(VIX_METADATA_KEY) or []:
            shifted = dict(window)
            shifted[TOKEN_IDX_KEY] = token_offset + int(shifted.get(TOKEN_IDX_KEY, 0))
            aggregate[VIX_METADATA_KEY].append(shifted)

    def update_from_scored_group(
        self,
        rewards: list[float],
        completion_lengths: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> dict[str, float]:
        records_added = 0
        for reward, completion_len, metadata in zip(rewards, completion_lengths, entropy_metadata, strict=True):
            if metadata.get(REQUESTED_INTERVENE_KEY, False):
                continue
            record = self._record_from_metadata(float(reward), completion_len, metadata)
            if record is None:
                continue
            self.calibrator.add_completion(record)
            records_added += 1
        if records_added:
            self.calibrator.maybe_update()

        metrics = self.calibrator.metrics()
        metrics.update(self._group_metrics(rewards, completion_lengths, entropy_metadata))
        return metrics

    @staticmethod
    def _record_from_metadata(
        reward: float,
        completion_len: int,
        metadata: dict[str, Any],
    ) -> VixCalibrationRecord | None:
        windows = metadata.get(VIX_METADATA_KEY) or []
        if not windows:
            return None
        return VixCalibrationRecord(
            reward=reward,
            completion_len=completion_len,
            vix_max=max(float(w.get(VIX_VALUE_KEY, 0.0)) for w in windows),
            drift_max=max(float(w.get(DRIFT_VALUE_KEY, 0.0)) for w in windows),
            drawup_max=max(float(w.get(DRAWUP_VALUE_KEY, 0.0)) for w in windows),
        )

    def _group_metrics(
        self,
        rewards: list[float],
        completion_lengths: list[int],
        entropy_metadata: list[dict[str, Any]],
    ) -> dict[str, float]:
        requested = [bool(m.get(REQUESTED_INTERVENE_KEY, False)) for m in entropy_metadata]
        interventions = [float(m.get(INTERVENTIONS_USED_KEY, 0.0) or 0.0) for m in entropy_metadata]
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

        metrics = {
            "entropy/intervention_rate": _safe_mean(requested),
            "entropy/avg_interventions": _safe_mean(interventions),
            "entropy/max_interventions": max(interventions) if interventions else 0.0,
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
            "entropy/treatment_noop_rate": _rate([interventions[i] == 0 for i in treatment_indices]),
        }
        metrics["entropy/treatment_minus_control_success_rate"] = (
            metrics["entropy/treatment_success_rate"] - metrics["entropy/control_success_rate"]
        )

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
