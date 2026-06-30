from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from forking.entropy_v2.classifier import (
    DEFAULT_CLASSIFIER_FEATURE_MODE,
    DEFAULT_CLASSIFIER_FRONTIER_CAPS,
    DEFAULT_CLASSIFIER_HIDDEN_DIMS,
    ClassifierParams,
    ClassifierTrainingResult,
    EntropyClassifierRecord,
    _train_classifier_snapshot,
)
from forking.entropy_v2.features import (
    COMBINED_FEATURE_NAMES,
    ENTROPY_WINDOW_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


class EntropyClassifierManager:
    def __init__(
        self,
        *,
        update_interval: int,
        min_success_records: int,
        min_failure_records: int,
        train_steps: int,
        learning_rate: float,
        l2: float,
        max_success_trigger_rate: float,
        feature_mode: str = DEFAULT_CLASSIFIER_FEATURE_MODE,
        hidden_dims: list[int] | None = None,
        frontier_caps: list[float] | None = None,
        update_url: str | None = None,
        update_timeout_s: float = 10.0,
    ):
        self.update_interval = update_interval
        self.min_success_records = min_success_records
        self.min_failure_records = min_failure_records
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.l2 = l2
        self.feature_mode = feature_mode
        self.hidden_dims = list(hidden_dims or DEFAULT_CLASSIFIER_HIDDEN_DIMS)
        self.max_success_trigger_rate = max_success_trigger_rate
        self.frontier_caps = list(frontier_caps or DEFAULT_CLASSIFIER_FRONTIER_CAPS)
        self.update_url = update_url
        self.update_timeout_s = update_timeout_s
        if self.max_success_trigger_rate not in self.frontier_caps:
            raise ValueError(
                "max_success_trigger_rate must be one of classifier_frontier_caps: "
                f"max_success_trigger_rate={self.max_success_trigger_rate}, "
                f"caps={self.frontier_caps}"
            )

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Future[ClassifierTrainingResult] | None = None
        self.params: ClassifierParams | None = None
        self.version = 0
        self.records_since_update = 0
        self.last_result: ClassifierTrainingResult | None = None
        self.last_error: str | None = None
        self.last_pushed_version = 0
        self.last_push_error: str | None = None

    def note_record_added(self) -> None:
        self.records_since_update += 1

    def maybe_enqueue_training(
        self,
        *,
        success_records: list[EntropyClassifierRecord],
        failure_records: list[EntropyClassifierRecord],
    ) -> None:
        if len(success_records) < self.min_success_records:
            return
        if len(failure_records) < self.min_failure_records:
            return
        if self.future is not None and not self.future.done():
            return
        if self.params is not None and self.records_since_update < self.update_interval:
            return

        self.future = self.executor.submit(
            _train_classifier_snapshot,
            success_records=list(success_records),
            failure_records=list(failure_records),
            version=self.version + 1,
            train_steps=self.train_steps,
            learning_rate=self.learning_rate,
            l2=self.l2,
            feature_mode=self.feature_mode,
            hidden_dims=list(self.hidden_dims),
            max_success_trigger_rate=self.max_success_trigger_rate,
            frontier_caps=list(self.frontier_caps),
        )

    def maybe_install_completed(self) -> None:
        if self.future is None:
            return
        if not self.future.done():
            return

        future = self.future
        self.future = None
        try:
            result = future.result()
        except Exception as error:
            self.last_error = repr(error)
            logger.exception("Entropy classifier training failed")
            return

        self.params = result.params
        self.version = result.params.version
        self.last_result = result
        self.last_error = None
        self.records_since_update = 0
        self._log_classifier_frontier(result)
        self._push_classifier_params(result.params)

    @staticmethod
    def _feature_names_for_mode(feature_mode: str) -> list[str]:
        if feature_mode == "entropy_window":
            return list(ENTROPY_WINDOW_FEATURE_NAMES)
        if feature_mode == "combined":
            return list(COMBINED_FEATURE_NAMES)
        return list(SUMMARY_FEATURE_NAMES)

    def _classifier_update_payload(self, params: ClassifierParams) -> dict[str, object]:
        payload = asdict(params)
        payload["feature_names"] = self._feature_names_for_mode(params.feature_mode)
        return payload

    def _push_classifier_params(self, params: ClassifierParams) -> None:
        if self.update_url is None:
            return

        body = json.dumps(self._classifier_update_payload(params)).encode("utf-8")
        request = Request(
            self.update_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.update_timeout_s) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            self.last_push_error = repr(error)
            logger.exception(
                "Entropy classifier push failed: version=%s url=%s",
                params.version,
                self.update_url,
            )
            return

        self.last_pushed_version = params.version
        self.last_push_error = None
        logger.info(
            "Entropy classifier pushed to server: version=%s url=%s",
            params.version,
            self.update_url,
        )

    def _log_classifier_frontier(self, result: ClassifierTrainingResult) -> None:
        parts = [
            (
                "cap<=%.3f threshold=%.6f success=%.4f failure=%.4f youden=%.4f"
                % (
                    point.cap,
                    point.threshold,
                    point.success_trigger_rate,
                    point.failure_trigger_rate,
                    point.youden_j,
                )
            )
            for point in result.frontier
        ]
        logger.info(
            "Entropy classifier trained: version=%s loss=%.6f elapsed_s=%.3f "
            "records_success=%s records_failure=%s windows_success=%s windows_failure=%s "
            "threshold=%.6f frontier=%s",
            result.params.version,
            result.train_loss,
            result.train_elapsed_s,
            result.train_records_success,
            result.train_records_failure,
            result.train_windows_success,
            result.train_windows_failure,
            result.params.threshold,
            " | ".join(parts),
        )

    def metrics(
        self,
        *,
        success_records: list[EntropyClassifierRecord],
        failure_records: list[EntropyClassifierRecord],
    ) -> dict[str, float]:
        result = self.last_result
        params = self.params
        metrics = {
            "entropy/classifier_ready": float(params is not None),
            "entropy/classifier_training_running": float(
                self.future is not None and not self.future.done()
            ),
            "entropy/classifier_version": float(self.version),
            "entropy/classifier_records_since_update": float(self.records_since_update),
            "entropy/classifier_pushed_version": float(self.last_pushed_version),
            "entropy/classifier_push_failed": float(self.last_push_error is not None),
            "entropy/classifier_record_success_count": float(len(success_records)),
            "entropy/classifier_record_failure_count": float(len(failure_records)),
            "entropy/classifier_record_window_success_count": float(
                sum(len(record.features) for record in success_records)
            ),
            "entropy/classifier_record_window_failure_count": float(
                sum(len(record.features) for record in failure_records)
            ),
            "entropy/classifier_train_failed": float(self.last_error is not None),
        }

        if params is not None:
            metrics.update(
                {
                    "entropy/classifier_threshold": params.threshold,
                    "entropy/classifier_max_success_trigger_rate": params.max_success_trigger_rate,
                }
            )

        if result is not None:
            metrics.update(
                {
                    "entropy/classifier_loss": result.train_loss,
                    "entropy/classifier_train_elapsed_s": result.train_elapsed_s,
                    "entropy/classifier_train_ms_per_step": (
                        result.train_elapsed_s * 1000.0 / max(self.train_steps, 1)
                    ),
                    "entropy/classifier_train_records_success": float(result.train_records_success),
                    "entropy/classifier_train_records_failure": float(result.train_records_failure),
                    "entropy/classifier_train_windows_success": float(result.train_windows_success),
                    "entropy/classifier_train_windows_failure": float(result.train_windows_failure),
                }
            )
            for point in result.frontier:
                cap_key = str(point.cap).replace(".", "_")
                metrics[f"entropy/classifier_frontier_{cap_key}_success"] = point.success_trigger_rate
                metrics[f"entropy/classifier_frontier_{cap_key}_failure"] = point.failure_trigger_rate
                metrics[f"entropy/classifier_frontier_{cap_key}_threshold"] = point.threshold

        return metrics
