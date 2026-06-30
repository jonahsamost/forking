from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionLogProbs,
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.renderers.inputs.preprocess import (
    extract_prompt_len,
)
from vllm.inputs import tokens_input
from vllm.entrypoints.utils import get_max_tokens
from vllm.outputs import CompletionOutput

from fastapi import Request
from copy import deepcopy
from dataclasses import fields
from threading import Lock
import time
import uuid
from typing import Any

import torch

from forking.entropy_v2.classifier import (
    DEFAULT_CLASSIFIER_ACTIVATION,
    ClassifierParams,
    EntropyFailureClassifier,
)
from forking.entropy_v2.features import (
    EntropyWindowFeatures,
    compute_entropy_trajectory,
    compute_rolling_vix,
    sampled_vix_metadata,
    window_feature_rows_from_entropy_trajectory,
)
from forking.entropy_v2.models import (
    INTERVENED_KEY,
    INTERVENTION_IMPROVEMENT_RATE_KEY,
    INTERVENTIONS_IMPROVED_KEY,
    INTERVENTIONS_USED_KEY,
    SPLIT_INDICES_KEY,
    TOPK_KEY,
    TOPK_LOGPROBS_KEY,
    VIX_METADATA_KEY,
    EntropyXargs,
    VixValues,
)

MASK_64_BITS = (1 << 64) - 1

def _random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


class EntropyHandler:
    def __init__(
        self,
        engine: AsyncLLM,
        chunk_size: int = 64,
        topk_entropy: int = 3,
        max_interventions: int = 3,
        num_samples: int = 3,
        classifier_inference_stride: int = 16,
    ):
        self.engine = engine
        self.model_config = engine.model_config
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.chunk_size = chunk_size
        self.topk_entropy = topk_entropy
        self.max_interventions = max_interventions
        self.num_samples = num_samples
        self.classifier_inference_stride = classifier_inference_stride
        self.lookahead = 4 * self.chunk_size
        self.classifier_params: ClassifierParams | None = None
        self.classifier_model: EntropyFailureClassifier | None = None
        self.classifier_feature_names: list[str] = []
        self.classifier_version = 0
        self._classifier_lock = Lock()

    def update_classifier(self, payload: dict[str, Any]) -> dict[str, int | str]:
        required_keys = {field.name for field in fields(ClassifierParams)} | {
            "feature_names"
        }
        missing_keys = sorted(required_keys - set(payload))
        if missing_keys:
            raise ValueError(f"Missing classifier payload keys: {missing_keys}")
        if not isinstance(payload["state_dict"], dict):
            raise ValueError("Classifier state_dict must be an object")

        params = ClassifierParams(
            version=int(payload["version"]),
            feature_mode=str(payload["feature_mode"]),
            input_dim=int(payload["input_dim"]),
            hidden_dims=[int(value) for value in payload["hidden_dims"]],
            activation=str(payload["activation"]),
            state_dict=deepcopy(payload["state_dict"]),
            feature_mean=[float(value) for value in payload["feature_mean"]],
            feature_std=[float(value) for value in payload["feature_std"]],
            threshold=float(payload["threshold"]),
            max_success_trigger_rate=float(payload["max_success_trigger_rate"]),
        )
        input_dim = params.input_dim
        feature_mean = params.feature_mean
        feature_std = params.feature_std
        feature_names = [str(value) for value in payload["feature_names"]]
        if len(feature_mean) != input_dim:
            raise ValueError(
                "Classifier feature_mean length does not match input_dim: "
                f"len={len(feature_mean)} input_dim={input_dim}"
            )
        if len(feature_std) != input_dim:
            raise ValueError(
                "Classifier feature_std length does not match input_dim: "
                f"len={len(feature_std)} input_dim={input_dim}"
            )
        if len(feature_names) != input_dim:
            raise ValueError(
                "Classifier feature_names length does not match input_dim: "
                f"len={len(feature_names)} input_dim={input_dim}"
            )
        if params.activation != DEFAULT_CLASSIFIER_ACTIVATION:
            raise ValueError(f"Unsupported classifier activation: {params.activation}")

        model = EntropyFailureClassifier(
            input_dim=params.input_dim,
            hidden_dims=params.hidden_dims,
        ).cpu()
        try:
            model.load_state_dict(
                {
                    key: torch.tensor(value, dtype=torch.float32, device="cpu")
                    for key, value in params.state_dict.items()
                },
                strict=True,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("Invalid classifier state_dict") from error
        model.eval()

        with self._classifier_lock:
            self.classifier_params = params
            self.classifier_model = model
            self.classifier_feature_names = feature_names
            self.classifier_version = params.version

        return {"status": "ok", "version": params.version}

    def _calculate_entropy_trajectory(
        self, output: CompletionOutput, 
    ) -> list[float]:
        return compute_entropy_trajectory(self._topk_logprob_trace(output))

    def _topk_logprob_trace(self, output: CompletionOutput) -> list[list[float]]:
        return [
            sorted(
                (float(lp.logprob) for lp in pos_logprobs.values()),
                reverse=True,
            )[:self.topk_entropy]
            if pos_logprobs
            else []
            for pos_logprobs in (output.logprobs or [])
        ]
    
    def _rolling_vix(self, token_entropy: list[float]) -> list[VixValues]:
        return compute_rolling_vix(token_entropy, self.chunk_size)

    async def _generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        request_id: str,
        n: int = 1,
    ) -> list[CompletionOutput]:
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.topk_entropy,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        final_result = None
        async for result in self.engine.generate(
            tokens_input(prompt_token_ids=prompt_ids),
            sampling_params,
            request_id,
        ):
            final_result = result
        if final_result is None or not final_result.outputs:
            raise RuntimeError("vLLM generation returned no output")
        return final_result.outputs

    def _sampled_token_logprobs(self, output: CompletionOutput) -> list[float]:
        assert output.logprobs is not None, "vLLM did not return token logprobs"
        assert len(output.token_ids) == len(output.logprobs), (
            "Misaligned token IDs and logprobs: "
            f"token_ids={len(output.token_ids)}, logprobs={len(output.logprobs)}"
        )
        token_logprobs: list[float] = []
        for token_idx, (token_id, pos) in enumerate(zip(output.token_ids, output.logprobs, strict=True)):
            assert pos is not None and token_id in pos, (
                "Sampled token missing from vLLM logprobs: "
                f"token_idx={token_idx}, token_id={token_id}"
            )
            token_logprobs.append(pos[token_id].logprob)
        return token_logprobs

    def _assert_aligned(
        self,
        token_ids: list[int],
        token_logprobs: list[float],
        entropies: list[float],
    ) -> None:
        assert len(token_ids) == len(token_logprobs) == len(entropies), (
            "Misaligned completion arrays: "
            f"token_ids={len(token_ids)}, "
            f"token_logprobs={len(token_logprobs)}, "
            f"entropies={len(entropies)}"
        )

    def _current_classifier_snapshot(
        self,
    ) -> tuple[ClassifierParams, EntropyFailureClassifier] | None:
        with self._classifier_lock:
            if self.classifier_params is None or self.classifier_model is None:
                return None
            return self.classifier_params, self.classifier_model

    @staticmethod
    def _classifier_vector(
        features: EntropyWindowFeatures,
        params: ClassifierParams,
    ) -> list[float]:
        if params.feature_mode == "entropy_window":
            return features.entropy_window_vector()
        if params.feature_mode == "combined":
            return features.combined_vector()
        raise ValueError(f"Unsupported classifier feature mode: {params.feature_mode}")

    def _classifier_scores_for_generated_tokens(
        self,
        local_entropies: list[float],
        *,
        generated_start_idx: int,
        params: ClassifierParams,
        model: EntropyFailureClassifier,
    ) -> list[tuple[int, float]]:
        start_idx = max(generated_start_idx, self.chunk_size - 1)
        rows = window_feature_rows_from_entropy_trajectory(
            local_entropies,
            self.chunk_size,
            window_stride=self.classifier_inference_stride,
            start_idx=start_idx,
        )
        if not rows:
            return []

        vectors = [self._classifier_vector(features, params) for features in rows]
        for vector in vectors:
            if len(vector) != params.input_dim:
                raise ValueError(
                    "Classifier feature dimension mismatch: "
                    f"features={len(vector)} input_dim={params.input_dim}"
                )
        X = torch.tensor(vectors, dtype=torch.float32, device="cpu")
        mean = torch.tensor(params.feature_mean, dtype=torch.float32, device="cpu")
        std = torch.tensor(params.feature_std, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            probabilities = torch.sigmoid(model((X - mean) / std)).detach().cpu().tolist()

        return [
            (
                start_idx
                + row_offset * self.classifier_inference_stride
                - generated_start_idx,
                float(probability),
            )
            for row_offset, probability in enumerate(probabilities)
        ]

    def _find_split_idx(
        self,
        local_entropies: list[float],
        *,
        generated_start_idx: int,
        params: ClassifierParams,
        model: EntropyFailureClassifier,
    ) -> int | None:
        best_local_idx = None
        best_probability = params.threshold
        for local_idx, probability in self._classifier_scores_for_generated_tokens(
            local_entropies,
            generated_start_idx=generated_start_idx,
            params=params,
            model=model,
        ):
            if probability >= best_probability:
                best_probability = probability
                best_local_idx = local_idx
        return best_local_idx

    def _branch_score(
        self,
        local_entropies: list[float],
        *,
        generated_start_idx: int,
        params: ClassifierParams,
        model: EntropyFailureClassifier,
    ) -> float:
        scores = self._classifier_scores_for_generated_tokens(
            local_entropies,
            generated_start_idx=generated_start_idx,
            params=params,
            model=model,
        )
        if not scores:
            return 0.0
        return max(probability for _, probability in scores)

    @staticmethod
    def _is_terminal_finish_reason(finish_reason: str | None) -> bool:
        # Internal lookahead/branch calls use bounded token budgets, so "length"
        # only means the local chunk budget was consumed. True stops should still
        # terminate the full completion.
        return finish_reason is not None and finish_reason != "length"
    
    async def non_intervention_handler(
        self, request: CompletionRequest, raw_request: Request
    ) -> CompletionResponse:
        request_id = _random_uuid()
        created_time = int(time.time())
        max_model_len = self.engine.model_config.max_model_len
        prompt_ids = list(request.prompt)
        engine_input = tokens_input(prompt_token_ids=prompt_ids)

        max_tokens = get_max_tokens(
            max_model_len,
            request.max_tokens,
            extract_prompt_len(self.model_config, engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
        )
        outputs = await self._generate(
            prompt_ids,
            max_tokens,
            request.temperature or 1.0,
            request_id,
        )
        output = outputs[0]
        token_logprobs = self._sampled_token_logprobs(output)
        topk_logprobs = self._topk_logprob_trace(output)
        
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
            CompletionResponseChoice(
                index=0,
                text="",  # trl ingests tokens, not text
                token_ids=output.token_ids,
                logprobs=CompletionLogProbs(
                    token_logprobs=token_logprobs,
                    tokens=[],
                    text_offset=[],
                    top_logprobs=[],
                ),
                finish_reason=output.finish_reason,
            )         
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(output.token_ids),
                total_tokens=len(prompt_ids) + len(output.token_ids),
            ),
            entropy={
                TOPK_LOGPROBS_KEY: topk_logprobs,
                TOPK_KEY: self.topk_entropy,
            }
        )
    
    async def intervention_handler(
        self, request: CompletionRequest, raw_request: Request
    ) -> CompletionResponse:
        request_id = _random_uuid()
        created_time = int(time.time())
        max_model_len = self.engine.model_config.max_model_len

        prompt_ids = list(request.prompt)
        engine_input = tokens_input(prompt_token_ids=prompt_ids)
        classifier_snapshot = self._current_classifier_snapshot()
        if classifier_snapshot is None:
            return await self.non_intervention_handler(request, raw_request)
        params, model = classifier_snapshot

        max_tokens = get_max_tokens(
            max_model_len,
            request.max_tokens,
            extract_prompt_len(self.model_config, engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
        )
        temperature = request.temperature or 1.0

        all_token_ids: list[int] = []
        all_token_logprobs: list[float] = []
        all_entropies: list[float] = []
        interventions_used = 0
        interventions_improved = 0
        split_indices: list[int] = []
        finish_reason = "length"

        while len(all_token_ids) < max_tokens:
            remaining = max_tokens - len(all_token_ids)

            if interventions_used >= self.max_interventions:
                output = (await self._generate(
                    prompt_ids + all_token_ids,
                    remaining,
                    temperature,
                    f"cmpl-{_random_uuid()}",
                ))[0]
                all_token_ids.extend(output.token_ids)
                all_token_logprobs.extend(self._sampled_token_logprobs(output))
                all_entropies.extend(self._calculate_entropy_trajectory(output))
                self._assert_aligned(all_token_ids, all_token_logprobs, all_entropies)
                finish_reason = output.finish_reason or "length"
                break
            
            output = (await self._generate(
                prompt_ids + all_token_ids,
                min(self.lookahead, remaining),
                temperature,
                f"cmpl-{_random_uuid()}",
            ))[0]

            chunk_ids = list(output.token_ids)
            chunk_logprobs = self._sampled_token_logprobs(output)
            chunk_entropies = self._calculate_entropy_trajectory(output)
            self._assert_aligned(chunk_ids, chunk_logprobs, chunk_entropies)

            local_entropies = all_entropies + chunk_entropies

            detected_idx = self._find_split_idx(
                local_entropies,
                generated_start_idx=len(all_entropies),
                params=params,
                model=model,
            )

            if detected_idx is None:
                all_token_ids.extend(chunk_ids)
                all_token_logprobs.extend(chunk_logprobs)
                all_entropies.extend(chunk_entropies)
                self._assert_aligned(all_token_ids, all_token_logprobs, all_entropies)
                if self._is_terminal_finish_reason(output.finish_reason):
                    finish_reason = output.finish_reason
                    break
                continue

            split_idx = max(0, detected_idx - self.chunk_size + 1)
            split_indices.append(len(all_token_ids) + split_idx)
            keep_ids = chunk_ids[:split_idx]
            keep_logprobs = chunk_logprobs[:split_idx]
            keep_entropies = chunk_entropies[:split_idx]
            self._assert_aligned(keep_ids, keep_logprobs, keep_entropies)
            branch_prompt_ids = prompt_ids + all_token_ids + keep_ids
            branch_max_tokens = max(1, len(chunk_ids) - split_idx)
            original_tail_entropies = all_entropies + chunk_entropies
            original_tail_score = self._branch_score(
                original_tail_entropies,
                generated_start_idx=len(all_entropies) + split_idx,
                params=params,
                model=model,
            )
    
            best_branch = None
            best_branch_logprobs = None
            best_branch_entropies = None
            best_branch_score = original_tail_score

            branch_outputs = await self._generate(
                branch_prompt_ids,
                branch_max_tokens,
                temperature,
                f"cmpl-{_random_uuid()}",
                n=self.num_samples,
            )

            for branch_output in branch_outputs:
                branch_entropies = self._calculate_entropy_trajectory(branch_output)
                branch_logprobs = self._sampled_token_logprobs(branch_output)
                self._assert_aligned(
                    list(branch_output.token_ids), branch_logprobs, branch_entropies
                )
                branch_local_entropies = all_entropies + keep_entropies + branch_entropies
                score = self._branch_score(
                    branch_local_entropies,
                    generated_start_idx=len(all_entropies) + len(keep_entropies),
                    params=params,
                    model=model,
                )
                if score < best_branch_score:
                    best_branch_score = score
                    best_branch = branch_output
                    best_branch_logprobs = branch_logprobs
                    best_branch_entropies = branch_entropies
            
            if best_branch is None:
                # keep original block if all branches are worse
                all_token_ids.extend(chunk_ids)
                all_token_logprobs.extend(chunk_logprobs)
                all_entropies.extend(chunk_entropies)
                self._assert_aligned(all_token_ids, all_token_logprobs, all_entropies)
                if self._is_terminal_finish_reason(output.finish_reason):
                    finish_reason = output.finish_reason
                    break
                continue

            all_token_ids.extend(keep_ids)
            all_token_logprobs.extend(keep_logprobs)
            all_entropies.extend(keep_entropies)
            all_token_ids.extend(best_branch.token_ids)
            all_token_logprobs.extend(best_branch_logprobs)
            all_entropies.extend(best_branch_entropies)
            self._assert_aligned(all_token_ids, all_token_logprobs, all_entropies)
            interventions_used += 1
            if best_branch_score < original_tail_score:
                interventions_improved += 1
            if self._is_terminal_finish_reason(best_branch.finish_reason):
                finish_reason = best_branch.finish_reason
                break

        final_vix = self._rolling_vix(all_entropies)
        sampled_vix = sampled_vix_metadata(all_entropies, final_vix, self.chunk_size)
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="",
                    token_ids=all_token_ids,
                    logprobs=CompletionLogProbs(
                        token_logprobs=all_token_logprobs,
                        tokens=[],
                        text_offset=[],
                        top_logprobs=[],
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(all_token_ids),
                total_tokens=len(prompt_ids) + len(all_token_ids),
            ),
            entropy={
                INTERVENED_KEY: interventions_used > 0,
                INTERVENTIONS_USED_KEY: interventions_used,
                INTERVENTIONS_IMPROVED_KEY: interventions_improved,
                INTERVENTION_IMPROVEMENT_RATE_KEY: (
                    interventions_improved / interventions_used
                    if interventions_used
                    else 0.0
                ),
                SPLIT_INDICES_KEY: split_indices,
                VIX_METADATA_KEY: sampled_vix,
            },
        )

    async def handler(
        self, request: CompletionRequest, raw_request: Request
    ) -> CompletionResponse:
        xargs = EntropyXargs.from_dict(request.vllm_xargs)
        if xargs.intervene:
            return await self.intervention_handler(request, raw_request)
        else:
            return await self.non_intervention_handler(request, raw_request)
