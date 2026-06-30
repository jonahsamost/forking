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
import time
import uuid

from forking.entropy_v2.features import (
    compute_entropy_trajectory,
    compute_rolling_vix,
    sampled_vix_metadata,
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
        self.lookahead = 4 * self.chunk_size

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

    def _vix_instability_score(self, v: VixValues, xargs: EntropyXargs) -> float:
        tau_vix = getattr(xargs, "tau_vix", None)
        tau_drift = getattr(xargs, "tau_drift", None)
        tau_drawup = getattr(xargs, "tau_drawup", None)
        score = 0.0
        if tau_vix is not None:
            vix_excess = max(0.0, v.vix - tau_vix)
            if vix_excess > 0.0:
                direction_excess = 0.0
                if tau_drift is not None:
                    direction_excess = max(direction_excess, v.drift - tau_drift)
                if tau_drawup is not None:
                    direction_excess = max(direction_excess, v.drawup - tau_drawup)
                if direction_excess > 0.0:
                    score = max(score, vix_excess + direction_excess)
        if tau_drawup is not None:
            score = max(score, v.drawup - tau_drawup)
        return max(0.0, score)

    def _find_split_idx(
        self,
        vix_values: list[VixValues],
        xargs: EntropyXargs,
    ) -> int | None:
        best_local_idx = None
        best_score = 0.0
        for local_idx, vix_value in enumerate(vix_values):
            score = self._vix_instability_score(vix_value, xargs)
            if score > best_score:
                best_score = score
                best_local_idx = local_idx
        return best_local_idx

    def _branch_score(
        self, branch_vix_values: list[VixValues],
    ) -> float:
        if not branch_vix_values:
            return 0.0
        return (
            sum(v.vix for v in branch_vix_values) / len(branch_vix_values)
            + sum(max(v.drift, 0.0) for v in branch_vix_values) / len(branch_vix_values)
            + max(v.drawup for v in branch_vix_values)
        )

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
        xargs = EntropyXargs.from_dict(request.vllm_xargs)

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

            prior_entropies = all_entropies[-(self.chunk_size - 1):]
            local_entropies = prior_entropies + chunk_entropies
            local_vix = self._rolling_vix(local_entropies)
            chunk_vix = local_vix[len(prior_entropies):]
            assert len(chunk_vix) == len(chunk_entropies), (
                "Misaligned chunk VIX: "
                f"chunk_vix={len(chunk_vix)}, "
                f"chunk_entropies={len(chunk_entropies)}"
            )

            detected_idx = self._find_split_idx(
                chunk_vix,
                xargs=xargs,
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
            original_tail_score = self._branch_score(chunk_vix[split_idx:])
    
            best_branch = None
            best_branch_logprobs = None
            best_branch_entropies = None
            best_branch_score = float("inf")

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
                branch_prior_entropies = (all_entropies + keep_entropies)[
                    -(self.chunk_size - 1):
                ]
                branch_local_entropies = branch_prior_entropies + branch_entropies
                branch_local_vix = self._rolling_vix(branch_local_entropies)
                branch_vix = branch_local_vix[len(branch_prior_entropies):]
                assert len(branch_vix) == len(branch_entropies), (
                    "Misaligned branch VIX: "
                    f"branch_vix={len(branch_vix)}, "
                    f"branch_entropies={len(branch_entropies)}"
                )
                score = self._branch_score(branch_vix)
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
