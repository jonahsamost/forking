from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionLogProbs,
    CompletionResponse,
    CompletionResponseChoice,
    ErrorResponse,
)
from vllm.entrypoints.openai.engine.protocol import (
    UsageInfo,
)
from vllm.sampling_params import SamplingParams
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.inputs import tokens_input
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

from fastapi import Request
import time
import uuid

MASK_64_BITS = (1 << 64) - 1

def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


class EntropyHandler:
    def __init__(
        self,
        engine: AsyncLLM,
        chunk_size: int = 64,
        threshold_burst: float = 1.36,
        threshold_rebound: float = 1.33,
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
        self.threshold_burst = threshold_burst
        self.threshold_rebound = threshold_rebound

    async def handler(
        self, request: CompletionRequest, raw_request: Request
    ) -> CompletionResponse:
        request_id = random_uuid()
        created_time = int(time.time())
        max_model_len = self.engine.model_config.max_model_len
        engine_input = tokens_input(prompt_token_ids=request.prompt)

        max_tokens = get_max_tokens(
            max_model_len,
            request.max_tokens,
            extract_prompt_len(self.model_config, engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )

        all_token_ids: list[int] = []
        all_token_logprobs: list[float] = []
        tokens_generated = 0
        finish_reason = "length"

        while tokens_generated < max_tokens:
            this_chunk_max = min(self.chunk_size, max_tokens - tokens_generated)

            sampling_params = SamplingParams(
                max_tokens=this_chunk_max,
                temperature=request.temperature or 1.0,
                logprobs=0,  # need logprobs
            )
            current_prompt = tokens_input(engine_input + all_token_ids)
            chunk_id = f'cmpl-{random_uuid()}'

            final_result = None
            async for result in self.engine.generate(current_prompt, sampling_params, chunk_id):
                final_result = result
            
            output = final_result.outputs[0]
            chunk_ids = list(output.token_ids)

            for i, tid in enumerate(chunk_ids):
                lp_dict = output.logprobs[i]
                all_token_logprobs.append(lp_dict[tid].logprob if lp_dict else 0.0)
            
            # TODO entropy check here 
    
        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
            CompletionResponseChoice(
                index=0,
                text="",  # trl ingests tokens, not text
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
                prompt_tokens=len(engine_input),
                completion_tokens=len(all_token_ids),
                total_tokens=len(engine_input) + len(all_token_ids),
            ),
        )