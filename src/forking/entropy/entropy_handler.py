
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from fastapi import Request


async def handler(
    request: CompletionRequest, raw_request: Request, engine: AsyncLLM
) -> CompletionResponse:
    ...