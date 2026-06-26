from __future__ import annotations

import logging
import signal
from pathlib import Path
import asyncio
import uvloop
from argparse import Namespace

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.system_utils import set_ulimit
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from fastapi import Request

from forking.entropy.entropy_handler import EntropyHandler, handler
from forking.utils import load_cfg

logger = logging.getLogger(__name__)
_CONF_PATH = Path(__file__).resolve().parent.parent / "train.yaml"


async def run_server(args: Namespace, **uvicorn_kwargs) -> None:
    cfg = load_cfg(_CONF_PATH)
    sock_addr = (args.host or "", args.port)
    from vllm.entrypoints.openai.api_server import create_server_socket
    sock = create_server_socket(sock_addr)
    set_ulimit()

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, signal_handler)
    
    engine_args = AsyncEngineArgs.from_cli_args(args)

    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        enable_log_requests=engine_args.enable_log_requests,
    )
    assert isinstance(engine.engine_core, AsyncMPClient)

    supported_tasks = await engine.get_supported_tasks()
    model_config = engine.model_config if hasattr(engine, "model_config") else None
    app = build_app(args, supported_tasks, model_config)

    entropy_handler = EntropyHandler(
        engine,
        chunk_size=cfg.training.threshold_chunk_size,
        threshold_burst=cfg.training.threshold_burst,
        threshold_rebound=cfg.training.threshold_rebound,
    )

    @app.post("/v1/completions")
    async def _entropy_completions(request: CompletionRequest, raw_request: Request):
        logger.info("Hit entropy completions!")
        return await entropy_handler.handler(request, raw_request, engine)

    await init_app_state(engine, app.state, args, supported_tasks)

    shutdown_task = await serve_http(
        app, sock=sock, host=args.host, port=args.port, **uvicorn_kwargs
    )

    try:
        await shutdown_task
    finally:
        sock.close()

def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible server with direct LoRA NCCL sync"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))



if __name__ == "__main__":
    main()
