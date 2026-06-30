from pathlib import Path
import os
import sys

from forking.utils import load_cfg

_CONF_PATH = Path(__file__).resolve().parent.parent / "train.yaml"


def run():
    cfg = load_cfg(_CONF_PATH)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    sys.argv = [
        "entropy_vllm_server",
        "--model", cfg.model.name,
        "--port", str(cfg.vllm.server_port),
        "--dtype", cfg.model.dtype,
        "--max-model-len", str(cfg.vllm.max_model_length),
        "--enable-prefix-caching",
        "--language-model-only",
        "--gpu-memory-utilization", "0.9",
        "--logprobs-mode", "processed_logprobs",
        "--weight-transfer-config", '{"backend":"nccl"}',
    ]
    from forking.entropy_v2.vllm_server import main as server_main

    server_main()


if __name__ == "__main__":
    run()