import os
from pathlib import Path
from forking.utils import load_cfg

_CONF_PATH = Path(__file__).resolve().parent.parent / "train.yaml"


def run():
    cfg = load_cfg(_CONF_PATH)
    cmd = [
        "vllm", "serve", cfg.model.name,
        "--port", str(cfg.vllm.server_port),
        "--dtype", cfg.model.dtype,
        "--max-model-len", str(cfg.vllm.max_model_length),
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.9",
        "--logprobs-mode", "processed_logprobs",
        "--weight-transfer-config", '{"backend":"nccl"}',
    ]
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    run()