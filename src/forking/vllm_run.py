from forking.vllm_serve import main
from forking.data_models import ScriptArguments
from forking.utils import load_cfg
from pathlib import Path

_CONF_PATH = Path(__file__).resolve().parent / "train.conf"


def run():
    cfg = load_cfg(_CONF_PATH)
    args = ScriptArguments(
        model=cfg.model.name,
        port=cfg.vllm.server_port,
        dtype=cfg.model.dtype,
        max_model_len=cfg.vllm.max_model_length,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
    main(args)


if __name__ == "__main__":
    run()