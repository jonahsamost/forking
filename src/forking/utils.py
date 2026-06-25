import sys
from omegaconf import DictConfig, OmegaConf
import logging
import time
import requests

logger = logging.getLogger(__name__)


def load_cfg(conf_path) -> DictConfig:
    base = OmegaConf.load(conf_path)
    args = sys.argv[1]
    yaml_args = [a for a in args if a.endswith((".yaml"))]
    dot_overrides = [a for a in args if "=" in a and not a.endswith((".yaml"))]
    cfg = OmegaConf.merge(base, OmegaConf.load(yaml_args[0])) if yaml_args else base    
    if dot_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot_overrides))
    return cfg


def wait_for_vllm_server(
    url: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Block until a single vLLM server responds healthy on ``/health``."""
    url = url.rstrip("/")
    logger.info("Waiting for vLLM server at %s ...", url)
    start = time.time()
    while True:
        elapsed = time.time() - start
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server at %s ready after %.1fs", url, elapsed)
                return
        except (requests.ConnectionError, requests.Timeout, OSError):
            pass
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for vLLM server at {url}."
            )
        if int(elapsed) % 10 < poll_interval_s:
            logger.info("Still waiting for vLLM server at %s... (%.0fs)", url, elapsed)
        time.sleep(poll_interval_s)


def wait_for_vllm_servers(
    urls: list[str],
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Block until every vLLM server in ``urls`` is healthy."""
    for url in urls:
        wait_for_vllm_server(url, timeout_s=timeout_s, poll_interval_s=poll_interval_s)