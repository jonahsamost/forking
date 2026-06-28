from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from omegaconf import OmegaConf

from trl.rewards import accuracy_reward

from forking.utils import load_cfg, sync_chat_template

logger = logging.getLogger(__name__)
_CONF_PATH = Path(__file__).resolve().parent.parent / "train.yaml"


def run():
    cfg = load_cfg(_CONF_PATH)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    if cfg.dataset.get("max_samples") and cfg.dataset.max_samples > 0:
        dataset = dataset.select(range(int(cfg.dataset.max_samples)))
    logger.info("Dataset size: %d", len(dataset))

    vllm_url = f"{cfg.vllm.server_url}:{cfg.vllm.server_port}"
    t = cfg.training

    # Sync first!
    sync_chat_template(
        cfg.model.name,
        Path(__file__).resolve().parent / "../../trl/trl/chat_templates/qwen3.jinja",
    )

    from forking.entropy.entropy_updates import EntropyUpdateTracker
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    config = AsyncGRPOConfig(
        output_dir=t.output_dir,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=t.num_generations,
        max_inflight_tasks=t.max_inflight_tasks,
        max_completion_length=t.max_completion_length,
        learning_rate=t.learning_rate,
        logging_steps=t.logging_steps,
        # save_steps=t.save_steps,
        save_strategy="no",
        temperature=t.temperature,
        report_to=t.report_to,
        seed=t.seed,
        bf16=cfg.model.dtype == "bfloat16",
        vllm_server_base_url=vllm_url,
        vllm_server_timeout=cfg.vllm.server_timeout,
        log_completions=False,
        chat_template_kwargs={"enable_thinking": False},

        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
    )
    single_dev_bs = t.gradient_accumulation_steps * t.per_device_train_batch_size
    entropy_tracker = EntropyUpdateTracker(
        bootstrap_records=single_dev_bs,
        update_interval=cfg.entropy.update_interval,
        calibration_ema=cfg.entropy.calibration_ema,
    )

    trainer = AsyncGRPOTrainer(
        model=cfg.model.name,
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
        entropy_tracker=entropy_tracker,
    )

    trainer.train()
    trainer.save_model("final_model")


if __name__ == "__main__":
    run()
