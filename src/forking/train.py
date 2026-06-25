from __future__ import annotations

import torch
import logging
import os
import signal
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv
from trl.trl.rewards.accuracy_rewards import accuracy_reward
from trl.trl.trainer.grpo_trainer import GRPOTrainer
from trl.trl.trainer.grpo_config import GRPOConfig
load_dotenv()

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


from forking.utils import load_cfg, wait_for_vllm_servers

logger = logging.getLogger(__name__)
_CONF_PATH = Path(__file__).resolve().parent / "train.conf"


def run():
    cfg = load_cfg(_CONF_PATH)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(cfg.model.dataset, split=cfg.dataset.split)
    if cfg.dataset.get("max_samples") > 0:
        dataset = dataset.select(range(int(cfg.dataset.max_samples)))
    logger.info(f"Dataset size: {len(dataset)}")
    
    vllm_url = f"{cfg.vllm.server_url}:{cfg.vllm.server_port}"
    vllm_urls = [vllm_url]
    wait_for_vllm_servers(vllm_urls, timeout_s=float(cfg.vllm.server_timeout))

    t = cfg.training
    training_args = GRPOConfig(
        output_dir=t.output_dir,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=t.num_generations,
        max_completion_length=t.max_completion_length,
        learning_rate=t.learning_rate,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        temperature=t.temperature,
        report_to=t.report_to,
        seed=t.seed,
        # vLLM
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=vllm_url,
        vllm_server_timeout=cfg.vllm.server_timeout,
        # Policy model init (training side only)
        model_init_kwargs={
            "dtype": getattr(__import__("torch"), cfg.model.dtype),
            "attn_implementation": cfg.model.attn_implementation,
        },
        bf16=cfg.model.dtype == "bfloat16",
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=accuracy_reward,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model("final_model")


if __name__ == '__main__':
    run()
