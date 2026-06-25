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

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET = "trl-lib/DeepMath-103K"


def run():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(DATASET, split="train")
    training_args = GRPOConfig(
    )
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=accuracy_reward,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model("final_model")
