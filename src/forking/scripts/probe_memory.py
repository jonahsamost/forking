"""Probe max sequence length / batch size for AsyncGRPO training and vLLM rollouts.

Two probes:

1. **Backward probe** — synthetic trainer batch, forward/backward + optional AdamW step.
2. **vLLM probe** — long /v1/completions request to check the server handles target seq len.

Usage (backward / trainer GPU memory):
    CUDA_VISIBLE_DEVICES=1 python -m forking.scripts.probe_memory \
        --batch-size 2 \
        --seq-len 4096

Usage (vLLM long-context smoke test):
    python -m forking.scripts.probe_memory \
        --vllm-url http://127.0.0.1:8800 \
        --seq-len 4096 \
        --max-tokens 64
"""

from __future__ import annotations

import argparse
import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe max seq len for trainer backward or vLLM inference.")
    p.add_argument("--vllm-url", default=None, help="If set, run the vLLM probe instead of backward.")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path.")
    p.add_argument("--batch-size", type=int, default=16, help="Per-device microbatch size (backward probe).")
    p.add_argument("--seq-len", type=int, required=True, help="Sequence length to probe.")
    p.add_argument("--max-tokens", type=int, default=64, help="Max completion tokens (vLLM probe).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--request-timeout", type=int, default=600, help="HTTP timeout for vLLM probe.")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--skip-optimizer-step", action="store_true", help="Only run backward, skip optimizer.step().")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _print_memory(label: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"[{label}] allocated_gib={alloc:.2f}  reserved_gib={reserved:.2f}  peak_allocated_gib={peak:.2f}")


def run_vllm_probe(args: argparse.Namespace) -> None:
    url = args.vllm_url.rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    filler = "x " * (args.seq_len * 2)
    messages = [{"role": "user", "content": f"Solve this: {filler}"}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(prompt_ids) < args.seq_len:
        filler = "x " * (args.seq_len * 4)
        messages = [{"role": "user", "content": f"Solve this: {filler}"}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    prompt_ids = prompt_ids[: args.seq_len]
    print(f"vLLM probe: url={url}  model={args.model}  prompt_tokens={len(prompt_ids)}  max_tokens={args.max_tokens}")

    payload = {
        "model": args.model,
        "prompt": prompt_ids,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n": 1,
        "logprobs": 1,
    }

    t0 = time.monotonic()
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=args.request_timeout)
    elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"vLLM probe failed: HTTP {resp.status_code} after {elapsed:.1f}s\n{resp.text[:2000]}")

    data = resp.json()
    usage = data.get("usage", {})
    choice = data["choices"][0]
    print(
        f"vLLM probe succeeded: elapsed_s={elapsed:.1f}  "
        f"finish_reason={choice.get('finish_reason')}  "
        f"prompt_tokens={usage.get('prompt_tokens')}  "
        f"completion_tokens={usage.get('completion_tokens')}"
    )


def run_backward_probe(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the backward memory probe.")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dtype = _DTYPES[args.dtype]
    print(
        f"Backward probe: model={args.model}  device={device}  dtype={args.dtype}  "
        f"batch_size={args.batch_size}  seq_len={args.seq_len}"
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=None, torch_dtype=dtype)
    model = model.to(device)
    model.train()
    model.config.use_cache = False

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable_count:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    _print_memory("after model + optimizer init", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vocab_size = len(tokenizer)
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        loss = outputs.loss
        print(f"loss={loss.item():.6f}")
        _print_memory("after forward", device)

        loss.backward()
        _print_memory("after backward", device)

        if not args.skip_optimizer_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _print_memory("after optimizer step", device)

        print("Backward probe succeeded.")
    except torch.OutOfMemoryError:
        _print_memory("OOM", device)
        raise


def main() -> None:
    args = parse_args()
    if args.vllm_url:
        run_vllm_probe(args)
    else:
        run_backward_probe(args)


if __name__ == "__main__":
    main()
