export LD_LIBRARY_PATH=/root/h3-llm-finetune/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync python -m forking.vllm_run