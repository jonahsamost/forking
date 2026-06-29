LD_LIBRARY_PATH=/root/forking/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
VLLM_SERVER_DEV_MODE=1 \
uv run --no-sync python -m forking.entropy.vllm_run