set -a; source .env; set +a;
wandb login

export LD_LIBRARY_PATH=/root/h3-llm-finetune/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 \
accelerate launch -m forking.train &> /tmp/train.log 2>&1