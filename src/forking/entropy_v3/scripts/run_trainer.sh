set -a; source .env; set +a;
wandb login

LD_LIBRARY_PATH=/root/forking/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=1 \
accelerate launch -m forking.entropy_v2.train &> /tmp/train.log 2>&1