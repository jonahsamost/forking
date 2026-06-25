set -a; source .env; set +a

export UV_CACHE_DIR=/root/.cache/uv
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub 

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# https://github.com/vllm-project/vllm/pull/39291/changes
uv pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "deep-gemm @ git+https://github.com/deepseek-ai/DeepGemm.git" --no-build-isolation
uv pip install sentencepiece protobuf datasets==4.3.0 huggingface-hub==1.9.2 hf_transfer "safetensors>=0.4.3"
uv pip install tokenizers psutil pillow wandb
uv pip install --no-deps bitsandbytes accelerate xformers==0.0.34 peft triton
uv pip install math_verify torchcodec
uv pip install --no-deps --upgrade timm
uv pip install vllm==0.20.0
uv pip install huggingface-hub==1.9.2
uv pip install --no-deps transformers==5.6.2
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install -e ./src/trl
uv pip install -e ./src/vllm

# Run load and save model first
export LD_LIBRARY_PATH=/root/h3-llm-finetune/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH