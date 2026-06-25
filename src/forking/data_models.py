from pydantic import BaseModel
from dataclasses import dataclass, field


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]

class InitCommunicatorRequest(BaseModel):
    host: str
    port: int
    world_size: int
    client_device_uuid: str

class ChatRequest(BaseModel):
    messages: list[list[dict]]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    logprobs: int | None = 0
    structured_outputs_regex: str | None = None
    generation_kwargs: dict = field(default_factory=dict)
    chat_template_kwargs: dict = field(default_factory=dict)
    tools: list | None = None

class ChatResponse(BaseModel):
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[list[float | None]]] | None
    logprob_token_ids: list[list[list[int]]] | None

class SequenceLogprobsRequest(BaseModel):
    sequences: list[list[int]]
    prompt_lengths: list[int]
    top_logprobs: int = 100
    temperature: float = 1.0
    response_format: str = "json"  # "json" (legacy) or "binary" (base64 numpy arrays)

class SequenceLogprobsResponse(BaseModel):
    logprobs: list[list[list[float | None]]] | None = None
    logprob_token_ids: list[list[list[int]]] | None = None
    # Binary format fields (base64-encoded numpy arrays)
    logprobs_b64: str | None = None
    token_ids_b64: str | None = None
    actual_logprobs_b64: str | None = None
    actual_token_ids_b64: str | None = None
    shape: list[int] | None = None  # [batch_size, max_completion_len, top_logprobs]
    completion_lengths: list[int] | None = None  # actual completion length per sample

class GenerateRequest(BaseModel):
    prompts: list[str] | list[list[int]]
    images: list[list[str] | None] | None = None
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    logprobs: int | None = 0
    structured_outputs_regex: str | None = None
    generation_kwargs: dict = field(default_factory=dict)

class GenerateResponse(BaseModel):
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[list[float | None]]] | None
    logprob_token_ids: list[list[list[int]]] | None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str`, *optional*):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use. For dense models, keep this at 1. Starting from vLLM `0.14.0`,
            setting this above `1` for dense models is no longer supported/useful and will error out (see vLLM PR
            #30739).
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int`, *optional*):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool`, *optional*):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models. Set to `True` to allow executing code from model
            repositories. This is required for some custom models but introduces security risks.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
        distributed_executor_backend (`str` or `None`, *optional*):
            Distributed executor backend for vLLM. Set to `"ray"` to distribute tensor parallel workers across multiple
            nodes via a Ray cluster. Required when `tensor_parallel_size` exceeds the number of local GPUs. If not set,
            vLLM defaults to the multiproc backend (single-node only).
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: str | None = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of data parallel workers to use. For dense models, keep this at 1. Starting from vLLM "
            "`0.14.0`, setting this above `1` for dense models is no longer supported/useful and will error out (see "
            "vLLM PR #30739)."
        },
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models. Set to True to allow executing code from model "
            "repositories. This is required for some custom models but introduces security risks."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    distributed_executor_backend: str | None = field(
        default=None,
        metadata={
            "help": "Distributed executor backend for vLLM. When set to 'ray', vLLM uses Ray to distribute tensor "
            "parallel workers across multiple nodes. Required when tensor_parallel_size exceeds the number of local "
            "GPUs. If not set, vLLM defaults to the multiproc backend (single-node only)."
        },
    )
