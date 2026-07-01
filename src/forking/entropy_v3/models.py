from dataclasses import dataclass, asdict, fields
from typing import Any

@dataclass
class EntropyXargs:
    intervene: int = 0  # whether server intervenes
    tau_vix: float | None = None
    tau_drift: float | None = None
    tau_drawup: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EntropyXargs":
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


ENTROPY_RESPONSE_KEY = "entropy"
VLLM_XARGS_KEY = "vllm_xargs"
REQUESTED_INTERVENE_KEY = "requested_intervene"
INTERVENED_KEY = "intervened"
INTERVENTIONS_USED_KEY = "interventions_used"
INTERVENTIONS_IMPROVED_KEY = "interventions_improved"
INTERVENTION_IMPROVEMENT_RATE_KEY = "intervention_improvement_rate"
SPLIT_INDICES_KEY = "split_indices"
VIX_METADATA_KEY = "vix"
TOPK_LOGPROBS_KEY = "topk_logprobs"
TOPK_KEY = "topk"
TOKEN_IDX_KEY = "token_idx"
TOKEN_ENTROPY_KEY = "entropy"
ENTROPY_XARGS_INTERVENE_KEY = next(
    field.name for field in fields(EntropyXargs) if field.name == "intervene"
)


@dataclass
class VixValues:
    vix: float = 0.0
    drift: float = 0.0
    up_vix: float = 0.0
    down_vix: float = 0.0
    drawup: float = 0.0
    drawdown: float = 0.0

    def to_dict(self):
        return asdict(self)


VIX_VALUE_KEY = next(field.name for field in fields(VixValues) if field.name == "vix")
DRIFT_VALUE_KEY = next(field.name for field in fields(VixValues) if field.name == "drift")
DRAWUP_VALUE_KEY = next(field.name for field in fields(VixValues) if field.name == "drawup")
