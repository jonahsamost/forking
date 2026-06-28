from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class EntropyXargs:
    entropy: int = 0  # whether to do entropy bookkeeping
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


@dataclass
class EntropyResponse:
    ...

    def to_dict(self):
        ...