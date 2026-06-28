from dataclasses import dataclass, asdict

@dataclass
class EntropyXargs:
    entropy: int  # whether to do entroy book-keeping
    intervene: int  # whether server intervenes
    max_interventions: int


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