from __future__ import annotations

from dataclasses import dataclass, fields
import math

from forking.entropy_v2.models import (
    TOKEN_ENTROPY_KEY,
    TOKEN_IDX_KEY,
    VixValues,
)


TRAIN_WINDOW_STRIDE = 8


@dataclass(frozen=True)
class EntropyWindowFeatures:
    entropy_window: list[float]
    entropy: float
    vix: float
    drift: float
    up_vix: float
    down_vix: float
    drawup: float
    drawdown: float
    vix_max_so_far: float
    drawup_max_so_far: float
    entropy_max_so_far: float
    entropy_min_so_far: float

    def summary_vector(self) -> list[float]:
        return [float(getattr(self, field.name)) for field in fields(self) if field.name != "entropy_window"]

    def as_vector(self) -> list[float]:
        return self.summary_vector()

    def entropy_window_vector(self) -> list[float]:
        return [float(value) for value in self.entropy_window]

    def combined_vector(self) -> list[float]:
        return self.summary_vector() + self.entropy_window_vector()


SUMMARY_FEATURE_NAMES = [field.name for field in fields(EntropyWindowFeatures) if field.name != "entropy_window"]
FEATURE_NAMES = SUMMARY_FEATURE_NAMES


def entropy_window_feature_names(chunk_size: int) -> list[str]:
    return [f"entropy_pos_{i}" for i in range(chunk_size)]


def combined_feature_names(chunk_size: int) -> list[str]:
    return SUMMARY_FEATURE_NAMES + entropy_window_feature_names(chunk_size)


def compute_entropy_trajectory(topk_logprobs: list[list[float]]) -> list[float]:
    entropies: list[float] = []
    for row in topk_logprobs:
        if not row:
            entropies.append(0.0)
            continue
        probs = [math.exp(float(lp)) for lp in row]
        psum = sum(probs)
        if psum <= 0.0:
            entropies.append(0.0)
            continue
        q = [p / psum for p in probs]
        entropies.append(-sum(p * math.log(p) for p in q if p > 0.0))
    return entropies


def compute_rolling_vix(token_entropy: list[float], chunk_size: int) -> list[VixValues]:
    if not token_entropy:
        return []

    deltas = [0.0] + [
        token_entropy[i] - token_entropy[i - 1]
        for i in range(1, len(token_entropy))
    ]

    out: list[VixValues] = []
    for i in range(len(deltas)):
        start = i - chunk_size + 1
        if start < 0:
            out.append(VixValues())
            continue

        local = deltas[start : i + 1]
        n = len(local)
        entropy_window = token_entropy[start : i + 1]
        current_entropy = token_entropy[i]

        out.append(
            VixValues(
                vix=math.sqrt(sum(d * d for d in local) / n),
                drift=sum(local) / n,
                up_vix=math.sqrt(sum(max(d, 0.0) ** 2 for d in local) / n),
                down_vix=math.sqrt(sum(min(d, 0.0) ** 2 for d in local) / n),
                drawup=current_entropy - min(entropy_window),
                drawdown=max(entropy_window) - current_entropy,
            )
        )
    return out


def sampled_vix_metadata(
    token_entropy: list[float],
    vix_values: list[VixValues],
    chunk_size: int,
) -> list[dict[str, float | int]]:
    return [
        {
            TOKEN_IDX_KEY: i,
            TOKEN_ENTROPY_KEY: token_entropy[i],
            **vix_values[i].to_dict(),
        }
        for i in range(chunk_size - 1, len(vix_values), chunk_size)
    ]


def vix_metadata_from_topk_logprobs(
    topk_logprobs: list[list[float]],
    chunk_size: int,
) -> list[dict[str, float | int]]:
    entropies = compute_entropy_trajectory(topk_logprobs)
    return sampled_vix_metadata(
        entropies,
        compute_rolling_vix(entropies, chunk_size),
        chunk_size,
    )


def window_feature_rows_from_topk_logprobs(
    topk_logprobs: list[list[float]],
    chunk_size: int,
) -> list[EntropyWindowFeatures]:
    entropies = compute_entropy_trajectory(topk_logprobs)
    return window_feature_rows_from_entropy_trajectory(
        entropies,
        chunk_size,
        window_stride=TRAIN_WINDOW_STRIDE,
    )


def window_feature_rows_from_entropy_trajectory(
    entropies: list[float],
    chunk_size: int,
    *,
    window_stride: int = 1,
    start_idx: int | None = None,
) -> list[EntropyWindowFeatures]:
    if window_stride < 1:
        raise ValueError(f"window_stride must be >= 1, got {window_stride}")

    vix_values = compute_rolling_vix(entropies, chunk_size)
    rows: list[EntropyWindowFeatures] = []
    first_idx = chunk_size - 1 if start_idx is None else max(start_idx, chunk_size - 1)

    for i in range(first_idx, len(vix_values), window_stride):
        vix_so_far = vix_values[: i + 1]
        entropy_so_far = entropies[: i + 1]
        start = i - chunk_size + 1
        entropy_window = [float(value) for value in entropies[start : i + 1]]
        rows.append(
            EntropyWindowFeatures(
                entropy_window=entropy_window,
                entropy=float(entropies[i]),
                vix=float(vix_values[i].vix),
                drift=float(vix_values[i].drift),
                up_vix=float(vix_values[i].up_vix),
                down_vix=float(vix_values[i].down_vix),
                drawup=float(vix_values[i].drawup),
                drawdown=float(vix_values[i].drawdown),
                vix_max_so_far=float(max(v.vix for v in vix_so_far)),
                drawup_max_so_far=float(max(v.drawup for v in vix_so_far)),
                entropy_max_so_far=float(max(entropy_so_far)),
                entropy_min_so_far=float(min(entropy_so_far)),
            )
        )
    return rows
