from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfidenceThresholds:
    p25: float
    p75: float
    sample_count: int
    window_size: int


class ConfidenceHistory:
    """保存最近 N 次检索 Top-1 置信度分数的 FIFO 窗口。"""

    def __init__(self, max_size: int = 100, initial_scores: Iterable[float] | None = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        self.max_size = max_size
        self._scores: deque[float] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        if initial_scores:
            self.extend(initial_scores)

    def update(self, score: float) -> float:
        normalized_score = self._coerce_score(score)
        with self._lock:
            self._scores.append(normalized_score)
        return normalized_score

    def extend(self, scores: Iterable[float]) -> None:
        with self._lock:
            for score in scores:
                self._scores.append(self._coerce_score(score))

    def clear(self) -> None:
        with self._lock:
            self._scores.clear()

    def values(self) -> list[float]:
        with self._lock:
            return list(self._scores)

    def __len__(self) -> int:
        with self._lock:
            return len(self._scores)

    @staticmethod
    def _coerce_score(score: float) -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("confidence score must be finite")
        return min(max(value, 0.0), 1.0)


class ConfidenceGate:
    """基于滑动窗口 P25/P75 的三段置信度门控。"""

    def __init__(
        self,
        history: ConfidenceHistory,
        fallback_p25: float = 0.5,
        fallback_p75: float = 0.9,
    ) -> None:
        if fallback_p25 > fallback_p75:
            raise ValueError("fallback_p25 must not be greater than fallback_p75")
        self.history = history
        self.fallback_p25 = fallback_p25
        self.fallback_p75 = fallback_p75

    def calibrate_thresholds(self) -> ConfidenceThresholds:
        scores = self.history.values()
        if len(scores) < 2:
            return ConfidenceThresholds(
                p25=self.fallback_p25,
                p75=self.fallback_p75,
                sample_count=len(scores),
                window_size=self.history.max_size,
            )

        return ConfidenceThresholds(
            p25=self._percentile(scores, 25),
            p75=self._percentile(scores, 75),
            sample_count=len(scores),
            window_size=self.history.max_size,
        )

    def decide(self, score: float) -> dict:
        confidence_score = ConfidenceHistory._coerce_score(score)
        thresholds = self.calibrate_thresholds()

        if confidence_score >= thresholds.p75:
            decision = "HIGH"
        elif confidence_score >= thresholds.p25:
            decision = "MEDIUM"
        else:
            decision = "LOW"

        return {
            "decision": decision,
            "confidence_score": confidence_score,
            "p25": thresholds.p25,
            "p75": thresholds.p75,
            "sample_count": thresholds.sample_count,
            "window_size": thresholds.window_size,
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            raise ValueError("values must not be empty")
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        rank = (len(ordered) - 1) * (percentile / 100.0)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[int(rank)]

        weight = rank - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
