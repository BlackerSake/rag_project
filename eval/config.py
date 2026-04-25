"""Configuration for the independent RAG evaluation module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalWeights:
    """Weights used to aggregate retrieval metrics."""

    recall_at_k: float = 0.30
    precision_at_k: float = 0.20
    mrr: float = 0.25
    ndcg_at_k: float = 0.25


@dataclass(frozen=True)
class GenerationWeights:
    """Weights used to aggregate LLM-based generation metrics."""

    faithfulness: float = 0.35
    answer_relevancy: float = 0.30
    context_relevancy: float = 0.20
    context_recall: float = 0.15


@dataclass(frozen=True)
class E2EWeights:
    """Weights used to aggregate end-to-end text similarity metrics."""

    rouge_1: float = 0.20
    rouge_2: float = 0.15
    rouge_l: float = 0.15
    bleu: float = 0.15
    bertscore_f1: float = 0.35


@dataclass(frozen=True)
class FallbackWeights:
    """Weights used to aggregate fallback capability metrics."""

    reject_accuracy: float = 0.50
    non_hallucination_rate: float = 0.30
    safe_reply_rate: float = 0.20


@dataclass(frozen=True)
class OverallWeights:
    """Top-level balanced weights for the final RAG evaluation score."""

    retrieval: float = 0.30
    generation: float = 0.30
    e2e: float = 0.25
    fallback: float = 0.15


@dataclass(frozen=True)
class EvalConfig:
    """Runtime configuration and scoring weights for evaluators."""

    top_k: int = 5
    safe_reply_keywords: tuple[str, ...] = (
        "不知道",
        "无法回答",
        "抱歉",
        "没有找到",
        "超出范围",
        "无法提供",
    )
    retrieval_weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    generation_weights: GenerationWeights = field(default_factory=GenerationWeights)
    e2e_weights: E2EWeights = field(default_factory=E2EWeights)
    fallback_weights: FallbackWeights = field(default_factory=FallbackWeights)
    overall_weights: OverallWeights = field(default_factory=OverallWeights)


BALANCED_CONFIG = EvalConfig()
HIGH_RETRIEVAL_CONFIG = EvalConfig(
    overall_weights=OverallWeights(retrieval=0.40, generation=0.30, e2e=0.20, fallback=0.10)
)
HIGH_GENERATION_CONFIG = EvalConfig(
    overall_weights=OverallWeights(retrieval=0.25, generation=0.45, e2e=0.20, fallback=0.10)
)
SAFETY_FIRST_CONFIG = EvalConfig(
    overall_weights=OverallWeights(retrieval=0.20, generation=0.25, e2e=0.15, fallback=0.40)
)

