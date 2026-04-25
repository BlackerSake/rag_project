"""End-to-end quality evaluator for generated RAG answers."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import clamp_score, mean, safe_divide

logger = logging.getLogger(__name__)


class E2EEvaluator:
    """Evaluate ROUGE, BLEU and BERTScore for generated answers."""

    def __init__(self, config: EvalConfig | None = None, enable_bertscore: bool = True) -> None:
        """Initialize the end-to-end evaluator.

        Args:
            config: Optional evaluation configuration.
            enable_bertscore: Whether to attempt BERTScore calculation.

        Returns:
            None.

        Raises:
            No exceptions are raised.
        """
        self.config = config or EvalConfig()
        self.enable_bertscore = enable_bertscore

    def evaluate_case(self, ground_truth_answer: str, generated_answer: str) -> dict[str, float]:
        """Evaluate one generated answer against one reference answer.

        Args:
            ground_truth_answer: Reference answer.
            generated_answer: Generated answer.

        Returns:
            Dictionary with ROUGE-1/2/L, BLEU, BERTScore F1 and aggregate score.

        Raises:
            No exceptions are raised; metric failures are logged and downgraded.
        """
        ref_tokens = self._tokenize(ground_truth_answer)
        hyp_tokens = self._tokenize(generated_answer)
        if not ref_tokens or not hyp_tokens:
            logger.warning("Empty reference or hypothesis tokens in E2E evaluation")

        rouge_scores = self._rouge_scores(ref_tokens, hyp_tokens)
        bleu = self._bleu_score(ref_tokens, hyp_tokens)
        bertscore_f1 = self._bertscore_f1(ground_truth_answer, generated_answer)

        metrics = {
            **rouge_scores,
            "bleu": bleu,
            "bertscore_f1": bertscore_f1,
        }
        metrics["e2e_score"] = self.calculate_score(metrics)
        logger.info("E2E case evaluated: %.4f", metrics["e2e_score"])
        return metrics

    def evaluate_batch(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate end-to-end quality for multiple answer pairs.

        Args:
            cases: Items with ``ground_truth_answer`` and ``generated_answer``.

        Returns:
            Averaged metrics, aggregate score and per-case details.

        Raises:
            No exceptions are raised.
        """
        if not cases:
            logger.warning("Empty E2E case list")
            return self._empty_summary()

        details = [
            self.evaluate_case(
                ground_truth_answer=case.get("ground_truth_answer", ""),
                generated_answer=case.get("generated_answer", ""),
            )
            for case in cases
        ]
        summary = {
            "rouge_1": mean([item["rouge_1"] for item in details]),
            "rouge_2": mean([item["rouge_2"] for item in details]),
            "rouge_l": mean([item["rouge_l"] for item in details]),
            "bleu": mean([item["bleu"] for item in details]),
            "bertscore_f1": mean([item["bertscore_f1"] for item in details]),
            "details": details,
        }
        summary["e2e_score"] = self.calculate_score(summary)
        logger.info("E2E batch evaluated: %.4f", summary["e2e_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """Aggregate E2E metrics according to configured weights.

        Args:
            metrics: Dictionary with E2E metric values.

        Returns:
            Weighted E2E score.

        Raises:
            No exceptions are raised.
        """
        weights = self.config.e2e_weights
        return (
            weights.rouge_1 * metrics.get("rouge_1", 0.0)
            + weights.rouge_2 * metrics.get("rouge_2", 0.0)
            + weights.rouge_l * metrics.get("rouge_l", 0.0)
            + weights.bleu * metrics.get("bleu", 0.0)
            + weights.bertscore_f1 * metrics.get("bertscore_f1", 0.0)
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Chinese text with jieba when available.

        Args:
            text: Raw answer text.

        Returns:
            Token list.

        Raises:
            No exceptions are raised.
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return []
        try:
            import jieba

            return [token for token in jieba.cut(cleaned) if token.strip()]
        except Exception as exc:
            logger.warning("jieba tokenization failed, falling back to characters: %s", exc)
            return list(cleaned)

    def _rouge_scores(self, ref_tokens: list[str], hyp_tokens: list[str]) -> dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2 and ROUGE-L F1.

        Args:
            ref_tokens: Tokenized reference answer.
            hyp_tokens: Tokenized generated answer.

        Returns:
            ROUGE metric dictionary.

        Raises:
            No exceptions are raised.
        """
        try:
            from rouge import Rouge

            if not ref_tokens or not hyp_tokens:
                return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hyp_tokens), " ".join(ref_tokens))[0]
            return {
                "rouge_1": clamp_score(scores["rouge-1"]["f"]),
                "rouge_2": clamp_score(scores["rouge-2"]["f"]),
                "rouge_l": clamp_score(scores["rouge-l"]["f"]),
            }
        except Exception as exc:
            logger.warning("rouge package calculation failed, using local fallback: %s", exc)
            return {
                "rouge_1": self._rouge_n_f1(ref_tokens, hyp_tokens, 1),
                "rouge_2": self._rouge_n_f1(ref_tokens, hyp_tokens, 2),
                "rouge_l": self._rouge_l_f1(ref_tokens, hyp_tokens),
            }

    def _bleu_score(self, ref_tokens: list[str], hyp_tokens: list[str]) -> float:
        """Calculate BLEU with smoothing.

        Args:
            ref_tokens: Tokenized reference answer.
            hyp_tokens: Tokenized generated answer.

        Returns:
            BLEU score.

        Raises:
            No exceptions are raised.
        """
        if not ref_tokens or not hyp_tokens:
            return 0.0
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            smoothie = SmoothingFunction().method1
            return clamp_score(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
        except Exception as exc:
            logger.warning("nltk BLEU calculation failed, using unigram precision fallback: %s", exc)
            ref_counts = Counter(ref_tokens)
            overlap = 0
            for token in hyp_tokens:
                if ref_counts[token] > 0:
                    overlap += 1
                    ref_counts[token] -= 1
            return safe_divide(overlap, len(hyp_tokens))

    def _bertscore_f1(self, reference: str, hypothesis: str) -> float:
        """Calculate BERTScore F1 when enabled and available.

        Args:
            reference: Reference answer.
            hypothesis: Generated answer.

        Returns:
            BERTScore F1, or 0.0 when unavailable.

        Raises:
            No exceptions are raised.
        """
        if not self.enable_bertscore:
            logger.info("BERTScore disabled by configuration")
            return 0.0
        if not reference or not hypothesis:
            return 0.0
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score([hypothesis], [reference], lang="zh", rescale_with_baseline=True)
            return clamp_score(f1.item())
        except Exception as exc:
            logger.warning("BERTScore calculation failed, returning 0.0: %s", exc)
            return 0.0

    def _rouge_n_f1(self, ref_tokens: list[str], hyp_tokens: list[str], n: int) -> float:
        """Calculate local ROUGE-N F1 fallback.

        Args:
            ref_tokens: Tokenized reference answer.
            hyp_tokens: Tokenized generated answer.
            n: N-gram size.

        Returns:
            ROUGE-N F1 score.

        Raises:
            No exceptions are raised.
        """
        ref_ngrams = self._ngrams(ref_tokens, n)
        hyp_ngrams = self._ngrams(hyp_tokens, n)
        if not ref_ngrams or not hyp_ngrams:
            return 0.0
        ref_counts = Counter(ref_ngrams)
        overlap = 0
        for ngram in hyp_ngrams:
            if ref_counts[ngram] > 0:
                overlap += 1
                ref_counts[ngram] -= 1
        precision = safe_divide(overlap, len(hyp_ngrams))
        recall = safe_divide(overlap, len(ref_ngrams))
        return safe_divide(2 * precision * recall, precision + recall)

    def _rouge_l_f1(self, ref_tokens: list[str], hyp_tokens: list[str]) -> float:
        """Calculate local ROUGE-L F1 fallback.

        Args:
            ref_tokens: Tokenized reference answer.
            hyp_tokens: Tokenized generated answer.

        Returns:
            ROUGE-L F1 score.

        Raises:
            No exceptions are raised.
        """
        if not ref_tokens or not hyp_tokens:
            return 0.0
        lcs = self._lcs_length(ref_tokens, hyp_tokens)
        precision = safe_divide(lcs, len(hyp_tokens))
        recall = safe_divide(lcs, len(ref_tokens))
        return safe_divide(2 * precision * recall, precision + recall)

    def _ngrams(self, tokens: list[str], n: int) -> list[tuple[str, ...]]:
        """Build n-grams from token list.

        Args:
            tokens: Token list.
            n: N-gram size.

        Returns:
            List of n-gram tuples.

        Raises:
            No exceptions are raised.
        """
        if n <= 0 or len(tokens) < n:
            return []
        return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]

    def _lcs_length(self, left: list[str], right: list[str]) -> int:
        """Compute longest common subsequence length.

        Args:
            left: First token sequence.
            right: Second token sequence.

        Returns:
            LCS length.

        Raises:
            No exceptions are raised.
        """
        previous = [0] * (len(right) + 1)
        for left_token in left:
            current = [0]
            for index, right_token in enumerate(right, start=1):
                if left_token == right_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        return previous[-1]

    def _empty_summary(self) -> dict[str, Any]:
        """Return an empty E2E summary.

        Args:
            None.

        Returns:
            Zero-valued summary.

        Raises:
            No exceptions are raised.
        """
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
            "bertscore_f1": 0.0,
            "e2e_score": 0.0,
            "details": [],
        }

