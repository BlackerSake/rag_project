"""RAG 生成答案的端到端质量评测器。"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import clamp_score, mean, safe_divide

logger = logging.getLogger(__name__)


class E2EEvaluator:
    """评估生成答案的 ROUGE、BLEU 和 BERTScore。"""

    def __init__(self, config: EvalConfig | None = None, enable_bertscore: bool = True) -> None:
        """初始化端到端质量评测器。

        参数:
            config: 可选评测配置。
            enable_bertscore: 是否尝试计算 BERTScore。

        返回:
            无。

        异常:
            不主动抛出异常。
        """
        self.config = config or EvalConfig()
        self.enable_bertscore = enable_bertscore

    def evaluate_case(self, ground_truth_answer: str, generated_answer: str) -> dict[str, float]:
        """用一个参考答案评估一条生成答案。

        参数:
            ground_truth_answer: 参考答案。
            generated_answer: 生成答案。

        返回:
            包含 ROUGE-1/2/L、BLEU、BERTScore F1 和聚合得分的字典。

        异常:
            不主动抛出异常；指标计算失败会记录日志并降级。
        """
        ref_tokens = self._tokenize(ground_truth_answer)
        hyp_tokens = self._tokenize(generated_answer)
        if not ref_tokens or not hyp_tokens:
            logger.warning("端到端评测中的参考答案或生成答案分词为空")

        rouge_scores = self._rouge_scores(ref_tokens, hyp_tokens)
        bleu = self._bleu_score(ref_tokens, hyp_tokens)
        bertscore_f1 = self._bertscore_f1(ground_truth_answer, generated_answer)

        metrics = {
            **rouge_scores,
            "bleu": bleu,
            "bertscore_f1": bertscore_f1,
        }
        metrics["e2e_score"] = self.calculate_score(metrics)
        logger.info("端到端单用例评测完成: %.4f", metrics["e2e_score"])
        return metrics

    def evaluate_batch(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """评估多组答案对的端到端质量。

        参数:
            cases: 包含 ``ground_truth_answer`` 和 ``generated_answer`` 的用例列表。

        返回:
            平均指标、聚合得分和用例详情。

        异常:
            不主动抛出异常。
        """
        if not cases:
            logger.warning("端到端评测用例列表为空")
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
        logger.info("端到端批量评测完成: %.4f", summary["e2e_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """按配置权重聚合端到端指标。

        参数:
            metrics: 包含端到端指标值的字典。

        返回:
            加权端到端质量得分。

        异常:
            不主动抛出异常。
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
        """优先使用 jieba 对中文文本分词。

        参数:
            text: 原始答案文本。

        返回:
            分词列表。

        异常:
            不主动抛出异常。
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return []
        try:
            import jieba

            return [token for token in jieba.cut(cleaned) if token.strip()]
        except Exception as exc:
            logger.warning("jieba 分词失败，降级为按字符切分: %s", exc)
            return list(cleaned)

    def _rouge_scores(self, ref_tokens: list[str], hyp_tokens: list[str]) -> dict[str, float]:
        """计算 ROUGE-1、ROUGE-2 和 ROUGE-L F1。

        参数:
            ref_tokens: 已分词的参考答案。
            hyp_tokens: 已分词的生成答案。

        返回:
            ROUGE 指标字典。

        异常:
            不主动抛出异常。
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
            logger.warning("rouge 包计算失败，使用本地降级实现: %s", exc)
            return {
                "rouge_1": self._rouge_n_f1(ref_tokens, hyp_tokens, 1),
                "rouge_2": self._rouge_n_f1(ref_tokens, hyp_tokens, 2),
                "rouge_l": self._rouge_l_f1(ref_tokens, hyp_tokens),
            }

    def _bleu_score(self, ref_tokens: list[str], hyp_tokens: list[str]) -> float:
        """使用平滑方法计算 BLEU。

        参数:
            ref_tokens: 已分词的参考答案。
            hyp_tokens: 已分词的生成答案。

        返回:
            BLEU 分数。

        异常:
            不主动抛出异常。
        """
        if not ref_tokens or not hyp_tokens:
            return 0.0
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            smoothie = SmoothingFunction().method1
            return clamp_score(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
        except Exception as exc:
            logger.warning("nltk BLEU 计算失败，使用 unigram precision 降级实现: %s", exc)
            ref_counts = Counter(ref_tokens)
            overlap = 0
            for token in hyp_tokens:
                if ref_counts[token] > 0:
                    overlap += 1
                    ref_counts[token] -= 1
            return safe_divide(overlap, len(hyp_tokens))

    def _bertscore_f1(self, reference: str, hypothesis: str) -> float:
        """在启用且依赖可用时计算 BERTScore F1。

        参数:
            reference: 参考答案。
            hypothesis: 生成答案。

        返回:
            BERTScore F1；不可用时返回 0.0。

        异常:
            不主动抛出异常。
        """
        if not self.enable_bertscore:
            logger.info("BERTScore 已按配置禁用")
            return 0.0
        if not reference or not hypothesis:
            return 0.0
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score([hypothesis], [reference], lang="zh", rescale_with_baseline=True)
            return clamp_score(f1.item())
        except Exception as exc:
            logger.warning("BERTScore 计算失败，返回 0.0: %s", exc)
            return 0.0

    def _rouge_n_f1(self, ref_tokens: list[str], hyp_tokens: list[str], n: int) -> float:
        """计算本地降级版 ROUGE-N F1。

        参数:
            ref_tokens: 已分词的参考答案。
            hyp_tokens: 已分词的生成答案。
            n: N-gram 大小。

        返回:
            ROUGE-N F1 分数。

        异常:
            不主动抛出异常。
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
        """计算本地降级版 ROUGE-L F1。

        参数:
            ref_tokens: 已分词的参考答案。
            hyp_tokens: 已分词的生成答案。

        返回:
            ROUGE-L F1 分数。

        异常:
            不主动抛出异常。
        """
        if not ref_tokens or not hyp_tokens:
            return 0.0
        lcs = self._lcs_length(ref_tokens, hyp_tokens)
        precision = safe_divide(lcs, len(hyp_tokens))
        recall = safe_divide(lcs, len(ref_tokens))
        return safe_divide(2 * precision * recall, precision + recall)

    def _ngrams(self, tokens: list[str], n: int) -> list[tuple[str, ...]]:
        """从分词列表构造 n-gram。

        参数:
            tokens: 分词列表。
            n: N-gram 大小。

        返回:
            n-gram 元组列表。

        异常:
            不主动抛出异常。
        """
        if n <= 0 or len(tokens) < n:
            return []
        return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]

    def _lcs_length(self, left: list[str], right: list[str]) -> int:
        """计算最长公共子序列长度。

        参数:
            left: 第一个分词序列。
            right: 第二个分词序列。

        返回:
            LCS 长度。

        异常:
            不主动抛出异常。
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
        """返回空端到端质量评测汇总。

        参数:
            无。

        返回:
            全零汇总结果。

        异常:
            不主动抛出异常。
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
