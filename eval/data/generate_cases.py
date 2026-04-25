"""从 FAQ 风格 JSON 数据生成初始评测用例。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def stable_doc_id(text: str) -> str:
    """根据文本生成稳定文档 ID。

    参数:
        text: 源文本。

    返回:
        基于 SHA-256 的短 ID。

    异常:
        不主动抛出异常。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_cases_from_faq(faq_path: Path, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    """从 FAQ 数据生成基础检索、生成和端到端评测用例。

    参数:
        faq_path: 包含问答对的 FAQ JSON 文件路径。
        limit: 最多转换的 FAQ 条目数量。

    返回:
        与 ``run_eval.py`` 兼容的测试用例数据。

    异常:
        FileNotFoundError: 当 ``faq_path`` 不存在时抛出。
        json.JSONDecodeError: 当 FAQ 文件不是合法 JSON 时抛出。
    """
    with faq_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = [(item.get("question", ""), item.get("answer", "")) for item in payload if isinstance(item, dict)]
    else:
        logger.warning("不支持的 FAQ 数据类型: %s", type(payload).__name__)
        items = []

    retrieval_cases = []
    generation_cases = []
    e2e_cases = []
    for question, answer in items[: max(0, limit)]:
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        if not question or not answer:
            logger.warning("跳过空 FAQ 条目: question=%r answer=%r", question, answer)
            continue
        doc_id = stable_doc_id(f"{question}\n{answer}")
        context = f"问题: {question}\n答案: {answer}"
        retrieval_cases.append({"query": question, "relevant_docs": [doc_id]})
        generation_cases.append(
            {
                "question": question,
                "ground_truth_answer": answer,
                "generated_answer": answer,
                "retrieved_context": [context],
            }
        )
        e2e_cases.append(
            {
                "question": question,
                "ground_truth_answer": answer,
                "generated_answer": answer,
            }
        )

    return {
        "retrieval_cases": retrieval_cases,
        "generation_cases": generation_cases,
        "e2e_cases": e2e_cases,
        "fallback_cases": [
            {
                "question": "请告诉我明天的实时天气",
                "expected_behavior": "reject",
                "retrieved_context": [],
            }
        ],
    }


def main() -> None:
    """从命令行运行用例生成器。

    参数:
        无。

    返回:
        无。

    异常:
        文件和 JSON 异常会继续抛出到命令行。
    """
    parser = argparse.ArgumentParser(description="从 FAQ JSON 生成 RAG 评测测试用例。")
    parser.add_argument("--faq", type=Path, default=Path("faq_data.json"), help="FAQ JSON 路径")
    parser.add_argument("--output", type=Path, default=Path("eval/data/test_cases.json"), help="输出 JSON 路径")
    parser.add_argument("--limit", type=int, default=20, help="最多转换的 FAQ 条目数量")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cases = generate_cases_from_faq(args.faq, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
    logger.info("评测用例已生成: %s", args.output)


if __name__ == "__main__":
    main()
