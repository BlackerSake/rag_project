"""Generate starter evaluation cases from FAQ-style JSON data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def stable_doc_id(text: str) -> str:
    """Generate a stable document ID from text.

    Args:
        text: Source text.

    Returns:
        Short SHA-256 based ID.

    Raises:
        No exceptions are raised.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_cases_from_faq(faq_path: Path, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    """Generate basic retrieval, generation and E2E cases from FAQ data.

    Args:
        faq_path: Path to a JSON FAQ file containing question-answer pairs.
        limit: Maximum number of FAQ items to convert.

    Returns:
        Test case payload compatible with ``run_eval.py``.

    Raises:
        FileNotFoundError: If ``faq_path`` does not exist.
        json.JSONDecodeError: If the FAQ file is not valid JSON.
    """
    with faq_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = [(item.get("question", ""), item.get("answer", "")) for item in payload if isinstance(item, dict)]
    else:
        logger.warning("Unsupported FAQ payload type: %s", type(payload).__name__)
        items = []

    retrieval_cases = []
    generation_cases = []
    e2e_cases = []
    for question, answer in items[: max(0, limit)]:
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        if not question or not answer:
            logger.warning("Skipping empty FAQ item: question=%r answer=%r", question, answer)
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
    """Run the case generator from the command line.

    Args:
        None.

    Returns:
        None.

    Raises:
        Propagates file and JSON errors to the shell.
    """
    parser = argparse.ArgumentParser(description="Generate RAG evaluation test cases from FAQ JSON.")
    parser.add_argument("--faq", type=Path, default=Path("faq_data.json"), help="FAQ JSON path")
    parser.add_argument("--output", type=Path, default=Path("eval/data/test_cases.json"), help="Output JSON path")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of FAQ items")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cases = generate_cases_from_faq(args.faq, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
    logger.info("Generated evaluation cases at %s", args.output)


if __name__ == "__main__":
    main()

