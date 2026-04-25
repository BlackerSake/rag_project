"""One-command runner for the independent RAG evaluation module."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.config import EvalConfig
from eval.evaluators import E2EEvaluator, FallbackEvaluator, GenerationEvaluator, RetrievalEvaluator

logger = logging.getLogger(__name__)


class LangGraphRAGAdapter:
    """Adapter that exposes ``answer(question)`` for the existing LangGraph app."""

    def __init__(self) -> None:
        """Initialize the adapter.

        Args:
            None.

        Returns:
            None.

        Raises:
            ImportError: If the main graph dependencies cannot be imported.
        """
        from core.builder import compiled_graph
        from langchain_core.messages import HumanMessage

        self.compiled_graph = compiled_graph
        self.human_message_cls = HumanMessage

    async def answer(self, question: str) -> str:
        """Run the main RAG graph and return the latest assistant answer.

        Args:
            question: User question.

        Returns:
            Assistant answer text.

        Raises:
            No exceptions are raised; failures return a safe fallback answer.
        """
        try:
            logger.info("Invoking LangGraph RAG adapter")
            state = {
                "messages": [self.human_message_cls(content=question)],
                "history": [],
                "summary": "无",
                "rounds": 0,
            }
            result = await self.compiled_graph.ainvoke(
                state,
                config={"configurable": {"thread_id": f"eval-{abs(hash(question))}"}},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                latest = messages[-1]
                return str(getattr(latest, "content", latest) or "")
        except Exception as exc:
            logger.error("LangGraph RAG invocation failed: %s", exc)
        return "抱歉，当前无法回答该问题。"


async def default_llm_client(prompt: str) -> str:
    """Evaluate prompts through the project's configured chat model.

    Args:
        prompt: Evaluation prompt.

    Returns:
        LLM response text.

    Raises:
        Propagates model invocation errors to evaluator-level fallback handling.
    """
    from core.models import model
    from langchain_core.messages import HumanMessage

    response = await model.ainvoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response) or "")


def load_cases(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load evaluation cases from JSON.

    Args:
        path: JSON test-case path.

    Returns:
        Case payload with default empty lists for missing sections.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If path is invalid JSON.
    """
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {
        "retrieval_cases": payload.get("retrieval_cases", []),
        "generation_cases": payload.get("generation_cases", []),
        "e2e_cases": payload.get("e2e_cases", []),
        "fallback_cases": payload.get("fallback_cases", []),
    }


def overall_score(report: dict[str, Any], config: EvalConfig) -> float:
    """Calculate the final weighted score.

    Args:
        report: Report containing four metric sections.
        config: Evaluation configuration.

    Returns:
        Overall weighted score.

    Raises:
        No exceptions are raised.
    """
    weights = config.overall_weights
    return (
        weights.retrieval * report.get("retrieval", {}).get("retrieval_score", 0.0)
        + weights.generation * report.get("generation", {}).get("generation_score", 0.0)
        + weights.e2e * report.get("e2e", {}).get("e2e_score", 0.0)
        + weights.fallback * report.get("fallback", {}).get("fallback_score", 0.0)
    )


async def run_evaluation(cases_path: Path, output_path: Path | None = None, enable_bertscore: bool = True) -> dict[str, Any]:
    """Run all RAG evaluation stages.

    Args:
        cases_path: Path to JSON test cases.
        output_path: Optional report output path.
        enable_bertscore: Whether to calculate BERTScore.

    Returns:
        Full evaluation report.

    Raises:
        Propagates test-case loading errors.
    """
    cases = load_cases(cases_path)
    config = EvalConfig()

    from data.knowledge_base import KnowledgeBase

    knowledge_base = KnowledgeBase()
    rag_system = LangGraphRAGAdapter()
    llm_client = default_llm_client

    retrieval = RetrievalEvaluator(knowledge_base, config)
    generation = GenerationEvaluator(llm_client, config)
    e2e = E2EEvaluator(config, enable_bertscore=enable_bertscore)
    fallback = FallbackEvaluator(rag_system, llm_client, config)

    logger.info("Starting retrieval evaluation")
    retrieval_report = await retrieval.evaluate_batch(cases["retrieval_cases"], k=config.top_k)
    logger.info("Starting generation evaluation")
    generation_report = await generation.evaluate_batch(cases["generation_cases"])
    logger.info("Starting E2E evaluation")
    e2e_report = e2e.evaluate_batch(cases["e2e_cases"])
    logger.info("Starting fallback evaluation")
    fallback_report = await fallback.evaluate_batch(cases["fallback_cases"])

    report = {
        "retrieval": retrieval_report,
        "generation": generation_report,
        "e2e": e2e_report,
        "fallback": fallback_report,
    }
    report["overall_score"] = overall_score(report, config)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        logger.info("Evaluation report written to %s", output_path)
    return report


def main() -> None:
    """Parse CLI arguments and run evaluation.

    Args:
        None.

    Returns:
        None.

    Raises:
        Propagates unhandled setup errors to the shell.
    """
    parser = argparse.ArgumentParser(description="Run independent RAG evaluation.")
    parser.add_argument("--cases", type=Path, default=Path("eval/data/test_cases.json"), help="Test cases JSON path")
    parser.add_argument("--output", type=Path, default=Path("eval/eval_report.json"), help="Report JSON output path")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore calculation")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("EVAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = asyncio.run(
        run_evaluation(
            cases_path=args.cases,
            output_path=args.output,
            enable_bertscore=not args.no_bertscore,
        )
    )
    print(json.dumps({"overall_score": report["overall_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

