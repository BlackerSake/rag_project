"""独立 RAG 评测模块的一键运行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pathlib import Path
from dotenv import load_dotenv

# 定位到项目根目录（假设 run_eval.py 在 eval/ 下）
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.config import EvalConfig
from eval.evaluators import E2EEvaluator, FallbackEvaluator, GenerationEvaluator, RetrievalEvaluator

logger = logging.getLogger(__name__)


class LangGraphRAGAdapter:
    """为现有 LangGraph 应用提供 ``answer(question)`` 接口的适配器。"""

    def __init__(self) -> None:
        """初始化 LangGraph 适配器。

        参数:
            无。

        返回:
            无。

        异常:
            ImportError: 当主系统图依赖无法导入时抛出。
        """
        from core.builder import compiled_graph
        from langchain_core.messages import HumanMessage

        self.compiled_graph = compiled_graph
        self.human_message_cls = HumanMessage

    async def answer(self, question: str) -> str:
        """运行主 RAG 图并返回最新助手回复。

        参数:
            question: 用户问题。

        返回:
            助手回复文本。

        异常:
            不主动抛出异常；调用失败时返回安全兜底回复。
        """
        try:
            logger.info("调用 LangGraph RAG 适配器")
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
            logger.error("LangGraph RAG 调用失败: %s", exc)
        return "抱歉，当前无法回答该问题。"


async def default_llm_client(prompt: str) -> str:
    """通过项目已配置的聊天模型执行评测提示词。

    参数:
        prompt: 评测提示词。

    返回:
        LLM 响应文本。

    异常:
        模型调用异常会继续抛出，由 Evaluator 层统一降级处理。
    """
    from core.models import model
    from langchain_core.messages import HumanMessage

    response = await model.ainvoke([HumanMessage(content=prompt)])
    return str(getattr(response, "content", response) or "")


def load_cases(path: Path) -> dict[str, list[dict[str, Any]]]:
    """从 JSON 文件加载评测用例。

    参数:
        path: JSON 测试用例路径。

    返回:
        评测用例数据；缺失的评测分区会使用空列表补齐。

    异常:
        FileNotFoundError: 当路径不存在时抛出。
        json.JSONDecodeError: 当 JSON 格式无效时抛出。
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
    """计算最终加权总分。

    参数:
        report: 包含四类评测结果的报告。
        config: 评测配置。

    返回:
        最终加权总分。

    异常:
        不主动抛出异常。
    """
    weights = config.overall_weights
    return (
        weights.retrieval * report.get("retrieval", {}).get("retrieval_score", 0.0)
        + weights.generation * report.get("generation", {}).get("generation_score", 0.0)
        + weights.e2e * report.get("e2e", {}).get("e2e_score", 0.0)
        + weights.fallback * report.get("fallback", {}).get("fallback_score", 0.0)
    )


async def run_evaluation(cases_path: Path, output_path: Path | None = None, enable_bertscore: bool = True) -> dict[str, Any]:
    """运行完整 RAG 评测流程。

    参数:
        cases_path: JSON 测试用例路径。
        output_path: 可选的评测报告输出路径。
        enable_bertscore: 是否计算 BERTScore。

    返回:
        完整评测报告。

    异常:
        测试用例加载异常会继续抛出。
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

    logger.info("开始检索质量评测")
    retrieval_report = await retrieval.evaluate_batch(cases["retrieval_cases"], k=config.top_k)
    logger.info("开始生成质量评测")
    generation_report = await generation.evaluate_batch(cases["generation_cases"])
    logger.info("开始端到端质量评测")
    e2e_report = e2e.evaluate_batch(cases["e2e_cases"])
    logger.info("开始兜底能力评测")
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
        logger.info("评测报告已写入: %s", output_path)
    return report


def main() -> None:
    """解析命令行参数并运行评测。

    参数:
        无。

    返回:
        无。

    异常:
        未处理的初始化异常会继续抛出到命令行。
    """
    parser = argparse.ArgumentParser(description="运行独立 RAG 评测。")
    parser.add_argument("--cases", type=Path, default=Path("eval/data/test_cases.json"), help="测试用例 JSON 路径")
    parser.add_argument("--output", type=Path, default=Path("eval/eval_report.json"), help="评测报告 JSON 输出路径")
    parser.add_argument("--no-bertscore", action="store_true", help="禁用 BERTScore 计算")
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
