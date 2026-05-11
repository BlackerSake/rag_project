import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

log_dir = ROOT_DIR / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "intent_eval.log", encoding="utf-8"),
    ],
    force=True,
)

env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

from core.intent_manager import get_intent_manager
from data.confidence import ConfidenceGate
from intent import intent_gate_decide, load_intent_baseline


def load_queries(path: Path) -> list[dict]:
    """加载带 intent 标注的离线评测集。"""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def score_distribution(scores: list[float]) -> dict:
    """统计分数分布，包含 P25/P75。"""
    if not scores:
        return {
            "sample_count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    return {
        "sample_count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "avg": sum(scores) / len(scores),
        "p25": ConfidenceGate._percentile(scores, 25),
        "p75": ConfidenceGate._percentile(scores, 75),
    }


def intent_margin(candidates: list[dict]) -> float:
    """计算意图 Top1 与 Top2 的 margin。"""
    if not candidates:
        return 0.0
    top1 = float(candidates[0].get("score", 0.0))
    if len(candidates) < 2:
        return top1
    top2 = float(candidates[1].get("score", 0.0))
    return max(top1 - top2, 0.0)


def build_confusion_groups(error_samples: list[dict]) -> list[dict]:
    """基于错误样本统计真实意图与预测意图的混淆组。"""
    counter = Counter()
    for sample in error_samples:
        expected = sample.get("expected_intent_id")
        predicted = sample.get("predicted_intent_id")
        if expected and predicted and expected != predicted:
            counter[(expected, predicted)] += 1

    return [
        {
            "expected_intent_id": expected,
            "predicted_intent_id": predicted,
            "count": count,
        }
        for (expected, predicted), count in counter.most_common()
    ]


def summarize_intent_records(records: list[dict], k: int) -> dict:
    """汇总意图评测记录，输出分布、错误样本和混淆组。"""
    top1_scores = [record["top1_score"] for record in records]
    margins = [record["margin"] for record in records]
    error_samples = [record for record in records if not record["hit_top1"]]
    gate_counts = Counter(record["intent_gate_action"] for record in records)
    confidence_counts = Counter(record["intent_confidence_level"] for record in records)

    return {
        "num_queries": len(records),
        "k": k,
        "top1_accuracy": (
            sum(1 for record in records if record["hit_top1"]) / len(records)
            if records else 0.0
        ),
        "top1_score_distribution": score_distribution(top1_scores),
        "margin_distribution": score_distribution(margins),
        "intent_gate_action_distribution": dict(gate_counts),
        "intent_confidence_distribution": dict(confidence_counts),
        "error_count": len(error_samples),
        "error_samples": error_samples,
        "confusion_groups": build_confusion_groups(error_samples),
    }


def build_intent_baseline_from_eval_records(records: list[dict]) -> dict:
    """从当前意图评测记录生成 baseline 配置。"""
    top1_scores = [record["top1_score"] for record in records]
    margins = [record["margin"] for record in records]
    confusion_groups = [
        [item["expected_intent_id"], item["predicted_intent_id"]]
        for item in build_confusion_groups([record for record in records if not record["hit_top1"]])
        if item["count"] >= 2
    ]

    score_stats = score_distribution(top1_scores)
    margin_stats = score_distribution(margins)
    return {
        "score_high": score_stats["p75"],
        "score_low": score_stats["p25"],
        "margin_high": margin_stats["p75"],
        "margin_low": margin_stats["p25"],
        "min_intent_samples": 20,
        "confusion_groups": confusion_groups,
        "stats": {
            "sample_count": len(records),
            "score_count": len(top1_scores),
            "margin_count": len(margins),
            "error_count": sum(1 for record in records if not record["hit_top1"]),
        },
    }


async def run_intent_eval(
    k: int = 3,
    dataset_path: Path = ROOT_DIR / "evaluation" / "dataset" / "eval_test.json",
    baseline_output: Path | None = None,
) -> dict:
    """运行意图离线评测，统计 Top1 score、margin、错误样本与混淆组。"""
    queries = load_queries(dataset_path)
    manager = await get_intent_manager()
    baseline = load_intent_baseline()
    records = []

    for index, q in enumerate(queries, 1):
        query = q["query"]
        expected_intent_id = q.get("intent") or q.get("expected_intent_id")
        candidates = await manager.search_intent_candidates(query, k=k)
        top1 = candidates[0] if candidates else {}
        top2 = candidates[1] if len(candidates) > 1 else {}
        predicted_intent_id = top1.get("intent_id")
        top1_score = float(top1.get("score", 0.0)) if top1 else 0.0
        top2_score = float(top2.get("score", 0.0)) if top2 else 0.0
        margin = intent_margin(candidates)
        gate_decision = intent_gate_decide(query, candidates, baseline)

        record = {
            "index": index,
            "query": query,
            "expected_intent_id": expected_intent_id,
            "predicted_intent_id": predicted_intent_id,
            "hit_top1": bool(expected_intent_id and predicted_intent_id == expected_intent_id),
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": margin,
            "candidates": candidates,
            "intent_confidence_level": gate_decision.get("intent_confidence_level"),
            "intent_gate_action": gate_decision.get("intent_gate_action"),
            "intent_gate_reason": gate_decision.get("intent_gate_reason"),
        }
        records.append(record)

        logging.info(
            "[%d/%d] expected=%s predicted=%s hit=%s top1=%.4f margin=%.4f action=%s | query: %s",
            index,
            len(queries),
            expected_intent_id,
            predicted_intent_id,
            record["hit_top1"],
            top1_score,
            margin,
            record["intent_gate_action"],
            query,
        )

    summary = summarize_intent_records(records, k)
    report = {
        "summary": summary,
        "records": records,
    }

    if baseline_output:
        baseline_payload = build_intent_baseline_from_eval_records(records)
        baseline_output.parent.mkdir(parents=True, exist_ok=True)
        with baseline_output.open("w", encoding="utf-8") as f:
            json.dump(baseline_payload, f, ensure_ascii=False, indent=2)
        report["generated_baseline"] = baseline_payload

    logging.info("=" * 60)
    logging.info("Intent Gate 评测汇总")
    logging.info("-" * 60)
    logging.info("  总查询数: %d", summary["num_queries"])
    logging.info("  Top1准确率: %.4f", summary["top1_accuracy"])
    logging.info(
        "  Top1 score分布: p25=%.4f p75=%.4f avg=%.4f min=%.4f max=%.4f",
        summary["top1_score_distribution"]["p25"],
        summary["top1_score_distribution"]["p75"],
        summary["top1_score_distribution"]["avg"],
        summary["top1_score_distribution"]["min"],
        summary["top1_score_distribution"]["max"],
    )
    logging.info(
        "  Top1-Top2 margin分布: p25=%.4f p75=%.4f avg=%.4f min=%.4f max=%.4f",
        summary["margin_distribution"]["p25"],
        summary["margin_distribution"]["p75"],
        summary["margin_distribution"]["avg"],
        summary["margin_distribution"]["min"],
        summary["margin_distribution"]["max"],
    )
    logging.info("  错误样本数: %d", summary["error_count"])
    logging.info("  混淆组数: %d", len(summary["confusion_groups"]))
    logging.info("=" * 60)
    return report


if __name__ == "__main__":
    default_dataset = ROOT_DIR / "evaluation" / "dataset" / "eval_test.json"

    parser = argparse.ArgumentParser(description="运行 Intent Gate 意图评测")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-output", type=Path, help="可选：将本次评测生成的 baseline 写入指定JSON文件")
    args = parser.parse_args()

    result = asyncio.run(
        run_intent_eval(
            k=args.k,
            dataset_path=args.dataset,
            baseline_output=args.baseline_output,
        )
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info("Intent评测报告已写入: %s", args.output)
