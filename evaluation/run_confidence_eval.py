import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.confidence import ConfidenceGate, ConfidenceHistory
from data.knowledge_base import KnowledgeBase
from evaluation.metrics import recall_at_k

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [
                        logging.FileHandler(log_dir / "confidence_eval.log",encoding="utf-8"),
                        ],
                    force=True,
                    )

from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)



def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unique_doc_ids(vector_results: list) -> list:
    pred_ids = []
    seen = set()
    for doc, _score in vector_results:
        doc_id = doc.metadata.get("mysql_id")
        if doc_id is not None and doc_id not in seen:
            pred_ids.append(doc_id)
            seen.add(doc_id)
    return pred_ids


def top1_score(vector_results: list) -> float:
    if not vector_results:
        return 0.0
    return float(vector_results[0][1])


def extract_signal(vector_results: list, signal_type: str) -> dict:
    """從檢索結果 提取置信度信號"""
    if not vector_results:
        return {
            "score": 0.0,
            "degraded": False,
            "update_window": False,
            "reason": "empty_results",
        }
    
    top1 = float(vector_results[0][1])
    
    if signal_type == "margin":
        if len(vector_results) >= 2:
            top2 = float(vector_results[1][1])
            return {
                "score": max(top1 - top2, 0.0),
                "degraded": False,
                "update_window": True,
                "reason": None,
            }

        # 只有一條結果時，本條仍可用 top1 做记录和决策，但不进入 margin 阈值窗口。
        return {
            "score": top1,
            "degraded": True,
            "update_window": False,
            "reason": "margin_degraded_to_top1",
        }
    
    return {
        "score": top1,
        "degraded": False,
        "update_window": True,
        "reason": None,
    }


def offline_thresholds(scores: list[float]) -> dict:
    if len(scores) < 2:
        return {"p25": 0.5, "p75": 0.9, "sample_count": len(scores)}
    return {
        "p25": ConfidenceGate._percentile(scores, 25),
        "p75": ConfidenceGate._percentile(scores, 75),
        "sample_count": len(scores),
    }


def collect_vector_scores(kb: KnowledgeBase, queries: list[dict], k: int, signal_type: str = "absolute") -> list[float]:
    scores = []
    for q in queries:
        vector_results, _vector_time = kb.vector_search(q["query"], k=k)
        signal = extract_signal(vector_results, signal_type)
        if signal["update_window"]:
            scores.append(signal["score"])
    return scores


def summarize(records: list[dict], window_size: int) -> dict:
    phase_summary = {}
    for phase_name in ("fill", "steady"):
        phase_records = [record for record in records if record["phase"] == phase_name]
        counts = Counter(record["decision"] for record in phase_records)
        phase_summary[phase_name] = {
            "num_queries": len(phase_records),
            "decision_distribution": {
                decision: counts.get(decision, 0)
                for decision in ("HIGH", "MEDIUM", "LOW")
            },
        }

    recall_by_decision = {}
    grouped_recall = defaultdict(list)
    for record in records:
        grouped_recall[record["decision"]].append(record["recall_at_k"])

    for decision in ("HIGH", "MEDIUM", "LOW"):
        recalls = grouped_recall.get(decision, [])
        recall_by_decision[decision] = {
            "num_queries": len(recalls),
            "avg_recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        }

    return {
        "num_queries": len(records),
        "window_size": window_size,
        "phase_summary": phase_summary,
        "recall_by_decision": recall_by_decision,
    }


def run_confidence_eval(
    k: int = 3,
    prefill_offline: bool = False,
    signal_type: str = "absolute",
    dataset_path: Path = Path("evaluation/dataset/eval_test.json"),
) -> dict:

    queries = load_queries(dataset_path)
    kb = KnowledgeBase()
    kb.ensure_connected()
    window_size = int(getattr(kb.config, "confidence_window_size", 100))
    confidence_history = ConfidenceHistory(max_size=window_size)
    confidence_gate = ConfidenceGate(
        history=confidence_history,
        fallback_p25=float(getattr(kb.config, "confidence_fallback_p25", 0.5)),
        fallback_p75=float(getattr(kb.config, "confidence_fallback_p75", 0.9)),
    )

    offline = None
    if prefill_offline:
        logging.info("爲置信度分數 收集離線向量 Top-1 結果")
        calibration_scores = collect_vector_scores(kb, queries, k=k, signal_type=signal_type)
        offline = offline_thresholds(calibration_scores)
        confidence_history.extend(calibration_scores)
        logging.info(
            "離線置信度 加載✅: scores=%d, p25=%.4f, p75=%.4f",
            offline["sample_count"],
            offline["p25"],
            offline["p75"],
        )

    records = []
    total_latency = 0.0

    for index, q in enumerate(queries, 1):
        start = time.time()
        vector_results, vector_time = kb.vector_search(q["query"], k=k)
        signal = extract_signal(vector_results, signal_type)
        if signal["update_window"]:
            confidence_history.update(signal["score"])
        gate_result = confidence_gate.decide(signal["score"])
        latency = time.time() - start
        total_latency += latency

        pred_ids = unique_doc_ids(vector_results)
        relevant_items = set(q.get("expected_doc_ids", []))
        recall = recall_at_k(pred_ids, relevant_items, k)
        phase = "steady" if prefill_offline or index > window_size else "fill"

        record = {
            "index": index,
            "phase": phase,
            "signal_type": signal_type,
            "signal_degraded": signal["degraded"],
            "signal_reason": signal["reason"],
            "signal_updated_window": signal["update_window"],
            "query": q["query"],
            "intent": q.get("intent"),
            "expected_doc_ids": q.get("expected_doc_ids", []),
            "pred_doc_ids": pred_ids,
            "decision": gate_result["decision"],
            "confidence_score": gate_result["confidence_score"],
            "p25": gate_result["p25"],
            "p75": gate_result["p75"],
            "sample_count": gate_result["sample_count"],
            "recall_at_k": recall,
            "vector_latency_seconds": vector_time,
            "latency_seconds": latency,
        }
        records.append(record)

        logging.info(
            "[%d/%d] %s | score=%.4f | p25=%.4f p75=%.4f | recall@%d=%.4f\n"
            "  query: %s",
            index,
            len(queries),
            record["decision"],
            record["confidence_score"],
            record["p25"],
            record["p75"],
            k,
            recall,
            q["query"],
        )

    summary = summarize(records, window_size=window_size)
    summary["k"] = k
    summary["avg_latency_ms"] = (total_latency / len(records)) * 1000 if records else 0.0
    summary["prefill_offline"] = prefill_offline
    summary["offline_thresholds"] = offline

    report = {
        "summary": summary,
        "records": records,
    }

    summary["signal_type"] = signal_type


    logging.info("=" * 60)
    logging.info("Confidence Gate 评测汇总")
    logging.info("-" * 60)
    logging.info("  总查询数: %d", summary["num_queries"])
    logging.info("  窗口大小: %d", summary["window_size"])
    logging.info("  平均延迟: %.2f ms", summary.get("avg_latency_ms", 0))
    logging.info("  冷启动预填: %s", summary.get("prefill_offline", False))
    logging.info("  信号源: %s", signal_type)

    if summary.get("offline_thresholds"):
        ot = summary["offline_thresholds"]
        logging.info("  离线阈值: p25=%.4f  p75=%.4f  (样本数=%d)", ot["p25"], ot["p75"], ot["sample_count"])

    logging.info("-" * 60)
    logging.info("  阶段分布:")
    for phase_name, phase_info in summary["phase_summary"].items():
        logging.info("    %s 阶段: 共%d条  HIGH %d  MEDIUM %d  LOW %d",
                    phase_name,
                    phase_info["num_queries"],
                    phase_info["decision_distribution"].get("HIGH", 0),
                    phase_info["decision_distribution"].get("MEDIUM", 0),
                    phase_info["decision_distribution"].get("LOW", 0))

    logging.info("-" * 60)
    logging.info("  各决策平均 Recall@%d:", summary.get("k", 3))
    for dec in ("HIGH", "MEDIUM", "LOW"):
        info = summary["recall_by_decision"].get(dec, {})
        logging.info("    %s: %.4f  (共%d条)",
                    dec,
                    info.get("avg_recall_at_k", 0.0),
                    info.get("num_queries", 0))
    logging.info("=" * 60)


    return report


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent
    DEFAULT_DATASET = ROOT_DIR / "evaluation" / "dataset" / "eval_test.json"
    
    parser = argparse.ArgumentParser(description="运行 Confidence Gate 检索评测")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--signal",
        type=str,
        default=None,
        choices=["absolute", "margin"],
        help="置信度信号源类型：absolute=Top-1分数, margin=Top1-Top2分差（不指定则两种都跑）",
    )
    parser.add_argument(
        "--prefill-offline",
        action="store_true",
        help="先用完整评测集收集 Vector Top-1 分数并预填滑动窗口，再执行门控评测",
    )
    args = parser.parse_args()

    # 确定要运行的信号源类型列表
    if args.signal is None:
        signal_types = ["absolute", "margin"]
    else:
        signal_types = [args.signal]

    # 批量执行
    reports = {}
    for signal_type in signal_types:
        logging.info("\n" + "=" * 80)
        logging.info("开始运行信号源: %s", signal_type.upper())
        logging.info("=" * 80)
        
        report = run_confidence_eval(
            k=args.k,
            dataset_path=args.dataset,
            prefill_offline=args.prefill_offline,
            signal_type=signal_type,
        )
        reports[signal_type] = report

    # 如果运行了多种信号源，输出对比总结
    if len(reports) > 1:
        logging.info("\n" + "=" * 80)
        logging.info("多信号源对比总结")
        logging.info("=" * 80)
        
        for signal_type, report in reports.items():
            summary = report["summary"]
            logging.info("\n--- %s ---", signal_type.upper())
            logging.info("  总查询数: %d", summary["num_queries"])
            logging.info("  平均延迟: %.2f ms", summary.get("avg_latency_ms", 0))
            
            logging.info("  各决策平均 Recall@%d:", summary.get("k", 3))
            for dec in ("HIGH", "MEDIUM", "LOW"):
                info = summary["recall_by_decision"].get(dec, {})
                logging.info("    %s: %.4f  (共%d条)",
                            dec,
                            info.get("avg_recall_at_k", 0.0),
                            info.get("num_queries", 0))
        
        logging.info("\n" + "=" * 80)
        logging.info("所有信号源评测完成")
        logging.info("=" * 80)

    if args.output:
        # 指定單一 signal 時輸出單份報告，方便主程式冷啟動直接讀取 records。
        output_payload = reports[signal_types[0]] if len(signal_types) == 1 else reports
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        logging.info("评测报告已写入: %s", args.output)
