

import json
import hashlib
import logging
from datetime import datetime

from data.kb_config import KBConfig

logger = logging.getLogger(__name__)

class SearchLogger:
    def __init__(self, config: KBConfig, thread_pool, evaluator):
        self.config = config
        self._thread_pool = thread_pool
        self._retrieval_evaluator = evaluator
        self.retrieval_evaluation_events = []
        

    def _async_log_search(self, query, result_count, response_time, from_cache, vector_time=0, bm25_time=0, merge_time=0):
        """异步记录搜索日志"""
        def log_task():
            try:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "result_count": result_count,
                    "response_time": response_time,
                    "from_cache": from_cache,
                    "vector_time": vector_time,
                    "bm25_time": bm25_time,
                    "merge_time": merge_time
                }
                # 记录到日志文件
                logger.info("Search: %s", json.dumps(log_entry, ensure_ascii=False))
            except Exception as e:
                logger.error("异步记录搜索日志失败: %s", e)

        self._thread_pool.submit(log_task)

    def _async_evaluate_retrieval(self, query, results, k, relevant_docs,
                                  expected_intent_id, response_time, from_cache,
                                  evaluation_context):
        """异步记录主系统检索评测事件。"""
        if not self.config.retrieval_eval_enabled:
            return

        def evaluate_task():
            try:
                event = self._retrieval_evaluator.evaluate_results(
                    query,
                    results,
                    k=k,
                    relevant_docs=relevant_docs,
                    expected_intent_id=expected_intent_id,
                    response_time=response_time,
                    from_cache=from_cache,
                    context=evaluation_context,
                )
                self.retrieval_evaluation_events.append(event)
                self._retrieval_evaluator.log_event(event)
            except Exception as e:
                logger.error("异步检索评测失败: %s", e)

        self._thread_pool.submit(evaluate_task)

    def get_retrieval_evaluation_summary(self):
        """返回当前进程内已记录检索评测事件的均值摘要。"""
        if not self.retrieval_evaluation_events:
            return {"message": "暂无检索评测数据"}

        metric_names = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")
        summary = {
            "total_queries": len(self.retrieval_evaluation_events),
            "labeled_queries": 0,
            "intent_labeled_queries": 0,
            "unlabeled_queries": 0,
        }
        metric_values = {name: [] for name in metric_names}

        for event in self.retrieval_evaluation_events:
            status = event.get("metric_status", "unlabeled")
            summary[f"{status}_queries"] = summary.get(f"{status}_queries", 0) + 1
            metrics = event.get("metrics", {})
            for name in metric_names:
                value = metrics.get(name)
                if value is not None:
                    metric_values[name].append(value)

        for name, values in metric_values.items():
            summary[f"average_{name}"] = sum(values) / len(values) if values else None

        return summary
