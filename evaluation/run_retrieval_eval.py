
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.knowledge_base import KnowledgeBase
from evaluation.metrics import recall_at_k, mrr, ndcg_at_k
import json
import logging
import argparse
import time

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [
                        logging.FileHandler(log_dir / "eval.log",encoding="utf-8"),
                        ]
                    )

from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def run_eval(method: str , k: int = 5):

    # 加載測試數據和知識庫
    kb = KnowledgeBase()
    kb.ensure_connected()
    with open("evaluation/dataset/eval_test.json", "r") as f:
        queries = json.load(f)
    
    # 初始化評測指標
    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_latency = 0.0
    single_query_latency = 0.0
    num_queries = len(queries)

    # 開始評測
    for index , q in enumerate(queries , 1):
        retrieved_items = []

        if method == "vector":
            vector_results, vector_time = kb.vector_search(q["query"], k=k)
            retrieved_items = [(doc.metadata.get("mysql_id"), score) for doc, score in vector_results]
            total_latency += vector_time
            single_query_latency = vector_time

            """
            if index == 17:
                print(f"=== 向量检索第 {index} 条查询调试 ===")
                print("查询:", q["query"])
                print("期望 ID:", q.get("expected_doc_ids"))
                print("检索到的 ID 和分数:", [(doc.metadata.get("mysql_id"), score) for doc, score in vector_results])
                print("检索到的文本前 50 字:", [doc.page_content[:50] for doc, _ in vector_results])
            """
        elif method == "bm25":
            bm25_results, bm25_time = kb.bm25_search(q["query"], k=k)
            retrieved_items = [(doc.metadata.get("mysql_id"), score) for doc, score in bm25_results]
            total_latency += bm25_time
            single_query_latency = bm25_time

            """
            if index == 7:
                print(f"=== BM25 第 {index} 条查询调试 ===")
                print("查询:", q["query"])
                print("期望 ID:", q.get("expected_doc_ids"))
                print("检索到的 ID 和分数:", [(doc.metadata.get("mysql_id"), score) for doc, score in bm25_results])
                print("检索到的文本前 50 字:", [doc.page_content[:50] for doc, _ in bm25_results])

            """
        elif method == "hybrid":
            start_time = time.time()

            vector_results, _ = kb.vector_search(q["query"], k=k*2)
            bm25_results, _ = kb.bm25_search(q["query"], k=k*2)
            hybrid_list = kb.hybrid_search(vector_results, bm25_results, k=k)

            end_time = time.time()
            total_latency += (end_time - start_time)
            single_query_latency = end_time - start_time

            # 提取 mysql_id，注意 hybrid_list 里的元素是字典，包含 'doc' 和 'rrf_score' 等键
            retrieved_items = [
                (item["doc"].metadata.get("mysql_id"), item["rrf_score"]) 
                for item in hybrid_list 
                if item["doc"].metadata.get("mysql_id") is not None
                ]
        else:
            logging.error(f"未知的檢索方法: {method}，將跳過該查詢。")
            return None
        
        # 計算評測指標
        relevant_items = set(q.get("expected_doc_ids", []))
        # 提取檢索結果中的文檔 ID，並去重
        pred_ids = []
        seen = set()
        for item in retrieved_items:
            doc_id = item[0]
            if doc_id is not None and doc_id not in seen:
                pred_ids.append(doc_id)
                seen.add(doc_id)

        #print(f"DEBUG: pred_ids={pred_ids}, relevant_items={relevant_items}, len(relevant)={len(relevant_items)}")

        recall = recall_at_k(pred_ids, relevant_items, k)
        total_recall += recall
        logging.info(f"[{index}/{num_queries}]Recall@{k} for query '{q['query']}': {recall:.4f}")

        mrr_val = mrr(pred_ids, relevant_items)
        total_mrr += mrr_val
        logging.info(f"[{index}/{num_queries}]MRR for query '{q['query']}': {mrr_val:.4f}")

        ndcg = ndcg_at_k(pred_ids, relevant_items, k)
        total_ndcg += ndcg
        logging.info(f"[{index}/{num_queries}]NDCG@{k} for query '{q['query']}': {ndcg:.4f}")

        logging.info(f"[{index}/{num_queries}]檢索延遲: {single_query_latency:.4f} 秒")
    # 輸出平均評測指標
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    avg_mrr = total_mrr / num_queries if num_queries > 0 else 0
    avg_ndcg = total_ndcg / num_queries if num_queries > 0 else 0
    avg_latency_ms = ( total_latency / num_queries ) * 1000 if num_queries > 0 else 0

    logging.info(f"平均 Recall@{k}: {avg_recall:.4f}")
    logging.info(f"平均 MRR: {avg_mrr:.4f}")
    logging.info(f"平均 NDCG@{k}: {avg_ndcg:.4f}")
    logging.info(f"平均檢索延遲: {avg_latency_ms:.4f} 毫秒")

    return {
        "method" : method,
        "k" : k,
        "num_queries" : num_queries,
        "avg_recall" : avg_recall,
        "avg_mrr" : avg_mrr,
        "avg_ndcg" : avg_ndcg,
        "avg_latency_ms" : avg_latency_ms
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="運行檢索評測")
    parser.add_argument("--method",
                        type=str, 
                        default="all", 
                        choices=["vector", "bm25", "hybrid","all"])
    parser.add_argument("--k",
                        type=int,
                        default=3)
    args = parser.parse_args()

    methods = ["vector", "bm25", "hybrid"] if args.method == "all" else [args.method]
    all_results = []

    for method in methods:
        logging.info(f"開始評測 , 評測方法: {method}")
        result = run_eval(method, k=args.k)
        if result:
            all_results.append(result)

    if all_results:
        logging.info(f"\\n{'='*60}")
        logging.info(f"所有評估方法對比總結 (K={args.k})")
        logging.info(f"{'='*60}")
        logging.info(f"{'方法':<10} {'Recall@K':<12} {'MRR':<12} {'NDCG@K':<12} {'延遲(ms)':<12}")
        logging.info(f"{'='*60}")
        for r in all_results:
            logging.info(f"{r['method']:<10} {r['avg_recall']:<12.4f} {r['avg_mrr']:<12.4f} {r['avg_ndcg']:<12.4f} {r['avg_latency_ms']:<12.4f}")
        logging.info(f"{'='*60}")
