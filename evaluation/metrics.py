import numpy as np

def recall_at_k(retrieved_items, relevant_items, k):
    """
    計算單個查詢的 Recall@K。
    Args:
        retrieved_items (list): 系統檢索出的項目ID列表，按相關性排序。
        relevant_items (set): 與查詢相關的項目ID集合。
        k (int): 要考慮的前 K 個檢索項目。
    Returns:
        float: Recall@K 的值，範圍 [0, 1]。
    """
    if k <= 0 or not retrieved_items or not relevant_items:
        return 0.0
    
    # 取前 K 個檢索項目
    top_k_retrieved = retrieved_items[:k]
    
    # 計算在前 K 個檢索項目中有多少是相關的
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in relevant_items)
    
    # 計算 Recall@K
    
    return relevant_retrieved / len(relevant_items) if relevant_items else 0.0

def mrr(retrieved_items, relevant_items):
    """
    計算單個查詢的 MRR（Mean Reciprocal Rank）。
    Args:
        retrieved_items (list): 系統檢索出的項目ID列表，按相關性排序。
        relevant_items (set): 與查詢相關的項目ID集合。
    Returns:
        float: MRR 的值，範圍 [0, 1]。
    """
    if not retrieved_items or not relevant_items:
        return 0.0
    
    for rank, item in enumerate(retrieved_items, start=1):
        if item in relevant_items:
            return 1 / rank
    return 0.0

def ndcg_at_k(retrieved_items, relevant_items, k):
    """
    計算單個查詢的 NDCG@K（Normalized Discounted Cumulative Gain）。
    Args:
        retrieved_items (list): 系統檢索出的項目ID列表，按相關性排序。
        relevant_items (set): 與查詢相關的項目ID集合。
        k (int): 要考慮的前 K 個檢索項目。
    Returns:
        float: NDCG@K 的值，範圍 [0, 1]。
    """
    if k <= 0 or not retrieved_items or not relevant_items:
        return 0.0
    
    def dcg(items):
        """計算 DCG（Discounted Cumulative Gain）"""
        return sum((1 / np.log2(rank + 1)) for rank, item in enumerate(items, start=1) if item in relevant_items)

    def idcg_at_k(relevant_set, k_val):
        """
        計算 IDCG（Ideal DCG）
        假設所有相關項目都排在前 min(k, len(relevant_set)) 位
        """
        ideal_count = min(k_val, len(relevant_set))
        return sum(1 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))

    # 取前 K 個檢索項目
    top_k_retrieved = retrieved_items[:k]
    
    # 計算 DCG
    dcg_value = dcg(top_k_retrieved)
    
    # 計算 IDCG（理想情況下，所有相關項目都在最前面）
    idcg_value = idcg_at_k(relevant_items, k)
    
    # 計算 NDCG@K
    ndcg = dcg_value / idcg_value if idcg_value > 0 else 0.0
    
    return ndcg

def precision_at_k(retrieved_items, relevant_items, k):
    """
    計算單個查詢的 Precision@K。
    Args:
        retrieved_items (list): 系統檢索出的項目ID列表，按相關性排序。
        relevant_items (set): 與查詢相關的項目ID集合。
        k (int): 要考慮的前 K 個檢索項目。
    Returns:
        float: Precision@K 的值，範圍 [0, 1]。
    """
    if k <= 0 or not retrieved_items:
        return 0.0
    
    # 取前 K 個檢索項目
    top_k_retrieved = retrieved_items[:k]
    
    # 計算在前 K 個檢索項目中有多少是相關的
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in relevant_items)
    
    # 計算 Precision@K
    return relevant_retrieved / k if k > 0 else 0.0

def hit_rate_at_k(retrieved_items, relevant_items, k):
    """
    計算單個查詢的 Hit Rate@K。
    Args:
        retrieved_items (list): 系統檢索出的項目ID列表，按相關性排序。
        relevant_items (set): 與查詢相關的項目ID集合。
        k (int): 要考慮的前 K 個檢索項目。
    Returns:
        float: Hit Rate@K 的值，範圍 [0, 1]。
    """
    top_k = set(retrieved_items[:k])
    return 1.0 if top_k & relevant_items else 0.0

