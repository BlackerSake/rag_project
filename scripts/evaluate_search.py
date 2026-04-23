import sys
import os
import io

# 将当前项目的根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

from data.knowledge_base import KnowledgeBase

class SearchEvaluator:
    def __init__(self):
        """初始化搜索评估器"""
        self.knowledge_base = KnowledgeBase()
        # 加载已有的评估数据
        self.knowledge_base.load_evaluation_data()
    
    def evaluate_sample_queries(self):
        """评估样例查询"""
        # 定义测试查询和相关文档
        test_cases = [
            {
                "query": "如何退货",
                "relevant_docs": ["退货", "退款", "退货流程"]
            },
            {
                "query": "客服上班时间",
                "relevant_docs": ["客服", "上班时间", "工作时间"]
            },
            {
                "query": "订单配送时间",
                "relevant_docs": ["配送", "物流", "到货时间"]
            },
            {
                "query": "如何使用优惠券",
                "relevant_docs": ["优惠券", "使用方法", "折扣"]
            }
        ]
        
        print("=== 开始评估混合检索 ===")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试用例 {i}: {test_case['query']}")
            print(f"相关文档关键词: {test_case['relevant_docs']}")
            
            try:
                # 执行搜索并评估
                results = self.knowledge_base.search(
                    test_case['query'],
                    k=5,
                    evaluate=True,
                    relevant_docs=test_case['relevant_docs']
                )
                
                print(f"搜索结果数量: {len(results)}")
                for j, (doc, score) in enumerate(results, 1):
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  结果 {j} (分数: {score:.4f}): {content_preview}")
                    
            except Exception as e:
                print(f"  评估失败: {str(e)}")
        
        print("\n=== 评估完成 ===")
    
    def print_evaluation_summary(self):
        """打印评估摘要"""
        summary = self.knowledge_base.get_evaluation_summary()
        print("\n=== 评估摘要 ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    evaluator = SearchEvaluator()
    evaluator.evaluate_sample_queries()
    evaluator.print_evaluation_summary()
