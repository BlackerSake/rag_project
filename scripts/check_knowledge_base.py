import sys
import os
import io 
# 将当前项目的根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

from data.knowledge_base import KnowledgeBase

class KnowledgeBaseChecker:
    def __init__(self):
        """初始化知识库检查器"""
        self.knowledge_base = KnowledgeBase()
    
    def check_collection(self):
        """检查集合信息"""
        try:
            from pymilvus import utility
            collection_name = "customer_service"
            
            if utility.has_collection(collection_name):
                print(f"集合 {collection_name} 存在")
                # 获取集合统计信息
                from pymilvus import Collection
                collection = Collection(collection_name)
                collection.load()
                print(f"集合中的文档数量: {collection.num_entities}")
                collection.release()
            else:
                print(f"集合 {collection_name} 不存在")
        except Exception as e:
            print(f"检查集合失败: {str(e)}")
    
    def search_sample(self, query="你好", k=5):
        """搜索知识库示例"""
        try:
            print(f"搜索知识库，查询: {query}")
            results = self.knowledge_base.search(query, k=k)
            print(f"找到 {len(results)} 条结果")
            
            for i, (doc, score) in enumerate(results):
                print(f"\n结果 {i+1} (分数: {score:.4f}):")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"元数据: {doc.metadata}")
        except Exception as e:
            print(f"搜索失败: {str(e)}")
    
    def check_fields(self):
        """检查集合字段"""
        try:
            from pymilvus import Collection, utility
            collection_name = "customer_service"
            
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                schema = collection.schema
                print("集合字段:")
                for field in schema.fields:
                    print(f"- {field.name} (类型: {field.dtype})")
                collection.release()
            else:
                print(f"集合 {collection_name} 不存在")
        except Exception as e:
            print(f"检查字段失败: {str(e)}")

if __name__ == "__main__":
    checker = KnowledgeBaseChecker()
    
    print("=== 知识库检查 ===")
    checker.check_collection()
    
    print("\n=== 集合字段检查 ===")
    checker.check_fields()
    
    print("\n=== 搜索示例 ===")
    checker.search_sample()
    
    print("\n=== 搜索特定意图 ===")
    checker.search_sample("退货", k=3)
