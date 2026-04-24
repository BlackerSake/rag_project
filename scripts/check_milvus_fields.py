import sys
import os
# 将当前项目的根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import Collection, utility, connections
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('.env')

class MilvusFieldChecker:
    def __init__(self):
        """初始化Milvus字段检查器"""
        self.collection_name = "customer_service"
        self.milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
        self.connect()
    
    def connect(self):
        """连接到Milvus"""
        try:
            if not connections.has_connection("default"):
                print(f"正在连接到Milvus: {self.milvus_uri}")
                connections.connect("default", uri=self.milvus_uri)
                print("Milvus连接成功")
            else:
                print("Milvus已有连接，无需重复连接")
        except Exception as e:
            print(f"连接Milvus失败: {str(e)}")
    
    def check_collection(self):
        """检查集合是否存在"""
        if utility.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 存在")
            return True
        else:
            print(f"集合 {self.collection_name} 不存在")
            return False
    
    def check_fields(self):
        """检查集合字段"""
        if not self.check_collection():
            return
        
        try:
            collection = Collection(self.collection_name)
            schema = collection.schema
            print("\n集合字段:")
            for field in schema.fields:
                print(f"- {field.name} (类型: {field.dtype})")
            collection.release()
        except Exception as e:
            print(f"检查字段失败: {str(e)}")
    
    def check_sample_data(self, limit=5):
        """检查样本数据"""
        if not self.check_collection():
            return
        
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            # 查询前几条数据
            results = collection.query(
                expr="",
                output_fields=["*"],
                limit=limit
            )
            
            print(f"\n样本数据 ({limit} 条):")
            for i, result in enumerate(results):
                print(f"\n记录 {i+1}:")
                print(f"  ID: {result.get('id')}")
                print(f"  元数据: {result.get('metadata', {})}")
                if 'metadata' in result:
                    print(f"  包含intent_id: {'intent_id' in result['metadata']}")
            
            collection.release()
        except Exception as e:
            print(f"检查样本数据失败: {str(e)}")
    
    def drop_collection(self):
        """删除集合（谨慎使用）"""
        if utility.has_collection(self.collection_name):
            confirm = input(f"确定要删除集合 {self.collection_name} 吗？(y/n): ")
            if confirm.lower() == 'y':
                utility.drop_collection(self.collection_name)
                print(f"集合 {self.collection_name} 已删除")
            else:
                print("取消删除操作")
        else:
            print(f"集合 {self.collection_name} 不存在")

if __name__ == "__main__":
    checker = MilvusFieldChecker()
    
    print("=== Milvus集合检查 ===")
    checker.check_collection()
    
    print("\n=== 字段检查 ===")
    checker.check_fields()
    
    print("\n=== 样本数据检查 ===")
    checker.check_sample_data()
   