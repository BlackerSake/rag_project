import sys
import os
import logging
# 将当前项目的根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

from data.knowledge_base import KnowledgeBase

"""
从MySQL导入FAQ到Milvus
"""
# 加载环境变量
load_dotenv('.env')
logger = logging.getLogger(__name__)

class MySQLFAQImporter:
    def __init__(self, mysql_config, batch_size=100):
        """
        初始化MySQL FAQ导入器
        
        Args:
            mysql_config (dict): MySQL连接配置
            batch_size (int): 批处理大小
        """
        self.mysql_config = mysql_config
        self.batch_size = batch_size
        self.knowledge_base = KnowledgeBase()
    
    def connect_mysql(self):
        """连接MySQL数据库"""
        try:
            connection = pymysql.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user'],
                password=self.mysql_config['password'],
                database=self.mysql_config['database'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"成功连接到MySQL数据库: {self.mysql_config['database']}")
            return connection
        except Exception as e:
            logger.error(f"MySQL连接失败: {str(e)}")
            return None
    
    def import_faq(self):
        """从MySQL导入FAQ到Milvus"""
        connection = self.connect_mysql()
        if not connection:
            return 0
        
        try:
            with connection.cursor() as cursor:
                # 获取FAQ总数
                cursor.execute("SELECT COUNT(*) as total FROM faq")
                total = cursor.fetchone()['total']
                logger.info(f"从{self.mysql_config['database']}数据库发现 {total} 条FAQ记录")
                
                # 分批读取数据
                total_es_imported = 0
                total_milvus_imported = 0
                total_batches = (total + self.batch_size - 1) // self.batch_size
                
                with tqdm(total=total_batches, desc="导入FAQ批次", unit="批") as pbar:
                    for offset in range(0, total, self.batch_size):
                        # 读取数据
                        cursor.execute(
                            "SELECT id,intent, action, question, answer FROM faq LIMIT %s OFFSET %s",
                            (self.batch_size, offset)
                        )
                        faq_records = cursor.fetchall()
                        
                        if not faq_records:
                            break
                        
                        # 处理数据
                        documents = []
                        for record in faq_records:
                            try:
                                # 增强语义：intent, action到问题中
                                enhanced_question = f"意图: {record['intent']} 动作: {record['action']} 问题: {record['question']}"
                                content = f"问题: {enhanced_question}\n答案: {record['answer']}"
                                
                                # 创建Document对象，使用MySQL ID作为元数据
                                # 确保包含所有必需的字段
                                metadata = {
                                    "type": "faq",
                                    "mysql_id": record['id'],
                                    "intent_id": record['intent'],
                                    "original_question": record['question'],
                                    "action": record['action']
                                }
                                # 确保所有字段都有值
                                for key, value in metadata.items():
                                    if value is None:
                                        metadata[key] = ""
                                
                                doc = Document(
                                    page_content=content,
                                    metadata=metadata
                                )
                                doc.metadata["doc_id"] = self.knowledge_base._generate_doc_id(doc)
                                documents.append(doc)
                            except Exception as e:
                                print(f"处理FAQ记录失败 (ID: {record.get('id', 'unknown')}): {str(e)}")
                                continue
                        
                        if documents:
                            # 确保知识库已连接
                            self.knowledge_base.ensure_connected()
                            # 批量添加到知识库
                            try:
                                es_success = False
                                milvus_success = False

                                # 先尝试添加到Elasticsearch（如果可用）
                                if self.knowledge_base.elasticsearch_store:
                                    try:
                                        self.knowledge_base.elasticsearch_store.add_documents(documents)
                                        print(f"成功添加 {len(documents)} 条文档到Elasticsearch")
                                        es_success = True
                                        total_es_imported += len(documents)
                                    except Exception as e_es:
                                        print(f"添加到Elasticsearch失败: {str(e_es)}")
                                
                                # 再尝试添加到Milvus（如果可用）
                                if self.knowledge_base.vector_store:
                                    try:
                                        self.knowledge_base.vector_store.add_documents(documents)
                                        print(f"成功添加 {len(documents)} 条FAQ到Milvus")
                                        milvus_success = True
                                        total_milvus_imported += len(documents)
                                    except Exception as e_milvus:
                                        print(f"添加到Milvus失败: {str(e_milvus)}")
                                else:
                                    print("Milvus未连接，跳过Milvus添加")
                            except Exception as e:
                                print(f"添加文档失败: {str(e)}")
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            "Milvus": total_milvus_imported,
                            "ES": total_es_imported
                        })
                
                print(
                    f"导入完成，Milvus成功导入 {total_milvus_imported} 条FAQ，"
                    f"Elasticsearch成功导入 {total_es_imported} 条FAQ"
                )
                return total_milvus_imported
                
        except Exception as e:
            print(f"导入过程中出现错误: {str(e)}")
            return 0
        finally:
            if connection:
                connection.close()

if __name__ == "__main__":
    # MySQL连接配置（请根据实际情况修改）
    mysql_config = {
        'host': '127.0.0.1',      # MySQL主机地址
        'user': 'root',  # MySQL用户名
        'password': '133466',  # MySQL密码
        'database': 'customer_service_db'   # 数据库名称
    }
    
    # 初始化导入器
    importer = MySQLFAQImporter(mysql_config, batch_size=100)
    
    # 执行导入
    importer.import_faq()
