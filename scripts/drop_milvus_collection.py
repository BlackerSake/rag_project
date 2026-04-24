import os
from pymilvus import connections, utility
from dotenv import load_dotenv

load_dotenv('.env')
milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
if milvus_uri.startswith("tcp://"):
    milvus_uri = f"http://{milvus_uri[len('tcp://'):]}"

connections.connect("milvus_dropper", uri=milvus_uri)
collection_name = "customer_service"

if utility.has_collection(collection_name, using="milvus_dropper"):
    utility.drop_collection(collection_name, using="milvus_dropper")
    print(f"集合 {collection_name} 已通过 utility 彻底删除")
else:
    print(f"集合 {collection_name} 不存在，无需删除")
