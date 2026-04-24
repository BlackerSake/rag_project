from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")
collection_name = "customer_service"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"集合 {collection_name} 已通过 utility 彻底删除")
else:
    print(f"集合 {collection_name} 不存在，无需删除")