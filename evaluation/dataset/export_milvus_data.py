import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from pymilvus.exceptions import MilvusException

from pymilvus import connections, Collection, utility

connections.connect("default", host="localhost", port="19530")

collection_name = "customer_service"
col = Collection(collection_name)

# 安全加载集合
if utility.load_state(collection_name) != "Loaded":
    col.load()

# 获取所有字段名
field_names = [field.name for field in col.schema.fields]
print(f"字段列表: {field_names}")

# 查询所有实体（如果你的数据量超过1万，请分批）
limit = col.num_entities
results = col.query(expr="pk >= 0", limit=limit, output_fields=field_names)

# 将向量缩短为前5个值 + 维度标记，避免 JSON 过大
processed = []
for row in results:
    new_row = {}
    for k, v in row.items():
        if k == "vector":
            # 只保留前5个向量值和总长度
            new_row[k] = {
                "dims": len(v),
                "sample": v[:5]
            }
        else:
            new_row[k] = v
    processed.append(new_row)

# 保存到文件
output_file = Path("evaluation/dataset/milvus_export.json")
output_file.parent.mkdir(exist_ok=True, parents=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"导出完成，共 {len(processed)} 条记录 -> {output_file}")