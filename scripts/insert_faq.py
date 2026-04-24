import pymysql, json
"""
将faq_data.json文件中的数据插入到customer_service_db数据库的faq表中
"""

with open('./faq_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

conn = pymysql.connect(host='127.0.0.1', user='root', password='133466', database='customer_service_db')
cursor = conn.cursor()
for item in data:
    cursor.execute(
        "INSERT INTO faq (domain, intent, action, question, answer) VALUES (%s, %s, %s, %s, %s)",
        (item.get('domain', ''), item['intent_id'], item.get('action', ''), item['question'], item['answer'])
    )
conn.commit()
conn.close()