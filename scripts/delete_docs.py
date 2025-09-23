# delete_docs.py
from pymilvus import Collection, connections, utility

# --- 1. 配置您的 Milvus 连接 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "askme_hybrid"

# --- 2. 定义要删除的文档标识 ---
# 这个表达式您已在 Attu 和验证脚本中确认过
DOCUMENT_PREFIX_TO_DELETE = "/Users/joe/Downloads/story/2_1.txt"

# --- 3. 执行删除操作 ---
try:
    print(f"正在连接到 Milvus ({MILVUS_HOST}: {MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print(f"错误: Collection '{COLLECTION_NAME}' 不存在。")
    else:
        collection = Collection(COLLECTION_NAME)
        collection.load()

        before_count = collection.num_entities
        print(f"删除前，Collection '{COLLECTION_NAME}' 中有 {before_count} 个实体。")

        # 经过 Attu 验证，可以放心使用此表达式
        delete_expr = f'id like "{DOCUMENT_PREFIX_TO_DELETE}%"'
        print(f"准备执行删除，表达式: {delete_expr}")

        # 发送批量删除指令
        collection.delete(expr=delete_expr)
        print("删除命令已发送。")

        # Flush a collection to make deletions take effect.
        print("正在执行 flush 操作以应用删除...")
        collection.flush()
        print("Flush 完成。")

        after_count = collection.num_entities
        print(f"删除后，Collection '{COLLECTION_NAME}' 中有 {after_count} 个实体。")
        print(f"成功删除了 {before_count - after_count} 个实体。")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    connections.disconnect("default")
    print("已断开与 Milvus 的连接。")
