# verify_docs.py (安全验证脚本)
from pymilvus import Collection, connections, utility

# --- 配置 ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "askme_hybrid"
DOCUMENT_PREFIX_TO_VERIFY = "/Users/joe/Downloads/story/2_1.txt"

# --- 执行验证查询 ---
try:
    print(f"正在连接到 Milvus ({MILVUS_HOST}: {MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print(f"错误: Collection '{COLLECTION_NAME}' 不存在。")
    else:
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 构造与删除操作完全相同的表达式
        query_expr = f'id like "{DOCUMENT_PREFIX_TO_VERIFY}%"'
        print(f"准备执行验证查询，表达式: {query_expr}")

        # 执行查询，只返回主键 id 字段以提高效率
        results = collection.query(expr=query_expr, output_fields=["id"])  # 我们只需要id来验证

        print(f"\n查询完成！根据您的表达式，共匹配到 {len(results)} 个实体。")

        if results:
            print("以下是匹配到的部分实体ID示例:")
            for item in results[:5]:  # 显示最多5个作为示例
                print(f"- {item['id']}")

        print("\n这只是一个验证查询，没有删除任何数据。")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    connections.disconnect("default")
    print("已断开与 Milvus 的连接。")
