# delete_docs.py
from pymilvus import Collection, connections, utility

# --- 1. Configure your Milvus connection ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "askme_hybrid"

# --- 2. Define document identifier to delete ---
# This expression has been verified in Attu and validation scripts
DOCUMENT_PREFIX_TO_DELETE = "/Users/joe/Downloads/story/2_1.txt"

# --- 3. Execute deletion operation ---
try:
    print(f"Connecting to Milvus ({MILVUS_HOST}: {MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
    else:
        collection = Collection(COLLECTION_NAME)
        collection.load()

        before_count = collection.num_entities
        print(f"Before deletion, '{COLLECTION_NAME}' has {before_count} entities.")

        # Verified through Attu, safe to use this expression
        delete_expr = f'id like "{DOCUMENT_PREFIX_TO_DELETE}%"'
        print(f"Ready to execute deletion, expression: {delete_expr}")

        # Send batch deletion command
        collection.delete(expr=delete_expr)
        print("Deletion command sent.")

        # Flush a collection to make deletions take effect.
        print("Executing flush operation to apply deletions...")
        collection.flush()
        print("Flush completed.")

        after_count = collection.num_entities
        print(f"After deletion, '{COLLECTION_NAME}' has {after_count} entities.")
        print(f"Successfully deleted {before_count - after_count} entities.")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    connections.disconnect("default")
    print("Disconnected from Milvus.")
