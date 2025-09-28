# verify_docs.py (safe verification script)
from pymilvus import Collection, connections, utility

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "askme_hybrid"
DOCUMENT_PREFIX_TO_VERIFY = "/Users/joe/Downloads/story/2_1.txt"

# --- Execute verification query ---
try:
    print(f"Connecting to Milvus ({MILVUS_HOST}: {MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
    else:
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # Construct expression identical to deletion operation
        query_expr = f'id like "{DOCUMENT_PREFIX_TO_VERIFY}%"'
        print(f"Ready to execute verification query, expression: {query_expr}")

        # Execute query, only return primary key id field for efficiency
        results = collection.query(
            expr=query_expr, output_fields=["id"]
        )  # We only need id for verification

        print(f"\nQuery completed! Found {len(results)} matching entities.")

        if results:
            print("Sample of matching entity IDs:")
            for item in results[:5]:  # Show maximum 5 as examples
                print(f"- {item['id']}")

        print("\nThis is only a verification query, no data was deleted.")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    connections.disconnect("default")
    print("Disconnected from Milvus.")
