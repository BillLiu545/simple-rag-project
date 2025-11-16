
from pymilvus import FieldSchema, CollectionSchema, DataType, utility
from pymilvus import connections, Collection

# Embedding model initialization
MILVUS_HOST = "host"
MILVUS_PORT = "port"
COLLECTION_NAME = "doc_chunks"

EMBED_DIM=4096

# Helper methods
def setup_milvus(collection_name="doc_chunks", dim=EMBED_DIM):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        print(f"Existing collection '{collection_name}' dropped.")
        utility.drop_collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    ]
    schema = CollectionSchema(fields, description="Document chunks")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created with embedding dimension '{dim}.")
        
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
    )
    print(f"Index has been created for collection '{collection_name}'.")
    collection.load()
    return collection


if __name__ == '__main__':
    print("start>>>>>>>>>>")
    setup_milvus()
