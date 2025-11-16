import os
import sys  
from PyPDF2 import PdfReader
from chonkie import SemanticChunker
from pymilvus  import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def read_pdf(file_path):
    try:        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""
    
def read_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

# Setup Milvus collection  
def setup_milvus(collection_name="doc_chunks", dim=1024):
    connections.connect("default", host="aipc", port="8787")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields, description="Document chunks")
    if collection_name not in utility.list_collections():
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    return collection

# Embedding model initialization
EMBED_MODEL_NAME = "your_model"
EMBED_DIM = "your_dim"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Chunk the content 
def ingest_file(file_path):
    if file_path.endswith(".pdf"):
        content = read_pdf(file_path)
    else:
        content = read_text(file_path)
    if not content:
        return [], [], [], []
    print(f"Content of {file_path}:\n{content[:100]}...\n ")  # Display first 100 characters of the content
    # Basic initialization with default parameters
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  # Default model
        threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
        chunk_size=2048,                              # Maximum tokens per chunk
        min_sentences=1                              # Initial sentences per chunk
    )    
    chunks = chunker.chunk(content)

    texts = [chunk for chunk in chunks]
    embeddings = embedder.encode(texts).tolist()  # Convert embeddings to list for Milvus
    file_names = [file_path] * len(chunks)
    chunk_indexes = list(range(len(chunks)))
    
    return texts, embeddings, file_names, chunk_indexes

# Insert the content into Milvus        
def insert_into_milvus(texts, embeddings, file_names, chunk_indexes, collection_name="document_chunks"):
    collection = Collection(collection_name)
    data = [embeddings, file_names, chunk_indexes, texts]
    collection.insert(data)
    collection.flush()
    print(f"Inserted {len(texts)} chunks into Milvus collection '{collection_name}'.")
    return collection

def search_in_milvus(query, collection_name="doc_chunks", top_k=5):
    collection = Collection(collection_name)
    collection.load()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["file", "chunk_index", "text"]
    )
    retrieved_chunks = []
    print(f"\n Search results for query '{query}':")
    for hit in results[0]:
        retrieved_chunks.append(hit.entoty.get["text"])
        print(f"-(Score: {hit.score:.4f}) File: {hit.entity.get('file')}, Chunk Index: {hit.entity.get('chunk_index')}")
        print(f" {hit.entity.get('text')[:200]}...\n")
        
    return retrieved_chunks

def generate_ans(query, context_chunks):
    client = OpenAI()
    content_text = "\n\n".join(context_chunks)
    
    prompt = f"Answer the question based on the context below:\n\nContext: {content_text}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="your_model",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    answer = response.choices[0].message["content"]
    print(f"\nAnswer: {answer}")
    return answer

def process_files_in_folder(folder_path, collection_name="document_chunks"):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswidth((".pdf", ".txt")):
                file_path = os.path.join(root,file)
                print(f"Processing {file_path}")
                texts, embeddings, file_names, chunk_indexes = ingest_file(file_path)
                if texts:
                    insert_into_milvus(texts, embeddings, file_names, chunk_indexes, collection_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path> [query]")
        sys.exit(1)

    folder_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else None

    setup_milvus(collection_name="doc_chunks", dim=EMBED_DIM)
    process_files_in_folder(folder_path, collection_name="doc_chunks")

    if query:
        search_in_milvus(query, collection_name="doc_chunks")