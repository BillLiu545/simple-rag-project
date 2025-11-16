import os
from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel
from typing import List
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from chonkie import SemanticChunker
from PyPDF2 import PdfReader
from pymilvus import FieldSchema, CollectionSchema, DataType, utility

# Embedding model initialization
MILVUS_HOST = "host"
MILVUS_PORT = "port"
COLLECTION_NAME = "doc_chunks"

EMBED_MODEL_NAME = "your_model"
EMBED_DIM = 1024
embedder = SentenceTransformer(EMBED_MODEL_NAME)

client = OpenAI()
app = FastAPI(title="RAG API", version="1.0")

# Helper methods
def setup_milvus(collection_name="doc_chunks", dim=1024):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if not utility.has.collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        ]
        schema = CollectionSchema(fields, description="Document chunks")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

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
    
def chunk_and_embed(content: str, file_name: str):
    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  # Default model
        threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
        chunk_size=2048,                              # Maximum tokens per chunk
        min_sentences=1                              # Initial sentences per chunk
    )    
    chunks = chunker.chunk(content)
    texts = [chunk for chunk in chunks]
    embeddings = embedder.encode(texts).tolist()
    file_names = [file_name] * len(chunks)
    indexes = list(range(len(chunks)))
    return texts, embeddings, file_names, indexes

def insert_into_milvus(texts, embeddings, file_names, indexes):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    data = [embeddings, file_names, indexes, texts]
    collection.insert(data)
    collection.flush()
    print(f"Inserted {len(texts)} chunks into Milvus collection '{COLLECTION_NAME}'.")
    return len(texts)

def search_in_milvus(query, top_k=5):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["file", "chunk_index", "text"]
    )
    return results[0]

def generate_ans(query, context_chunks):
    content_text = "\n\n".join(context_chunks)
    
    prompt = f"Answer the question based on the context below:\n\nContext: {content_text}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model="Qwen2-5-VL-7B",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    answer = response.choices[0].message["content"]
    print(f"\nAnswer: {answer}")
    return answer

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    
@app.post("/ingest")
async def ingest(file: UploadFile):
    if file.filename.endswith(".pdf"):
        content = read_pdf(file.file)
    else:
        content = ( await file.read() ).decode("utf-8")
        
    texts, embeddings, file_names, indexes = chunk_and_embed(content, file.filename)
    inserted = insert_into_milvus(texts, embeddings, file_names, indexes)
    return {"message": f"Inserted {inserted} chunks from {file.filename}"}

@app.post("/query")
async def query(request: QueryRequest):
    results = search_in_milvus(request.query, top_k=request.top_k)
    chunks = [hit.entity.get("text") for hit in results]
    answer = generate_ans(request.query, chunks)
    return {
        "query": request.query,
        "answer": answer,
        "context": chunks,
        "hits": [
            {
                "score": hit.score,
                "file": hit.entity.get("file"),
                "chunk_index": hit.entity.get("chunk_index"),
            } for hit in results
        ]
    }