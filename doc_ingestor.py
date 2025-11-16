from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from chunklet import Chunklet
from PyPDF2 import PdfReader
from pymilvus import FieldSchema, CollectionSchema, DataType, utility

# Embedding model initialization
MILVUS_HOST = "host"
MILVUS_PORT = "port"
COLLECTION_NAME = "doc_chunks"
EMBED_DIM=4096

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
    
def chunk_text(text, max_sentences=2):
    sentences = text.split(". ")
    chunks = []
    for i in range (0, len(sentences), max_sentences):
        chunks.append(". ".join(sentences[i:i+max_sentences]))
    return chunks

import openai
import os

def generate_openai_embedding(text, model="your_model_here"):
    client = openai.OpenAI(api_key="your_api_key", base_url="your_base_url")
    response = client.embeddings.create(model=model, input="text", dimensions=EMBED_DIM)
    return response.data[0].embedding

def insert_chunks(file_path):
    text = read_pdf(file_path)
    chunks = chunk_text(text)
    
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection=Collection(COLLECTION_NAME)
    collection.load()
    
    for idx, chunk in enumerate(chunks):
        embedding = generate_openai_embedding(chunk)
        collection.insert([[embedding], [file_path], [idx+1], [chunk]])
    
if __name__ == '__main__':
    print("start>>>>>>>>>>")
    insert_chunks("/your_pdf.pdf")