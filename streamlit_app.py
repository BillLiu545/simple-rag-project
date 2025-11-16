import streamlit as st
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import openai
import json
import matplotlib.pyplot as plt
# Embedding model initialization
MILVUS_HOST = "host"
MILVUS_PORT = "port"
COLLECTION_NAME = "doc_chunks"
EMBED_DIM = 4096
OPENAI_API_KEY = "my_api_key"

#Client initialization
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url = "client_base_url")
chat_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url = "chat_client_url")

# Helper method to search relevant chunks
def search_chunks(query_embedding, top_k=5):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field = "embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text","file","chunk_index"]
    )
    return [hit.entity.get("text") for hit in results[0]]

# Helper method to generate embedding
def generate_embedding(text, model="Qwen3-Embedding-8B"):
    query_embedding = client.embeddings.create(model=model, input=text, dimensions=EMBED_DIM).data[0].embedding
    return query_embedding

# helper method to generate answer
def generate_ans(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
    Answer the user's question based on the context below.
    If the answer can be represented as a chart, return JSON with this schema:
    {{
      "answer": "string",
      "chart": {{
        "type": "bar" | "pie" | "line",
        "labels": [list of strings],
        "values": [list of numbers]
      }}
    }}
    If no chart is needed, return JSON with only the "answer".
    
    Context: {context_text}
    Question: {query}
    """
    response = chat_client.chat.completions.create(
        model="your_model",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_output = response.choices[0].message.content
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {"answer": raw_output, "chart": None}
    return parsed

# Separate method to create chart
def create_chart(chart_data):
    if not chart_data:
        return
    chart_type=chart_data["type"]
    labels=chart_data["labels"]
    
    try:
        values= [float(v) for v in chart_data["values"]]
    except KeyError:
        return
    
    fig, ax = plt.subplots()
    
    if chart_type == "bar":
        ax.bar(labels, values)
    elif chart_type == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%')
    elif chart_type == "line":
        ax.plot(labels, values, marker="o")
    
    st.pyplot(fig)

# Streamlit UI
st.title("RAG Chatbot with Milvus")
st.markdown("Ask a question about your ingested document!")

user_query = st.chat_input("Type your question here...")

if user_query:
    query_embedding=generate_embedding(user_query)
    relevant_chunks=search_chunks(query_embedding)
    response=generate_ans(user_query, relevant_chunks)
        
    st.write(response["answer"])
        
    if "chart" in response and response["chart"]:
        create_chart(response["chart"])