import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

st.title("PDF Query Application")

query = st.text_input("Enter your debug query:")

if "vector_store" not in st.session_state:
    st.session_state.vectorstore = None
    
if st.session_state.vectorstore and query:
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    
    st.write("Relevant Documents:")
    for doc in docs:
        st.markdown(f"- {doc.page_content[:200]}...")  # Display first 200 characters of each document