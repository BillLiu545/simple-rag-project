from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
vectorstore = None

# =========================
# Upload PDF endpoint
# =========================
@app.post("/upload_pdf")
async def uploadpdf(file: UploadFile):
    global vectorstore
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)

    return {"message": "PDF uploaded successfully.", "content": text[:2000]}


# =========================
# Query endpoint
# =========================
@app.post("/query")
async def query(query: str = Form(...)):
    if vectorstore is None:
        return {"response": "No documents uploaded yet."}

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Use ChatOpenAI safely
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        response = llm.predict(f"Answer the question based on the context: {context}\nQuestion: {query}")
    except Exception as e:
        response = f"Error generating response: {e}"

    return {"response": response}


# =========================
# Get raw PDF content
# =========================
@app.post("/getPdfContent")
async def get_pdf_content(file: UploadFile):
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return {"content": text}
