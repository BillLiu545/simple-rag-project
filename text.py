import os

from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = "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"

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_chat_openai():
    # 3️⃣ Initialize the ChatOpenAI model
    llm = ChatOpenAI(model_name="Qwen2-5-VL-7B")  # or another model you have access to

    # 4️⃣ Provide a simple prompt
    prompt = "Say hello in a friendly way."
    response = llm.predict(prompt)

    print("ChatOpenAI Response:")
    print(response)

def test_embeddings():
    # 5️⃣ Initialize embeddings
    embeddings = OpenAIEmbeddings()
    sample_text = ["Hello world!", "How are you?"]
    vectors = embeddings.embed_documents(sample_text)

    print("Embeddings:")
    print(vectors)

def test_text_splitter():
    text = "This is a long text that needs to be split into chunks for LangChain."
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_text(text)

    print("Text chunks:")
    print(chunks)

if __name__ == "__main__":
    test_chat_openai()
    test_embeddings()
    test_text_splitter()