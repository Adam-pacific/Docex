import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------- ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing")
    st.stop()

# ---------------- STREAMLIT ----------------
st.set_page_config(page_title="DocEx", page_icon="📄")
st.title("📄 DocEx – Chat with your Document")

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# ---------------- SESSION STATE ----------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload a PDF", type="pdf")

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # 🔥 THIS IS THE KEY FIX
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vectordb = Chroma.from_documents(
        docs,
        embeddings
    )

    st.success(f"Indexed {len(docs)} chunks")

# ---------------- CHAT ----------------
if st.session_state.vectordb:

    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant answering strictly from the document context.

        Context:
        {context}

        Question:
        {question}

        Answer clearly and precisely.
        """
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.chat_input("Ask something from the document")

    if query:
        st.session_state.chat.append(("user", query))

        with st.spinner("Thinking..."):
            response = chain.invoke(query)

        st.session_state.chat.append(("ai", response))

    for role, msg in st.session_state.chat:
        st.chat_message("user" if role == "user" else "assistant").write(msg)

else:
    st.info("Upload a PDF to start chatting")
