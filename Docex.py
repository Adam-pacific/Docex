import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
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

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="DocEx", page_icon="📄", layout="wide")
st.title("📄 DocEx — Chat with Documents, Websites & General AI")

st.caption(
    "DocEx answers questions from your documents and websites using RAG, "
    "and can also respond to general questions using its own knowledge."
)

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# ---------------- SESSION STATE ----------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
st.sidebar.header("📥 Knowledge Source")

source_type = st.sidebar.radio(
    "Choose source",
    ["Upload PDF", "Website URL"]
)

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# ---------------- LOAD PDF ----------------
if source_type == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        docs = text_splitter.split_documents(raw_docs)

        st.session_state.vectordb = Chroma.from_documents(
            docs,
            embeddings
        )

        st.sidebar.success(f"✅ Indexed {len(docs)} chunks from PDF")

# ---------------- LOAD WEBSITE ----------------
if source_type == "Website URL":
    url = st.sidebar.text_input("Enter website URL")

    if st.sidebar.button("Load Website") and url:
        loader = WebBaseLoader(url)
        raw_docs = loader.load()

        docs = text_splitter.split_documents(raw_docs)

        st.session_state.vectordb = Chroma.from_documents(
            docs,
            embeddings
        )

        st.sidebar.success(f"✅ Indexed {len(docs)} chunks from website")

# ---------------- PROMPTS ----------------
rag_prompt = ChatPromptTemplate.from_template(
    """
    You are DocEx, an AI assistant.
    Answer the question using the context below.
    If context is irrelevant, answer normally using your own knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

general_prompt = ChatPromptTemplate.from_template(
    """
    You are DocEx, a helpful AI assistant.
    Answer the following question clearly and concisely.

    Question:
    {question}

    Answer:
    """
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ---------------- CHAT ----------------
user_query = st.chat_input("Ask anything...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("Thinking..."):

        answer = None

        # 👉 Try RAG first if documents exist
        if st.session_state.vectordb:
            retriever = st.session_state.vectordb.as_retriever(
                search_kwargs={"k": 4}
            )
            retrieved_docs = retriever.get_relevant_documents(user_query)

            # If meaningful context exists → RAG
            if retrieved_docs and len(format_docs(retrieved_docs).strip()) > 200:
                rag_chain = (
                    {
                        "context": lambda _: format_docs(retrieved_docs),
                        "question": RunnablePassthrough()
                    }
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke(user_query)

        # 👉 Fallback to general LLM
        if not answer:
            general_chain = (
                general_prompt
                | llm
                | StrOutputParser()
            )
            answer = general_chain.invoke({"question": user_query})

    st.session_state.chat_history.append(("assistant", answer))

# ---------------- DISPLAY CHAT ----------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)
