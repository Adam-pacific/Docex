import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing")
    st.stop()

# --------------------------------------------------
# STREAMLIT
# --------------------------------------------------
st.set_page_config(page_title="DocEx", page_icon="📄", layout="wide")
st.title("📄 DocEx – Chat with PDF")

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
file = st.file_uploader("Upload a PDF", type="pdf")

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer ONLY using the context below.
        If not found, say you don't know.

        Context:
        {context}

        Question:
        {question}
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

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.chat_input("Ask about the document")

    if user_q:
        st.session_state.history.append(("user", user_q))
        with st.spinner("Thinking..."):
            ans = chain.invoke(user_q)
        st.session_state.history.append(("ai", ans))

    for role, msg in st.session_state.history:
        st.chat_message("user" if role == "user" else "assistant").write(msg)

else:
    st.info("Upload a PDF to begin")
