import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it to .env or Streamlit secrets.")
    st.stop()

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="DocEx 📄🤖",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocEx – Chat with your Documents")
st.caption("Powered by Groq + LangChain")

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.2
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant.
        Answer the question ONLY using the context below.
        If the answer is not present, say "I couldn't find that in the document."

        Context:
        {context}

        Question:
        {question}
        """
    )

    # --------------------------------------------------
    # RAG CHAIN (NEW v0.2 STYLE)
    # --------------------------------------------------
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------
    # CHAT UI
    # --------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something about the document...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_input)

        st.session_state.chat_history.append(("ai", response))

    # Display chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

else:
    st.info("📂 Upload a PDF to start chatting.")
