import streamlit as st
import os
import speech_recognition as sr
from dotenv import load_dotenv

# LangChain Core
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool

# Embeddings & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Loaders
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

# Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Retriever tool
from langchain.tools.retriever import create_retriever_tool

# External tools
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Agent + LLM
from langchain_groq import ChatGroq
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub

# ----------------------------
# ENV
# ----------------------------
load_dotenv()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="DocEx", page_icon="📄")
st.title("📄 DocEx – Universal Document Chatbot")

st.caption("Chat with PDFs, documents, and websites using AI")

# ----------------------------
# Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ----------------------------
# File Upload & URL Input
# ----------------------------
st.subheader("📤 Upload Documents or Paste Website URL")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

url_input = st.text_input("Website URL (optional)")

process_btn = st.button("🔍 Process Documents")

# ----------------------------
# Document Processing
# ----------------------------
if process_btn:
    docs = []

    # Handle file uploads
    if uploaded_files:
        for file in uploaded_files:
            file_path = f"/tmp/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.read())

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)

            docs.extend(loader.load())

    # Handle website URL
    if url_input:
        loader = WebBaseLoader(url_input)
        docs.extend(loader.load())

    if not docs:
        st.warning("Please upload a document or provide a URL.")
    else:
        with st.spinner("Processing documents..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectordb = FAISS.from_documents(split_docs, embeddings)
            st.session_state.retriever = vectordb.as_retriever()

        st.success("✅ Documents processed successfully! You can now chat.")

# ----------------------------
# Tools
# ----------------------------
wiki = Tool(
    name="wikipedia",
    func=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=1000
    ).run,
    description="Use for general world knowledge not in documents."
)

arxiv = Tool(
    name="arxiv",
    func=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    ).run,
    description="Use for academic or research paper queries."
)

tools = [wiki, arxiv]

# Add document retriever tool dynamically
if st.session_state.retriever:
    doc_tool = create_retriever_tool(
        st.session_state.retriever,
        "Document_Retriever",
        "Answer questions ONLY using the uploaded documents or provided website. "
        "If information is missing, say it is not found."
    )
    tools.insert(0, doc_tool)

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# Agent
# ----------------------------
from langchain_core.prompts import ChatPromptTemplate

base_prompt = hub.pull("hwchase17/openai-functions-agent")

system_prompt = """
You are a document-based assistant.

Rules:
- Always use Document_Retriever when available.
- Don't Answer only from the provided documents, try to use your own knowledge as well.
- If the answer is not in the documents, say:
  "Not found in the provided content."
- Do NOT hallucinate.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        *base_prompt.messages[1:]  # keep rest of original prompt
    ]
)


agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)

# ----------------------------
# Voice Input
# ----------------------------
recognizer = sr.Recognizer()
voice_input = None

if st.button("🎙️ Speak"):
    with sr.Microphone() as source:
        st.info("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

        try:
            voice_input = recognizer.recognize_google(audio)
        except:
            st.error("Could not understand voice input")

# ----------------------------
# Chat UI
# ----------------------------
user_input = st.chat_input("Ask about the document...") or voice_input

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Handle Chat
# ----------------------------
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    langchain_history = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": langchain_history
            })

            answer = response.get("output", "No response.")
            st.markdown(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
