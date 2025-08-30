# Docex.py

import streamlit as st
import speech_recognition as sr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.chains import ConversationChain
from dotenv import load_dotenv
load_dotenv() 

# ---------------------------
# Embeddings + Docs
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

vectordb = FAISS.from_documents(docs, embeddings)
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    description="Search for information about LangSmith. For any questions about LangSmith, you must use this tool."
)

# ---------------------------
# Tools (Wikipedia + Arxiv)
# ---------------------------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# ---------------------------
# LLM (Groq)
# ---------------------------
import os
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.4,
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------
# Agent setup
# ---------------------------
prompt = hub.pull("hwchase17/openai-functions-agent")
tools = [retriever_tool, wiki, arxiv]
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Conversation memory
convo = ConversationChain(llm=llm)

# ---------------------------
# Streamlit UI (ChatGPT style)
# ---------------------------
st.set_page_config(page_title="DocEx", page_icon="🌟")
st.title("🤖 DocEx - Ask the Agent Anything")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# Voice Input Handling
# ---------------------------
recognizer = sr.Recognizer()
voice_input = None

if st.button("🎙️ Speak Now"):
    with sr.Microphone() as source:
        st.info("Listening... Speak into the mic 🎤")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        try:
            voice_input = recognizer.recognize_google(audio)
            st.success(f"You said: {voice_input}")
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your speech.")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")

# ---------------------------
# Chat Input (Text or Voice)
# ---------------------------
user_input = st.chat_input("Enter your query...") or voice_input

if user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Agent response
    try:
        response = agent_executor.invoke({"input": user_input})
        answer = response["output"]
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    # Save bot response
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
