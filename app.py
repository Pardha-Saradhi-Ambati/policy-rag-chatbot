import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Policy RAG Chatbot", page_icon="🤖")

st.title("🤖 Policy RAG Chatbot")

# -----------------------------
# Load Documents
# -----------------------------
@st.cache_resource
def initialize_rag():

    loader = PyPDFDirectoryLoader("documents")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    return retriever, llm

retriever, llm = initialize_rag()

# -----------------------------
# RAG Prompt
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = RAG_PROMPT.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(prompt)

    return response.content

# -----------------------------
# Guardrails
# -----------------------------
OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier.

Determine whether the user's question can be answered using the policy documents.

Question: {question}

Respond with only:

YES - if the question is related to the policy documents.
NO - if it is unrelated.
""")

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions related to the provided policy documents."
)

def ask_bot(question):

    prompt = OOS_PROMPT.invoke({
        "question": question
    })

    decision = llm.invoke(prompt).content.strip().upper()

    if "NO" in decision:
        return REFUSAL_MESSAGE

    return rag_chain(question)

# -----------------------------
# UI
# -----------------------------
question = st.text_input("Ask a question")

if question:

    with st.spinner("Thinking..."):

        try:
            answer = ask_bot(question)
            st.write(answer)

        except Exception as e:
            st.error(str(e))