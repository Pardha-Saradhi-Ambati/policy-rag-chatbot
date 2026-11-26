# policy-rag-chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer user queries accurately using custom policy documents. 

## 📌 Overview
Searching through lengthy policy documents can be time-consuming. This application leverages RAG architecture to retrieve relevant sections from stored documents and generate clear, context-aware answers through a conversational interface.

## ✨ Features
- **Document-Based Q&A:** Answers questions strictly grounded in the provided policy documents.
- **Context Retrieval:** Searches and retrieves the most relevant policy excerpts before generating responses.
- **Interactive Interface:** Easy-to-use chat interface powered by `app.py`.

## 📂 Project Structure
```text
policy-rag-chatbot/
├── documents/         # Folder containing source policy files (PDF, TXT, etc.)
├── app.py             # Main application script for the RAG pipeline & UI
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
