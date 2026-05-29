# 🛡️ Insurance Policy AI Chatbot

An AI-powered chatbot that reads insurance policy PDFs and answers customer 
questions in plain English — no more waiting on hold.

## 🎯 What It Does
- Upload any insurance PDF policy document
- Ask questions in plain English: "Does this cover dental surgery?"
- Get accurate answers with page references from the actual document
- Built on RAG (Retrieval Augmented Generation) — same tech as Microsoft Copilot

## 🧠 Why I Built This
Insurance policy documents are notoriously complex. During my year at 
ICICI Lombard, I saw how customers struggled to understand their coverage.
This tool makes policy information instantly accessible.

## 🏗️ Architecture
User Question → Streamlit UI → FastAPI Backend → LangChain RAG Pipeline
↓
PDF → Chunks → Embeddings
↓
FAISS Vector Search
↓
Relevant Chunks + GPT-4o
↓
Clear Answer + Source Page
