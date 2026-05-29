"""
main.py — The API server for our Insurance Chatbot

This creates HTTP endpoints that the frontend can call:
- POST /upload  → accepts a PDF file, processes it
- POST /ask     → accepts a question, returns an answer
- GET /health   → checks if server is running

Think of this like a waiter in a restaurant:
- The frontend (Streamlit) is the customer
- This file is the waiter
- chatbot.py is the kitchen
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chatbot import load_and_split_pdf, create_vector_store, build_qa_chain, ask_question

# Create the FastAPI app
app = FastAPI(
    title="Insurance Policy AI Chatbot",
    description="Upload an insurance PDF and ask questions about your coverage",
    version="1.0.0"
)

# Allow the frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# We store the QA chain in memory while the server is running
# In production you'd use a database, but this is fine for a demo
qa_chain_store = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health_check():
    """Simple check to see if the API is running."""
    return {"status": "running", "message": "Insurance AI Chatbot is live!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, processes it, and prepares it for Q&A.
    
    What happens here:
    1. Save the uploaded PDF to a temporary location
    2. Split it into chunks
    3. Create vector embeddings
    4. Build the QA chain
    5. Store it in memory for the session
    """
    # Check it's actually a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process the PDF through our chatbot pipeline
        chunks = load_and_split_pdf(tmp_path)
        vector_store = create_vector_store(chunks)
        qa_chain = build_qa_chain(vector_store)
        
        # Store the chain for this session
        qa_chain_store["default"] = qa_chain
        
        return {
            "message": f"Successfully processed '{file.filename}'",
            "chunks_created": len(chunks),
            "ready": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        os.unlink(tmp_path)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Accepts a question and returns an AI-generated answer based on the PDF.
    """
    # Check if a document has been uploaded
    if request.session_id not in qa_chain_store:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )
    
    qa_chain = qa_chain_store[request.session_id]
    result = ask_question(qa_chain, request.question)
    
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )