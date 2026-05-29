"""
chatbot.py — The AI brain of our Insurance Chatbot

This file handles:
1. Reading PDF documents
2. Splitting them into chunks
3. Converting chunks to embeddings (numbers that capture meaning)
4. Storing them in a vector database (FAISS)
5. Finding relevant chunks when a user asks a question
6. Sending those chunks + question to GPT-4o for an answer

This pattern is called RAG — Retrieval Augmented Generation.
Think of it like: instead of asking GPT to remember everything,
we look up the relevant pages first, then ask GPT to answer
based only on those pages. More accurate, less expensive.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the .env file so Python can access OPENAI_API_KEY
load_dotenv()


def load_and_split_pdf(pdf_path: str):
    """
    Step 1 & 2: Read a PDF and split it into chunks.
    
    Why chunks? GPT can only read a limited amount of text at once.
    A 40-page PDF is too long. So we split it into ~500 word pieces.
    Later we only send the RELEVANT pieces to GPT, not the whole doc.
    """
    # Load the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks of 500 characters, with 50 character overlap
    # Overlap ensures we don't cut a sentence in the middle and lose context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"PDF split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """
    Step 3 & 4: Convert chunks to embeddings and store in FAISS.
    
    What is an embedding?
    It converts text like "dental surgery is covered" into a list of 
    1536 numbers like [0.23, -0.45, 0.89, ...]. Similar meanings 
    produce similar number patterns. This lets us find related content
    mathematically — no keyword matching needed.
    
    FAISS is a vector database by Meta — it stores these number-lists
    and can find the most similar ones in milliseconds.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create the vector store from our chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {len(chunks)} embeddings")
    return vector_store


def build_qa_chain(vector_store):
    """
    Step 5 & 6: Build the question-answering chain.
    
    This combines:
    - A retriever: finds the top 3 most relevant chunks for any question
    - A prompt: tells GPT exactly how to behave
    - GPT-4o: generates the final answer
    
    The prompt is important — it tells GPT to:
    1. Only use the provided document context (not its general knowledge)
    2. Say "I don't know" if the answer isn't in the document
    3. Mention where it found the answer (page number)
    """
    
    # This prompt is the instruction we give to GPT
    prompt_template = """You are an expert insurance policy assistant helping 
customers understand their insurance coverage.

Use ONLY the following context from the insurance document to answer the question.
If the answer is not found in the context, say "I couldn't find this information 
in the provided policy document. Please contact your insurance provider directly."

Always be clear, friendly, and use simple language that anyone can understand.
If you find the answer, mention which section or page it came from.

Context from document:
{context}

Customer Question: {question}

Your Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # The LLM (Large Language Model) — GPT-4o mini is cheaper, still great
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # 0 = factual, no creativity. Good for Q&A.
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # The retriever fetches top 3 most relevant chunks for each question
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # RetrievalQA combines retriever + LLM into one chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all chunks into one prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Also return which chunks were used
    )
    
    return qa_chain


def ask_question(qa_chain, question: str) -> dict:
    """
    The main function that puts it all together.
    Takes a question, returns an answer with source information.
    """
    result = qa_chain.invoke({"query": question})
    
    # Extract the answer
    answer = result["result"]
    
    # Extract source pages (where the answer came from)
    sources = []
    for doc in result["source_documents"]:
        page_num = doc.metadata.get("page", "unknown")
        sources.append(f"Page {page_num + 1}")  # +1 because pages start at 0
    
    # Remove duplicate page numbers
    sources = list(set(sources))
    
    return {
        "answer": answer,
        "sources": sources
    }