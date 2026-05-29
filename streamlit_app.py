"""
streamlit_app.py — The visual interface for our chatbot

Streamlit lets you build a web UI using only Python.
No HTML, no CSS, no JavaScript needed.
Perfect for demos and portfolio projects.
"""

import streamlit as st
import requests

# Configure the page
st.set_page_config(
    page_title="Insurance Policy AI Assistant",
    page_icon="🛡️",
    layout="centered"
)

# ── Header ──────────────────────────────────────────────────────
st.title("🛡️ Insurance Policy AI Assistant")
st.markdown("""
> Upload any insurance policy PDF and ask questions about your coverage 
> in plain English. Powered by GPT-4o and RAG technology.
""")

st.divider()

# ── API URL ──────────────────────────────────────────────────────
# When running locally, FastAPI runs on port 8000
API_URL = "http://localhost:8000"

# ── Upload Section ───────────────────────────────────────────────
st.subheader("📄 Step 1: Upload Your Policy Document")

uploaded_file = st.file_uploader(
    "Upload an insurance policy PDF",
    type=["pdf"],
    help="The AI will read this document and answer your questions based on it"
)

if uploaded_file is not None:
    # Show a button to process the file
    if st.button("🚀 Process Document", type="primary"):
        with st.spinner("Reading and indexing your document... (30-60 seconds)"):
            # Send the PDF to our FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ Document processed! Created {data['chunks_created']} searchable sections.")
                st.session_state["doc_ready"] = True
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

st.divider()

# ── Question Section ─────────────────────────────────────────────
st.subheader("💬 Step 2: Ask Your Question")

# Show sample questions for guidance
with st.expander("💡 Example questions to try"):
    st.markdown("""
    - Does my policy cover dental surgery?
    - What is my deductible amount?
    - Are pre-existing conditions covered?
    - What is the claim submission process?
    - Is emergency hospitalization covered?
    - What documents do I need to file a claim?
    """)

question = st.text_input(
    "Type your question about the policy:",
    placeholder="e.g., Does this policy cover hospitalization abroad?"
)

if st.button("🔍 Get Answer", type="primary"):
    if not question:
        st.warning("Please type a question first.")
    elif not st.session_state.get("doc_ready"):
        st.warning("Please upload and process a PDF document first.")
    else:
        with st.spinner("Searching your policy document..."):
            payload = {"question": question, "session_id": "default"}
            response = requests.post(f"{API_URL}/ask", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Show the answer
                st.subheader("📋 Answer")
                st.markdown(data["answer"])
                
                # Show the sources
                if data["sources"]:
                    st.info(f"📌 Found in: {', '.join(data['sources'])}")
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("Built by Javeed Kamal Shaik | IEEE AI Researcher | MS Information Technology, FAU")