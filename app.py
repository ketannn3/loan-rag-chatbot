import streamlit as st
from rag_utils import get_top_k_docs
from transformers import pipeline

# Load QA model
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_model()

st.title("🧠 Loan Approval RAG Chatbot (Free Hugging Face Model)")
st.write("Ask a question about loan approvals based on real data.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        context = "\n".join(get_top_k_docs(query))
        
        result = qa_pipeline({
            "context": context,
            "question": query
        })
        answer = result["answer"]
        
        # Display answer
        st.success(answer)

        # ✅ Show context used for transparency/debugging
        with st.expander("🔍 View Retrieved Context"):
            st.write(context)
