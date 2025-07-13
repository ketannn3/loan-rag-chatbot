import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_resources():
    docs = pd.read_csv("loan.csv")
    # Combine all text features into a single string column (fallback method)
    docs['combined'] = docs.astype(str).agg(' '.join, axis=1)

    retriever = TfidfVectorizer(stop_words='english')
    tfidf_matrix = retriever.fit_transform(docs['combined'])

    qa_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return docs, retriever, tfidf_matrix, qa_model

def answer_question(question, docs, retriever, tfidf_matrix, qa_model):
    question_vec = retriever.transform([question])
    scores = cosine_similarity(question_vec, tfidf_matrix)[0]
    best_idx = scores.argmax()
    best_doc = docs.iloc[best_idx]
    return best_doc["LoanAmount"]

# Streamlit UI
st.title("🧠 Loan Approval RAG Chatbot\n(Free Hugging Face Model)")
st.write("Ask a question about loan approvals based on real data.")

question = st.text_input("Enter your question:")
if question:
    try:
        docs, retriever, tfidf_matrix, qa_model = load_resources()
        answer = answer_question(question, docs, retriever, tfidf_matrix, qa_model)
        st.success(f"Loan Amount: {answer}")
    except Exception as e:
        st.error(f"Error: {e}")
