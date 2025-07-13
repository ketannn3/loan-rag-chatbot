import streamlit as st
from datasets import load_dataset
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load dataset and model once to avoid reloading on each interaction
@st.cache_resource
def load_resources():
    docs = pd.read_csv("docs.csv")
    retriever = TfidfVectorizer()
    docs['combined'] = docs.apply(lambda row: f"Gender: {row['Gender']}, Married: {row['Married']}, Education: {row['Education']}, Income: {row['ApplicantIncome']}, Loan Amount: {row['LoanAmount']}", axis=1)
    tfidf_matrix = retriever.fit_transform(docs['combined'])
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return docs, retriever, tfidf_matrix, qa_model

docs, retriever, tfidf_matrix, qa_model = load_resources()

# Streamlit UI
st.title("🧠 Loan Approval RAG Chatbot")
st.subheader("(Free Hugging Face Model)")
st.write("Ask a question about loan approvals based on real data.")

query = st.text_input("Enter your question:")

def retrieve_context(query, k=3):
    query_vec = retriever.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_k = similarity[0].argsort()[-k:][::-1]
    context = "\n".join(docs.iloc[i]['combined'] for i in top_k)
    return context

def get_answer(query):
    context = retrieve_context(query)
    response = qa_model(question=query, context=context)
    answer = response['answer']

    # Filter vague or low-confidence answers
    if answer.lower() in ['yes', 'no', 'education', ''] or response['score'] < 0.3:
        return "Not enough relevant data to answer accurately."
    
    return answer

if query:
    result = get_answer(query)
    st.success(result)
