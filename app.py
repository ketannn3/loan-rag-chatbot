import streamlit as st
from rag_utils import get_top_k_docs
import openai

# Set your OpenAI key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🤖 Loan Approval RAG Chatbot")
st.write("Ask a question about loan approvals based on real data.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        context = "\n".join(get_top_k_docs(query))
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        st.success(answer)
