# 🧠 Loan Approval RAG Chatbot (Free Hugging Face Model)

This project is a **Retrieval-Augmented Generation (RAG) based chatbot** that answers questions about loan approvals using real data. It combines semantic search with a Hugging Face QA model to produce natural-language answers based on relevant examples.

---

## 🌐 Live Demo

👉 [Click here to try the chatbot](https://loan-rag-chatbot-lbu8sq5ichdeypjdsmky6u.streamlit.app/)

---

## 🚀 Features

- 💬 Ask natural questions about loan approvals
- 📖 Answers are based on real-world loan data
- 🔍 Top-k document retrieval using TF-IDF
- 🤖 Uses Hugging Face’s `distilbert-base-uncased-distilled-squad` for Q&A
- 📊 Shows context retrieved for transparency
- 🌐 Streamlit-based web interface

---

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit
- Hugging Face Transformers
- Scikit-learn (TF-IDF)
- Pandas

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/loan-rag-chatbot.git
cd loan-rag-chatbot
pip install -r requirements.txt
streamlit run app.py
