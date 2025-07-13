from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load documents from CSV
docs = pd.read_csv("docs.csv", header=None)[0].tolist()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)

def get_top_k_docs(query, k=3):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [docs[i] for i in top_k_idx]
