# =========================================================
# AI Helpdesk Retrieval Pipeline (Deployment-Ready, HF XET)
# =========================================================

import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download

# =========================================================
# 1️⃣ Paths / Hugging Face downloads
# =========================================================
repo_id = "konainzehra/helpdesk-model"

# Large binary files loaded directly from Hugging Face
FAISS_PATH = hf_hub_download(repo_id=repo_id, filename="faiss_index.bin")
DATA_PICKLE_PATH = hf_hub_download(repo_id=repo_id, filename="data.pkl")

# Local model folder
MODEL_PATH = "minilm-finetuned"

# =========================================================
# 2️⃣ Load MiniLM embeddings
# =========================================================
embedding_model = SentenceTransformer(MODEL_PATH)

# =========================================================
# 3️⃣ Load FAISS index and dataset
# =========================================================
with open(DATA_PICKLE_PATH, "rb") as f:
    data = pickle.load(f)
questions = data["questions"]
answers = data["answers"]

index = faiss.read_index(FAISS_PATH)

# =========================================================
# 4️⃣ MongoDB Setup (optional)
# =========================================================
MONGO_URL = os.environ.get("MONGO_URL", "")
if MONGO_URL:
    client = MongoClient(MONGO_URL)
    db = client["ai_helpdesk"]
    collection = db["helpdesk_logs"]
else:
    collection = None

def store_log(question, answer, source="ai_helpdesk"):
    if collection:
        collection.insert_one({
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": datetime.now(timezone.utc)
        })

# =========================================================
# 5️⃣ Confidential Filter
# =========================================================
CONFIDENTIAL_KEYWORDS = ["password", "salary", "bank", "ssn", "credit card", "confidential"]

def is_confidential(query):
    return any(word in query.lower() for word in CONFIDENTIAL_KEYWORDS)

# =========================================================
# 6️⃣ Helpdesk Pipeline
# =========================================================
CONFIDENCE_THRESHOLD = 0.65

def helpdesk_pipeline(query, k=1, confidence_threshold=CONFIDENCE_THRESHOLD):
    # Confidential check
    if is_confidential(query):
        answer = "Access Restricted"
        store_log(query, answer, source="confidential_block")
        return answer

    # Encode query
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # FAISS search
    D, I = index.search(query_vec, k)
    if I[0][0] == -1:a
        answer = "Please visit the official website or contact HR department at (051) 5951821."
        store_log(query, answer, source="fallback")
        return answer

    similarity = float(D[0][0])
    answer_text = answers[I[0][0]]

    if similarity < confidence_threshold:
        answer = "Please visit the official website or contact HR department at (051) 5951821."
        store_log(query, answer, source="fallback")
        return answer

    store_log(query, answer_text, source="retrieval")
    return answer_text

# =========================================================
# 7️⃣ Interactive CLI
# =========================================================
if __name__ == "__main__":
    print("🤖 AI Help Desk (type 'exit' to quit)\n")
    while True:
        user_query = input("Ask your question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break
        response = helpdesk_pipeline(user_query)
        print("Answer:", response)
        print("-" * 50)
