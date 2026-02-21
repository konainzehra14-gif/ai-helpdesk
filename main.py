# main.py
# =========================================================
# AI Helpdesk Retrieval Pipeline (Deployment-Ready)
# =========================================================

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# =========================================================
# 1️⃣ Paths / Hugging Face downloads
# =========================================================
repo_id = "konainzehra/helpdesk-model"

FAISS_PATH = hf_hub_download(repo_id=repo_id, filename="faiss_index.bin")
DATA_PICKLE_PATH = hf_hub_download(repo_id=repo_id, filename="data.pkl")

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
    if I[0][0] == -1:
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
# 7️⃣ FastAPI app
# =========================================================
app = FastAPI(title="Fauji Foundation AI Helpdesk")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to Fauji Foundation AI Helpdesk!"}

@app.post("/ask")
def ask_helpdesk(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = helpdesk_pipeline(question)
    return {"question": question, "answer": answer}

# =========================================================
# 8️⃣ Optional CLI for local testing
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
