# =========================================================
# Lightweight HR Support Chatbot for Railway Deployment
# =========================================================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# 1️⃣ Download Dataset + Model from Hugging Face
# -----------------------------
REPO_ID = "konainzehra/helpdesk-model"

DATA_CSV_PATH = hf_hub_download(repo_id=REPO_ID, filename="hr_dataset_modified.csv")
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="minilm-finetuned")

# -----------------------------
# 2️⃣ Load Dataset & Model
# -----------------------------
df = pd.read_csv(DATA_CSV_PATH)  # Ensure CSV has columns: "question", "answer"
questions = df['question'].tolist()
answers = df['answer'].tolist()

embedding_model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
question_embeddings = embedding_model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

# -----------------------------
# 3️⃣ Confidential Filter
# -----------------------------
CONFIDENTIAL_KEYWORDS = ["password", "salary", "bank", "ssn", "credit card", "confidential"]

def is_confidential(query):
    return any(word in query.lower() for word in CONFIDENTIAL_KEYWORDS)

# -----------------------------
# 4️⃣ Chatbot Pipeline
# -----------------------------
CONFIDENCE_THRESHOLD = 0.65

def chatbot_pipeline(query):
    if is_confidential(query):
        return "Access Restricted"

    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    similarities = (question_embeddings @ query_vec.T).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        return "I am not confident about this answer. Please contact HR."

    return answers[best_idx]

# -----------------------------
# 5️⃣ FastAPI App
# -----------------------------
app = FastAPI(title="HR Support Chatbot")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to HR Support Chatbot!"}

@app.post("/ask")
def ask_helpdesk(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = chatbot_pipeline(question)
    return {"question": question, "answer": answer}

# -----------------------------
# 6️⃣ Local Testing
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(8000), reload=True)
