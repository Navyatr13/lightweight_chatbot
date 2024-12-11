# app/main.py
from fastapi import FastAPI, Request
from app.models import load_model
from app.retriever import build_vector_store, load_vector_store
from app.chain import query_chain
from app.config import DOCUMENT_PATH, VECTOR_DB_PATH, MODEL_NAME

app = FastAPI()

# Load resources
generator = load_model(MODEL_NAME)
vector_store = load_vector_store(VECTOR_DB_PATH)


@app.on_event("startup")
def setup_vector_store():
    """Build the vector store if it doesn't exist."""
    build_vector_store(DOCUMENT_PATH, VECTOR_DB_PATH)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("message", "")
    if not user_query:
        return {"error": "Message is required."}

    response = query_chain(user_query, vector_store, generator)
    return {"reply": response}
