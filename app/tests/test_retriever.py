# app/tests/test_retriever.py
from app.retriever import build_vector_store, load_vector_store, retrieve_context
from app.config import DOCUMENT_PATH, VECTOR_DB_PATH

DOCUMENT_PATH = "./app/data/documents.txt"
VECTOR_DB_PATH = "./app/data/vector_store"

# Build vector store
build_vector_store(DOCUMENT_PATH, VECTOR_DB_PATH)

# Load vector store
vector_store = load_vector_store(VECTOR_DB_PATH)

# Retrieve context
query = "What is artificial intelligence?"
context = retrieve_context(query, vector_store)
print("Retrieved Context:", context)

def test_retriever():
    build_vector_store(DOCUMENT_PATH, VECTOR_DB_PATH)
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context("What is AI?", vector_store)
    assert len(context) > 0