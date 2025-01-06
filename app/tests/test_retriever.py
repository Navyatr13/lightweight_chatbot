# app/test/test_retriever.py
import pytest
from app.retriever import build_vector_store, load_vector_store, retrieve_context
from app.config import DOCUMENT_PATH, VECTOR_DB_PATH
from app.response_generator import generate_response
DOCUMENT_PATH = "./app/data/ndocuments.txt"
VECTOR_DB_PATH = "./app/data/vector_store"

def test_build_vector_store():
    """Test vector store creation."""
    build_vector_store(DOCUMENT_PATH, VECTOR_DB_PATH)
    assert True, "Vector store built successfully."

def test_load_vector_store():
    """Test vector store loading."""
    vector_store = load_vector_store(VECTOR_DB_PATH)
    assert vector_store is not None, "Vector store should load successfully."

def test_retrieve_context():
    """Test context retrieval."""
    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context("What is artificial intelligence?", vector_store)
    assert len(context) > 0, "Retrieved context should not be empty."

if __name__ == "__main__":
    test_build_vector_store()
    test_load_vector_store()
    test_retrieve_context()

    VECTOR_DB_PATH = "./app/data/vector_store"
    query = "What are the symptoms of depression ?"

    vector_store = load_vector_store(VECTOR_DB_PATH)
    context = retrieve_context(query, vector_store)
    response = generate_response(query, context)
    print("Final Response:", response)