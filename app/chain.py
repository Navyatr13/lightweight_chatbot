# app/chain.py
from app.retriever import retrieve_context
from app.models import generate_response

def query_chain(query: str, vector_store, generator):
    """Combine retrieval and generation."""
    context = retrieve_context(query, vector_store)
    prompt = f"Context: {context}\n\nUser: {query}\n\nBot:"
    return generate_response(generator, prompt)
