from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import to resolve deprecation warnings
import os


def build_vector_store(document_path: str, vector_db_path: str):
    """
    Build and save a vector database from text documents.
    Args:
        document_path (str): Path to the text documents.
        vector_db_path (str): Path to save the vector database.
    """
    loader = TextLoader(document_path)
    docs = loader.load()

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build and save FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(vector_db_path)


def load_vector_store(vector_db_path: str):
    """
    Load an existing vector database.
    Args:
        vector_db_path (str): Path to the saved vector database.
    Returns:
        FAISS vector store object.
    """
    return FAISS.load_local(
        vector_db_path,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True  # Explicitly allow deserialization if the vector store is trusted
    )


def retrieve_context(query: str, vector_store, top_k: int = 2) -> str:
    """
    Retrieve the most relevant context for a given query.
    Args:
        query (str): The input query string.
        vector_store: The FAISS vector store object.
        top_k (int): Number of top results to retrieve.
    Returns:
        str: Concatenated content of the retrieved documents.
    """
    results = vector_store.similarity_search(query, top_k)
    return " ".join([result.page_content for result in results])
