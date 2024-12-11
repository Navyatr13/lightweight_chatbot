from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths to your documents and vector database
DOCUMENTS_PATH = "./app/data/documents.txt"
VECTOR_DB_PATH = "./app/data/vector_store"

def build_vector_store():
    """Generate and save embeddings for the document data."""
    #try:
    # Load the documents
    print(f"Loading documents from {DOCUMENTS_PATH}...")
    loader = TextLoader(DOCUMENTS_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Create embeddings
    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build the FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Embeddings generated successfully.")

    # Save the vector store
    print(f"Saving vector store to {VECTOR_DB_PATH}...")
    vector_store.save_local(VECTOR_DB_PATH)
    print("Vector store saved successfully.")
    #except Exception as e:
    #    print("Error occurred while building the vector store:", str(e))

if __name__ == "__main__":
    build_vector_store()
