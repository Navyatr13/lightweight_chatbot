import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from datasets import load_dataset, DatasetDict, Dataset

# Paths to your documents and vector database
DOCUMENTS_PATH = "./app/data/documents.txt"
JSON_PATH = "./app/data/combined_disease_prediction_symptom.json"
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
    
    
def prepare_chunks_from_df(df):
    """
    Convert symptom-disease pairs into retrieval-friendly text chunks.
    """
    chunks = []
    for _, row in df.iterrows():
        symptom = row["Input"]
        disease = row["Disease"]
        chunk = f"Symptom: {symptom}. Disease: {disease}."
        chunks.append(chunk)
    return chunks

def build_vector_store_from_json(json_path, vector_db_path):
    """
    Generate and save embeddings for symptom-disease pairs in JSON.
    """
    # Load JSON data
    print(f"Loading data from {json_path}...")
    dataset = load_dataset("prognosis/symptoms_disease_v1")
    # Convert to a pandas dataframe
    updated_data = [{'Input': item['instruction'], 'Disease': item['output']} for item in dataset['train']]
    df = pd.DataFrame(updated_data)
    print(f"Loaded {len(df)} records.")

    # Prepare text chunks
    print("Preparing text chunks...")
    print(df)
    chunks = prepare_chunks_from_df(df)
    documents = [Document(page_content=chunk) for chunk in chunks]
    print(f"Prepared {len(documents)} documents for embedding.")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Embeddings generated successfully.")

    # Save vector store
    print(f"Saving vector store to {vector_db_path}...")
    vector_store.save_local(vector_db_path)
    print("Vector store saved successfully.")

if __name__ == "__main__":
    build_vector_store_from_json(JSON_PATH, VECTOR_DB_PATH)
    #build_vector_store()
