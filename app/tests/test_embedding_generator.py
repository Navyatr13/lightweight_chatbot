import pandas as pd
from app.generate_embeddings import (
    build_vector_store,
    build_vector_store_from_json,
    build_vector_store_from_dataframe,
)

def test_build_vector_store():
    build_vector_store()
    print("Vector store built successfully from documents.")

def test_build_vector_store_from_json():
    json_path = "./app/data/mock_data.json"  # Replace with actual JSON path
    vector_db_path = "./app/data/vector_store_test"
    build_vector_store_from_json(json_path, vector_db_path)
    print("Vector store built successfully from JSON.")

def test_build_vector_store_from_dataframe():
    df = pd.DataFrame({
        "Input": ["Headache and fever", "Sore throat and fatigue"],
        "Disease": ["Flu", "Cold"]
    })
    vector_db_path = "./app/data/vector_store_test"
    build_vector_store_from_dataframe(df, vector_db_path)
    print("Vector store built successfully from DataFrame.")

if __name__ == "__main__":
    test_build_vector_store()
    test_build_vector_store_from_json()
    test_build_vector_store_from_dataframe()
