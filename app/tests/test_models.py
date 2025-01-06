# app/tests/test_models.py
import pytest
from app.models import load_model
from app.response_generator import generate_response

@pytest.fixture
def generator():
    """Load the model once for all tests."""
    return load_model("distilgpt2")


def test_model_response(generator):
    """Test if the model generates a non-empty response."""
    query = "What is AI?"
    context = "Artificial intelligence (AI) refers to machines simulating human intelligence."

    # Pass query and context explicitly
    response = generate_response(generator, f"Context: {context}\n\nUser Query: {query}\n\nBot Response:")
    print(response)
    assert len(response) > 0, "The response should not be empty."
    assert "AI" in response, "The response should mention 'AI'."

if __name__ == "main":
    test_model_response()