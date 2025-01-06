#app/tests/test_response_generator.py
from app.models import load_model
from app.response_generator import generate_response

# Example query and context
query = "What is reason for flu?"
context = "The common symptoms of flu include fever, cough, and body aches."

# Generate response
response = generate_response(query, context)
print("Generated Response:", response)

def test_response_generation():
    """Test response generation with a given query and context."""
    query = "What is the reason for flu?"
    context = "The common symptoms of flu include fever, cough, and body aches."
    generator = load_model("distilgpt2")

    response = generate_response(generator, f"Context: {context}\n\nUser: {query}\n\nBot:")
    assert len(response) > 0, "Response should not be empty."
    assert "flu" in response, "Response should mention 'flu'."
    assert len(response) < 400, "Response should not be excessively long."
