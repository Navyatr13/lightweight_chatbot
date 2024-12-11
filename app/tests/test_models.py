# app/tests/test_models.py
from app.models import load_model, generate_response

def test_model_response():
    generator = load_model("distilgpt2")
    response = generate_response(generator, "What is AI?")
    assert len(response) > 0
