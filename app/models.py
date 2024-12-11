# app/models.py
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str):
    """Load the text generation model."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

def generate_response(generator, prompt: str):
    """Generate a response using the model."""
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]
