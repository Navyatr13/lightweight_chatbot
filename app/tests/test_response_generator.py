from app.response_generator import generate_response

# Example query and context
query = "What is reason for flu?"
context = "The common symptoms of flu include fever, cough, and body aches."

# Generate response
response = generate_response(query, context)
print("Generated Response:", response)
