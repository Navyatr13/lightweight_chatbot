from app.response_generator import generate_response

# Example query and context
query = "What is artificial intelligence?"
context = "Artificial Intelligence is a branch of computer science. It aims to create systems capable of performing tasks that require human intelligence."

# Generate response
response = generate_response(query, context)
print("Generated Response:", response)
