from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Path to your fine-tuned model file
pt_file_path = "/home/ubuntu/Desktop/lightweight_chatbot/app/model_checkpoints/SmallMedLM.pt"  # Replace with the actual location of your `.pt` file
output_dir = "/home/ubuntu/Desktop/lightweight_chatbot/app/model_checkpoints/llm"  # Replace with the directory where you want to save

# Load the serialized model object directly
print("Loading the fine-tuned model...")
model = torch.load(pt_file_path, map_location=torch.device("cpu"))  # Load the entire model object

# Check if the loaded model is compatible with GPT2
if not isinstance(model, AutoModelForCausalLM):
    print("The loaded model is not directly compatible with Hugging Face. Converting...")
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = model.half()  # Ensure compatibility by copying weights
    base_model.load_state_dict(model.state_dict())  # Copy weights into base model
    model = base_model

# Save in Hugging Face-compatible format
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
