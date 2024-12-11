from transformers import GPT2LMHeadModel, GPT2Tokenizer

def initialize_model(model_name, device):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def save_model(model, tokenizer, output_dir):
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
