# train/train.py

from data_processing import load_datasets
from model import initialize_model
from utils import LanguageDataset
from utils import train_one_epoch, validate_one_epoch
import torch
import pandas as pd

BATCH_SIZE = 8
model_name = 'distilgpt2'

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_loader, valid_loader = load_datasets(batch_size=BATCH_SIZE)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model('distilgpt2', device)

    # Training configurations
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, model_name, batch_size=BATCH_SIZE )
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device, epoch, num_epochs, model_name,  batch_size=BATCH_SIZE)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f}")

    # Save the fine-tuned model
    save_model(model, tokenizer, './outputs/fine_tuned_model/')

if __name__ == "__main__":
    main()
