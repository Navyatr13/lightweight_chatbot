# train/dataset.py
import pandas as pd

from torch.utils.data import DataLoader, random_split
from train.utils import LanguageDataset
from transformers import GPT2Tokenizer

def load_datasets(batch_size=8):
    # Load dataset
    df = pd.read_json('./data/combined_disease_prediction_symptom.json')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and split
    dataset = LanguageDataset(df, tokenizer)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    return train_loader, valid_loader
