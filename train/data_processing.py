import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from utils import LanguageDataset
from transformers import GPT2Tokenizer
from datasets import load_dataset, DatasetDict, Dataset

def custom_collate(batch):
    # Filter out None values
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return None
    # Stack valid input tensors
    return {
        'input_ids': torch.stack([item['input_ids'] for item in filtered_batch])
    }


def load_datasets(batch_size=8):
    # Load dataset
    dataset = load_dataset("prognosis/symptoms_disease_v1")
    # Convert to a pandas dataframe
    updated_data = [{'Input': item['instruction'], 'Disease': item['output']} for item in dataset['train']]
    df = pd.DataFrame(updated_data)    
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and split
    dataset = LanguageDataset(df, tokenizer)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers = 7, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers = 7)

    return train_loader, valid_loader
