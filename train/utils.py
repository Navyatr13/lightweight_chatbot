import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class LanguageDataset(Dataset):
    """
    An extension of the Dataset object to:
      - Make training loop cleaner
      - Make ingestion easier from pandas df's
    """
    def __init__(self, df, tokenizer):
        self.labels = df.columns
        self.data = df.to_dict(orient='records')
        self.tokenizer = tokenizer
        x = self.fittest_max_length(df)  # Fix here
        self.max_length = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.labels[0]]
        y = self.data[idx][self.labels[1]]
        text = f"{x} | {y}"
        tokens = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return tokens

    def fittest_max_length(self, df):  # Fix here
        """
        Smallest power of two larger than the longest term in the data set.
        Important to set up max length to speed training time.
        """
        max_length = max(len(max(df[self.labels[0]], key=len)), len(max(df[self.labels[1]], key=len)))
        x = 2
        while x < max_length: x = x * 2
        return x


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, model_name, batch_size):
    model.train()
    epoch_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs} Batch Size: {batch_size}, Transformer: {model_name}")
    for batch in train_iterator:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({'Training Loss': loss.item()})
        epoch_loss += loss.item()
    avg_epoch_training_loss = epoch_loss / len(train_iterator)
    return avg_epoch_training_loss

def validate_one_epoch(model, valid_loader, criterion, device, epoch, num_epochs, model_name, batch_size):
    model.eval()
    epoch_loss = 0
    valid_iterator = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
    with torch.no_grad():
        for batch in valid_iterator:
            inputs = batch['input_ids'].to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            epoch_loss += loss.item()
    avg_epoch_validation_loss = epoch_loss / len(valid_loader)
    end_time = time.time()  # End the timer for the epoch
    epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

    return avg_epoch_validation_loss
