#train/data_processing.py
# Dataset Prep

class LanguageDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = self.fittest_max_length(df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]["Input"]
        target_text = self.data.iloc[idx]["Disease"]

        input_tokens = self.tokenizer.encode_plus(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target_tokens = self.tokenizer.encode_plus(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "input_ids": input_tokens["input_ids"].squeeze(0),
            "labels": target_tokens["input_ids"].squeeze(0),
        }

    def fittest_max_length(self, df):
        max_input_len = df["Input"].map(len).max()
        max_output_len = df["Disease"].map(len).max()
        return min(512, max(max_input_len, max_output_len))

# Cast the Huggingface data set as a LanguageDataset we defined above
data_sample = LanguageDataset(df, tokenizer)