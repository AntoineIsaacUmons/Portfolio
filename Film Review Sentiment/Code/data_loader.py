import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class IMDBDataset(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_length):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        sentiment = self.sentiments[idx]

        # Tokenize the review
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

def load_data(file_path, tokenizer, max_length=128, test_size=0.2):
    # Load the CSV file
    df = pd.read_csv(file_path)

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split the data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'].values, df['sentiment'].values, test_size=test_size, random_state=42
    )

    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)

    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader