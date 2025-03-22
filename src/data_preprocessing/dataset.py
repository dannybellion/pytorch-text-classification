"""Dataset loading and preprocessing functions."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the dataset from a JSON file.
    
    Args:
        data_path: Path to the JSON dataset file
        
    Returns:
        DataFrame containing the dataset
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Ensure balanced splits
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    
    return train_df, test_df


class LoanDefaultDataset(Dataset):
    """Dataset class for loan default classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer_name: str = "huawei-noah/TinyBERT_General_6L_768D"):
        """Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 or 1)
            tokenizer_name: Name of the pre-trained tokenizer
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the tokenized text and label
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Remove the batch dimension the tokenizer adds
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
        return item


def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer_name: str = "huawei-noah/TinyBERT_General_6L_768D",
    batch_size: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        tokenizer_name: Name of the pre-trained tokenizer
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = LoanDefaultDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer_name=tokenizer_name
    )
    
    test_dataset = LoanDefaultDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer_name=tokenizer_name
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader