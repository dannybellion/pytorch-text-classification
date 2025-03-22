"""Tests for the dataset module."""
import os
import tempfile
import json

import pandas as pd
import pytest
import torch

from src.data_preprocessing.dataset import (
    load_dataset, 
    split_dataset, 
    LoanDefaultDataset, 
    create_dataloaders
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {"text": "This is a sample text for testing", "label": 0},
        {"text": "Another sample text with a different label", "label": 1},
        {"text": "A third sample for good measure", "label": 0},
        {"text": "One more sample with the default label", "label": 1},
        {"text": "Final test sample for the dataset", "label": 0},
    ]


@pytest.fixture
def temp_dataset_file(sample_data):
    """Create a temporary dataset file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_file_name = f.name
    
    yield temp_file_name
    
    # Clean up
    os.unlink(temp_file_name)


def test_load_dataset(temp_dataset_file, sample_data):
    """Test loading dataset from a file."""
    df = load_dataset(temp_dataset_file)
    
    # Check if DataFrame is created correctly
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_data)
    assert "text" in df.columns
    assert "label" in df.columns


def test_split_dataset(sample_data):
    """Test splitting dataset into train and test sets."""
    df = pd.DataFrame(sample_data)
    
    train_df, test_df = split_dataset(df, test_size=0.2)
    
    # Check if splits are created with the correct sizes
    assert len(train_df) + len(test_df) == len(df)
    
    # Check if label distribution is maintained
    assert set(train_df["label"].unique()) == set(df["label"].unique())


def test_loan_default_dataset():
    """Test LoanDefaultDataset class."""
    texts = ["Sample text 1", "Sample text 2"]
    labels = [0, 1]
    
    dataset = LoanDefaultDataset(texts, labels)
    
    # Check dataset length
    assert len(dataset) == 2
    
    # Check item format
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    
    # Check tensor types
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)
    
    # Check label value
    assert item["label"].item() == labels[0]


def test_create_dataloaders(sample_data):
    """Test creating DataLoaders from dataset."""
    df = pd.DataFrame(sample_data)
    
    train_df, test_df = split_dataset(df, test_size=0.2)
    
    train_loader, test_loader = create_dataloaders(
        train_df, test_df, batch_size=2
    )
    
    # Check if DataLoaders are created
    assert train_loader is not None
    assert test_loader is not None
    
    # Check batch size
    for batch in train_loader:
        assert batch["input_ids"].shape[0] <= 2  # Batch size might be smaller for the last batch
        break