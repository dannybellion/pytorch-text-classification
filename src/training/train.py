"""Training script for TinyBERT classifier."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_preprocessing.dataset import create_dataloaders, load_dataset, split_dataset
from src.training.model import TinyBERTClassifier


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss, _ = model(input_ids, attention_mask, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            loss, logits = model(input_ids, attention_mask, labels)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            total_loss += loss.item()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return total_loss / len(dataloader), metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    output_dir: Path = Path("models")
) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """Train the model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        output_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (train_losses, val_losses, val_metrics)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses = []
    val_losses = []
    val_metrics = []
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, metrics = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_metrics.append(metrics)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {metrics}")
        
        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": metrics,
                },
                output_dir / "best_model.pt"
            )
            print(f"Saved best model with F1: {best_f1:.4f}")
        
        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": metrics,
            },
            output_dir / "latest_model.pt"
        )
    
    return train_losses, val_losses, val_metrics


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split dataset
    df = load_dataset(args.data_path)
    train_df, val_df, test_df = split_dataset(df, test_size=args.test_size, val_size=args.val_size)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer_name=args.model_name, batch_size=args.batch_size
    )
    
    # Initialize model
    model = TinyBERTClassifier(model_name=args.model_name)
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train model
    output_dir = Path(args.output_dir)
    train(
        model, train_loader, val_loader, optimizer, device, 
        num_epochs=args.num_epochs, output_dir=output_dir
    )
    
    # Evaluate on test set
    model.load_state_dict(torch.load(output_dir / "best_model.pt")["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyBERT for loan default classification")
    parser.add_argument("--data_path", type=str, default="data/loan_default_dataset.json", 
                        help="Path to dataset JSON file")
    parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_6L_768D", 
                        help="TinyBERT model name")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data for test set")
    parser.add_argument("--val_size", type=float, default=0.1, 
                        help="Proportion of data for validation set")
    
    args = parser.parse_args()
    main(args)