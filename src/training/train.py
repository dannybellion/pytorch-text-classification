"""Training script for TinyBERT classifier."""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import create_dataloaders, load_dataset, split_dataset
from src.training.model import TinyBERTClassifier

# Create a console for rich output
console = Console()


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
    total_batches = len(dataloader)
    
    progress = Progress(
        TextColumn("[bold green]Training[/bold green]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1
    )
    
    with progress:
        task = progress.add_task("Training", total=total_batches)
        
        for batch in dataloader:
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
            progress.update(task, advance=1)
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
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
    all_probs = []
    total_batches = len(dataloader)
    
    # Progress display that doesn't flicker
    progress = Progress(
        TextColumn("[bold blue]Evaluating[/bold blue]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1  # Low refresh rate to avoid flickering
    )
    
    with torch.no_grad():
        with progress:
            task = progress.add_task("Evaluating", total=total_batches)
            
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                # Forward pass
                loss, logits = model(input_ids, attention_mask, labels)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of the positive class
                
                total_loss += loss.item()
                progress.update(task, advance=1)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
    except Exception as e:
        print(f"Error calculating precision/recall: {e}")
        # If only one class is present, set metrics accordingly
        if len(np.unique(all_preds)) == 1:
            if np.unique(all_preds)[0] == np.unique(all_labels)[0]:
                precision, recall, f1 = 1.0, 1.0, 1.0
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
    
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
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    output_dir: Path = Path("models")
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Train the model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        output_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (train_losses, test_metrics)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_losses = []
    test_losses = []
    test_metrics = []
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        console.print(f"\n[bold cyan]Epoch {epoch + 1}/{num_epochs}[/bold cyan]")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        test_loss, metrics = evaluate(model, test_loader, device)
        test_losses.append(test_loss)
        test_metrics.append(metrics)
        
        # Display metrics in a formatted way
        console.print(f"[bold white on black]╔══════════════════════════════════════╗[/]")
        console.print(f"[bold white on black]║ [green]Train Loss:[/green] {train_loss:.4f}  [blue]Test Loss:[/blue] {test_loss:.4f} ║[/]")
        console.print(f"[bold white on black]╚══════════════════════════════════════╝[/]")
        console.print(f"[yellow]Metrics:[/yellow] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_loss,
                    "test_metrics": metrics,
                },
                output_dir / "best_model.pt"
            )
            console.print(f"[bold green]✓[/bold green] Saved best model with F1: {best_f1:.4f}")
        
        # Save latest model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
                "test_metrics": metrics,
            },
            output_dir / "latest_model.pt"
        )
    
    return train_losses, test_metrics


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split dataset
    df = load_dataset(args.data_path)
    train_df, test_df = split_dataset(df, test_size=args.test_size)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Test label distribution: {test_df['label'].value_counts().to_dict()}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_df, test_df, tokenizer_name=args.model_name, batch_size=args.batch_size
    )
    
    # Initialize model
    model = TinyBERTClassifier(model_name=args.model_name)
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train model
    output_dir = Path(args.output_dir)
    train(
        model, train_loader, test_loader, optimizer, device, 
        num_epochs=args.num_epochs, output_dir=output_dir
    )