from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps
from torch.utils.data import DataLoader

from src.logging.logging import log_panel, log_metrics, log_progress, log_print
from src.config import Hyperparameters
from src.data_preprocessing.dataset import create_dataloaders, load_dataset, split_dataset
from src.evaluation.evaluate import get_basic_metrics, generate_detailed_evaluation
from src.training.model import TinyBERTClassifier
from src.training.device import get_device


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    
    progress = log_progress(title="Training", colour="Green")
    
    with progress:
        task = progress.add_task("Training", total=total_batches)
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.update(task, advance=1)
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    train_loss: float = 0.0,
    final_epoch: bool = False,
    output_dir: Optional[Path] = None
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the model.
    
    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        device: Device to evaluate on
        train_loss: Training loss for the current epoch (for display purposes)
        final_epoch: Whether this is the final epoch evaluation
        output_dir: Directory to save evaluation results (required if final_epoch=True)
        
    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    total_batches = len(dataloader)
    
    progress = log_progress(title="Evaluating", colour="blue")
    
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
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                total_loss += loss.item()
                progress.update(task, advance=1)
    
    # Calculate basic metrics using the utility function from evaluate.py
    metrics = get_basic_metrics(all_labels, all_preds)
    test_loss = total_loss / len(dataloader)
    
    log_metrics(train_loss, test_loss, metrics)
    
    # If this is the final epoch, generate detailed evaluation reports
    if final_epoch and output_dir:
        eval_dir = output_dir / "evaluation_results"
        generate_detailed_evaluation(
            all_labels=all_labels,
            all_preds=all_preds,
            all_probs=all_probs,
            output_dir=eval_dir
        )
    
    return test_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    early_stop_threshold: float = 0.2,
    output_dir: Path = Path("models"),
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
    best_test_loss = 999
    
    for epoch in range(num_epochs):
        
        log_print(body=f"Epoch {epoch + 1}/{num_epochs}", colour="cyan")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        is_final_epoch = epoch == num_epochs - 1
        
        test_loss, metrics = evaluate(
            model, 
            test_loader, 
            device,
            train_loss=train_loss,
            final_epoch=is_final_epoch,
            output_dir=output_dir if is_final_epoch else None
        )
        test_losses.append(test_loss)
        test_metrics.append(metrics)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
                "test_metrics": metrics,
            }
                
            torch.save(save_dict, output_dir / "best_model.pt")
            log_print(
                body=f"Saved best model with test loss: {best_test_loss:.4f}", 
                colour="green",
            )
        
        if test_loss > best_test_loss + early_stop_threshold:
            log_print(body=f"Early stop")
            break
    
    return train_losses, test_metrics


def main(args: Hyperparameters):
    """Main training function."""
    
    log_panel(
        colour="green", 
        body= "Starting Model Training", 
        title="Training"
    )
        
    device = get_device()
    
    df = load_dataset(args.data_path)
    train_df, test_df = split_dataset(df, test_size=args.test_size)
    
    log_print(body=f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    log_print(body=f"Train label distribution: {train_df['label'].value_counts().to_dict()}")
    log_print(body=f"Test label distribution: {test_df['label'].value_counts().to_dict()}")
    
    train_loader, test_loader = create_dataloaders(
        train_df, 
        test_df, 
        tokenizer_name=args.model_name, 
        batch_size=args.batch_size
    )
    
    model = TinyBERTClassifier(model_name=args.model_name)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    output_dir = Path(args.output_dir)
    train(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        device, 
        num_epochs=args.num_epochs, 
        early_stop_threshold=args.early_stop_threshold,
        output_dir=output_dir,
    )