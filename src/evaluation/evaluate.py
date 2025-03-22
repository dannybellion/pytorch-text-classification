"""Evaluation script for the TinyBERT classifier."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import create_dataloaders, load_dataset, split_dataset
from src.training.model import TinyBERTClassifier

# Create a console for rich output
console = Console()


def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion matrix'):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)


def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)


def evaluate_model(
    model_path: str,
    data_path: str,
    model_name: str = "huawei-noah/TinyBERT_General_6L_768D",
    batch_size: int = 8,
    test_size: float = 0.2,
    output_dir: str = "evaluation_results"
):
    """Evaluate the trained model and generate performance reports.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_path: Path to the dataset JSON file
        model_name: Name of the pre-trained TinyBERT model
        batch_size: Batch size for evaluation
        test_size: Proportion of data to use for testing
        output_dir: Directory to save evaluation results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and split dataset
    df = load_dataset(data_path)
    train_df, test_df = split_dataset(df, test_size=test_size)
    
    # Create dataloaders
    _, test_loader = create_dataloaders(
        train_df, test_df, tokenizer_name=model_name, batch_size=batch_size
    )
    
    # Initialize model
    model = TinyBERTClassifier(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    class_names = ["Not Defaulted", "Defaulted"]
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Create a rich table for classification report
    table = Table(title="Classification Report")
    table.add_column("Class", style="cyan")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="green")
    table.add_column("F1-Score", style="yellow")
    table.add_column("Support", style="blue")
    
    for class_name in class_names:
        table.add_row(
            class_name,
            f"{report_dict[class_name]['precision']:.4f}",
            f"{report_dict[class_name]['recall']:.4f}",
            f"{report_dict[class_name]['f1-score']:.4f}",
            str(int(report_dict[class_name]['support']))
        )
    
    # Add average row
    table.add_row(
        "Avg / Total",
        f"{report_dict['macro avg']['precision']:.4f}",
        f"{report_dict['macro avg']['recall']:.4f}",
        f"{report_dict['macro avg']['f1-score']:.4f}",
        str(int(report_dict['macro avg']['support']))
    )
    
    console.print(table)
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create a rich table for confusion matrix
    cm_table = Table(title="Confusion Matrix")
    cm_table.add_column("", style="bold")
    for class_name in class_names:
        cm_table.add_column(f"Pred: {class_name}", style="cyan")
    
    for i, class_name in enumerate(class_names):
        row = [f"True: {class_name}"]
        for j in range(len(class_names)):
            cell_style = "green" if i == j else "red"
            row.append(f"[{cell_style}]{cm[i, j]}[/{cell_style}]")
        cm_table.add_row(*row)
    
    console.print(cm_table)
    
    plot_confusion_matrix(
        cm, 
        classes=class_names, 
        output_path=output_dir / "confusion_matrix.png"
    )
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plot_roc_curve(
        fpr, 
        tpr, 
        roc_auc,
        output_path=output_dir / "roc_curve.png"
    )
    
    # Print ROC AUC with rich formatting
    console.print(Panel(
        f"[bold yellow]ROC AUC:[/bold yellow] [cyan]{roc_auc:.4f}[/cyan]",
        title="Area Under ROC Curve",
        border_style="yellow"
    ))