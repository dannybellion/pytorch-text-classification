"""Evaluation functions for the TinyBERT classifier."""
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.utils.data import DataLoader

# Create a console for rich output
console = Console()


def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion matrix'):
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
    """
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
    """Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates 
        roc_auc: Area under the ROC curve
        output_path: Path to save the plot
    """
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


def get_basic_metrics(all_labels: List[int], all_preds: List[int]) -> Dict[str, float]:
    """Calculate basic classification metrics.
    
    Args:
        all_labels: List of ground truth labels
        all_preds: List of predicted labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Error calculating precision/recall: {e}")
        # If only one class is present, set metrics accordingly
        if len(np.unique(all_preds)) == 1:
            if np.unique(all_preds)[0] == np.unique(all_labels)[0]:
                precision, recall, f1 = 1.0, 1.0, 1.0
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def generate_detailed_evaluation(
    all_labels: List[int], 
    all_preds: List[int], 
    all_probs: List[float], 
    output_dir: Path,
    class_names: List[str] = ["Not Defaulted", "Defaulted"]
) -> None:
    """Generate detailed evaluation reports and visualizations.
    
    Args:
        all_labels: List of ground truth labels
        all_preds: List of predicted labels
        all_probs: List of prediction probabilities
        output_dir: Directory to save results
        class_names: List of class names
    """
    console.print("\n")
    console.print(
        Panel("[bold green]Starting model evaluation...[/bold green]", 
        title="Evaluation", border_style="green")
    )
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm, 
        classes=class_names, 
        output_path=output_dir / "confusion_matrix.png"
    )
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save ROC curve
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