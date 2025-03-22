"""Evaluation script for the TinyBERT classifier."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import create_dataloaders, load_dataset, split_dataset
from src.training.model import TinyBERTClassifier


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
    output_dir: str = "evaluation_results"
):
    """Evaluate the trained model and generate performance reports.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_path: Path to the dataset JSON file
        model_name: Name of the pre-trained TinyBERT model
        batch_size: Batch size for evaluation
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
    train_df, val_df, test_df = split_dataset(df)
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer_name=model_name, batch_size=batch_size
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
    print("Classification Report:")
    print(report)
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
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
    
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinyBERT loan default classifier")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the saved model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/loan_default_dataset.json", 
                        help="Path to dataset JSON file")
    parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_6L_768D", 
                        help="TinyBERT model name")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )