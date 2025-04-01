"""Script for making predictions using a trained DistilBERT model."""
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from src.training.model import DistilBERTClassifier


# Define prediction configuration here
PREDICTION_CONFIG = {
    "model_path": "models/best_model.pt",
    "model_name": "distilbert/distilbert-base-uncased",
    "input_file": None,  # Set to a file path to batch process texts, or None for interactive mode
}


def predict_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device
) -> Dict[str, Union[int, float]]:
    """Make a prediction for a single text input.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results
    """
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move tensors to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    result = {
        "prediction": pred_class,
        "probability": pred_prob,
        "prediction_text": "Default" if pred_class == 1 else "No Default",
    }
    
    return result


def batch_predict(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device
) -> List[Dict[str, Union[int, float]]]:
    """Make predictions for a batch of texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run inference on
        
    Returns:
        List of prediction results
    """
    results = []
    for text in texts:
        result = predict_text(model, tokenizer, text, device)
        results.append(result)
    
    return results


def main():
    """Main prediction function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    model_path = Path(PREDICTION_CONFIG["model_path"])
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python -m src.main")
        return
    
    # Load model
    model = DistilBERTClassifier(model_name=PREDICTION_CONFIG["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PREDICTION_CONFIG["model_name"])
    
    # Get input text
    if PREDICTION_CONFIG["input_file"]:
        with open(PREDICTION_CONFIG["input_file"], "r") as f:
            texts = [line.strip() for line in f.readlines()]
        
        print(f"Making predictions for {len(texts)} texts from {PREDICTION_CONFIG['input_file']}...")
        results = batch_predict(model, tokenizer, texts, device)
        
        for i, result in enumerate(results):
            print(f"Text {i+1}:")
            print(f"Prediction: {result['prediction_text']}")
            print(f"Probability: {result['probability']:.4f}")
            print()
    else:
        # Interactive mode
        print("Loan Default Prediction - Interactive Mode")
        print("Enter 'q' to quit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter business description: ")
            if text.lower() == 'q':
                break
            
            if len(text.split()) < 10:
                print("Please enter a more detailed business description (at least 10 words)")
                continue
            
            result = predict_text(model, tokenizer, text, device)
            print(f"\nPrediction: {result['prediction_text']}")
            print(f"Probability: {result['probability']:.4f}")


if __name__ == "__main__":
    main()