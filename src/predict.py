"""Script for making predictions using a trained TinyBERT model."""
import argparse
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from src.training.model import TinyBERTClassifier


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


def main(args):
    """Main prediction function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = TinyBERTClassifier(model_name=args.model_name)
    model.load_state_dict(torch.load(args.model_path, map_location=device)["model_state_dict"])
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Get input text
    if args.input_file:
        with open(args.input_file, "r") as f:
            texts = [line.strip() for line in f.readlines()]
        results = batch_predict(model, tokenizer, texts, device)
        
        for i, result in enumerate(results):
            print(f"Text {i+1}:")
            print(f"Prediction: {result['prediction_text']}")
            print(f"Probability: {result['probability']:.4f}")
            print()
    else:
        # Interactive mode
        while True:
            text = input("Enter text to classify (or 'q' to quit): ")
            if text.lower() == 'q':
                break
            
            result = predict_text(model, tokenizer, text, device)
            print(f"Prediction: {result['prediction_text']}")
            print(f"Probability: {result['probability']:.4f}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with TinyBERT loan default classifier")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the saved model checkpoint")
    parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_6L_768D", 
                        help="TinyBERT model name")
    parser.add_argument("--input_file", type=str, 
                        help="Path to file with input texts (one per line)")
    
    args = parser.parse_args()
    main(args)