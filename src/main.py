"""Main entry point for the loan default classification project."""
from pathlib import Path

from src.evaluation.evaluate import evaluate_model
from src.training.train import main as train_main
from src.training.train import argparse


# Define all hyperparameters and paths here
HYPERPARAMETERS = {
    # Data paths
    "data_path": "data/loan_default_dataset.json",
    "output_dir": "models",
    "eval_output_dir": "evaluation_results",
    
    # Model configuration
    "model_name": "huawei-noah/TinyBERT_General_6L_768D",
    
    # Training hyperparameters
    "batch_size": 8,
    "learning_rate": 3e-5,
    "num_epochs": 5,
    "test_size": 0.4,
}


def train_model():
    """Train the model with the predefined hyperparameters."""
    print("Starting model training...")
    
    # Create a namespace object to mimic argparse behavior
    args = argparse.Namespace()
    args.data_path = HYPERPARAMETERS["data_path"]
    args.model_name = HYPERPARAMETERS["model_name"]
    args.output_dir = HYPERPARAMETERS["output_dir"]
    args.batch_size = HYPERPARAMETERS["batch_size"]
    args.learning_rate = HYPERPARAMETERS["learning_rate"]
    args.num_epochs = HYPERPARAMETERS["num_epochs"]
    args.test_size = HYPERPARAMETERS["test_size"]
    
    # Run training with predefined hyperparameters
    train_main(args)
    print(f"Training completed. Model saved to {HYPERPARAMETERS['output_dir']}")


def evaluate_trained_model():
    """Evaluate the best trained model."""
    print("Starting model evaluation...")
    
    model_path = Path(HYPERPARAMETERS["output_dir"]) / "best_model.pt"
    
    # Ensure the model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python -m src.main")
        return
    
    # Run evaluation
    evaluate_model(
        model_path=str(model_path),
        data_path=HYPERPARAMETERS["data_path"],
        model_name=HYPERPARAMETERS["model_name"],
        batch_size=HYPERPARAMETERS["batch_size"],
        output_dir=HYPERPARAMETERS["eval_output_dir"]
    )
    
    print(f"Evaluation completed. Results saved to {HYPERPARAMETERS['eval_output_dir']}")


def main():
    """Main function to either train or evaluate the model."""
    print("TinyBERT Loan Default Classification")
    print("-----------------------------------")

    train_model()
    evaluate_trained_model()


if __name__ == "__main__":
    main()