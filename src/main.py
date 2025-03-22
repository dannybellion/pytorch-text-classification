"""Main entry point for the loan default classification project."""
import argparse
from pathlib import Path

from src.evaluation.evaluate import evaluate_model
from src.training.train import main as train_main


def main():
    """Parse arguments and execute the appropriate command."""
    parser = argparse.ArgumentParser(
        description="TinyBERT Loan Default Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_path", type=str, default="data/loan_default_dataset.json", 
                            help="Path to dataset JSON file")
    train_parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_6L_768D", 
                            help="TinyBERT model name")
    train_parser.add_argument("--output_dir", type=str, default="models", 
                            help="Directory to save model checkpoints")
    train_parser.add_argument("--batch_size", type=int, default=8, 
                            help="Batch size for training and evaluation")
    train_parser.add_argument("--learning_rate", type=float, default=2e-5, 
                            help="Learning rate")
    train_parser.add_argument("--num_epochs", type=int, default=5, 
                            help="Number of training epochs")
    train_parser.add_argument("--test_size", type=float, default=0.2, 
                            help="Proportion of data for test set")
    train_parser.add_argument("--val_size", type=float, default=0.1, 
                            help="Proportion of data for validation set")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model_path", type=str, required=True, 
                           help="Path to the saved model checkpoint")
    eval_parser.add_argument("--data_path", type=str, default="data/loan_default_dataset.json", 
                           help="Path to dataset JSON file")
    eval_parser.add_argument("--model_name", type=str, default="huawei-noah/TinyBERT_General_6L_768D", 
                           help="TinyBERT model name")
    eval_parser.add_argument("--batch_size", type=int, default=8, 
                           help="Batch size for evaluation")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                           help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_main(args)
    elif args.command == "evaluate":
        evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()