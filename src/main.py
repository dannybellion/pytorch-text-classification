"""Main entry point for the loan default classification project."""
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import TextColumn
from rich.table import Table

from src.evaluation.evaluate import evaluate_model
from src.training.train import main as train_main
from src.training.train import argparse

# Create a console for rich output
console = Console()


@dataclass
class Hyperparameters:
    """Dataclass for storing model hyperparameters and configuration."""
    # Data paths
    data_path: str = "data/loan_default_dataset.json"
    output_dir: str = "models"
    eval_output_dir: str = "evaluation_results"
    
    # Model configuration
    model_name: str = "huawei-noah/TinyBERT_General_6L_768D"
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 2
    test_size: float = 0.4


# Create a single instance of the hyperparameters
HYPERPARAMETERS = Hyperparameters()


def train_model():
    """Train the model with the predefined hyperparameters."""
    console.print(Panel("[bold green]Starting model training...[/bold green]", 
                       title="Training", border_style="green"))
    
    # Display hyperparameters in a table
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in vars(HYPERPARAMETERS).items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    # Create a namespace object to mimic argparse behavior
    args = argparse.Namespace(**vars(HYPERPARAMETERS))
    
    # Run training with predefined hyperparameters
    train_main(args)
    
    console.print(f"\n[bold green]✓[/bold green] Training completed. Model saved to [bold]{HYPERPARAMETERS.output_dir}[/bold]")


def evaluate_trained_model():
    """Evaluate the best trained model."""
    console.print("\n\n")
    console.print(Panel("[bold blue]Starting model evaluation...[/bold blue]", 
                       title="Evaluation", border_style="blue"))
    
    model_path = Path(HYPERPARAMETERS.output_dir) / "best_model.pt"
    
    # Ensure the model exists
    if not model_path.exists():
        console.print(f"[bold red]Error:[/bold red] Model file not found at {model_path}")
        console.print("[yellow]Please train the model first by running:[/yellow] python -m src.main")
        return
    
    # Run evaluation
    evaluate_model(
        model_path=str(model_path),
        data_path=HYPERPARAMETERS.data_path,
        model_name=HYPERPARAMETERS.model_name,
        batch_size=HYPERPARAMETERS.batch_size,
        test_size=HYPERPARAMETERS.test_size,
        output_dir=HYPERPARAMETERS.eval_output_dir
    )
    
    console.print(f"[bold blue]✓[/bold blue] Evaluation completed. Results saved to [bold]{HYPERPARAMETERS.eval_output_dir}[/bold]")


def main():
    """Main function to either train or evaluate the model."""
    console.print(Panel.fit(
        "[bold magenta]TinyBERT Loan Default Classification[/bold magenta]",
        border_style="magenta"
    ))

    train_model()
    evaluate_trained_model()


if __name__ == "__main__":
    main()