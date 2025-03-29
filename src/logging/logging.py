from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.pretty import Pretty

from src.config import Hyperparameters

console = Console()

def log_print(body: str, colour: str = None):
  console.print(f"\n[bold {colour}]{body}[/bold {colour}]")

def log_panel(colour: str, body: str, title: str = None):

  console.print(
    Panel(
      f"[bold {colour}]{body}[/bold {colour}]", 
      title=title, 
      border_style=colour
    )
  )
    
    
def log_hyperparameters(hyperparameters: Hyperparameters):
  """Prints the hyperparameters in a formatted table.

  Args:
    hyperparameters: A dictionary containing the hyperparameters.
  """

  table = Table(title="Training Configuration")
  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="yellow")

  for key, value in hyperparameters.__dict__.items():
    table.add_row(key, str(value))

  console.print(table)
  

def log_metrics(train_loss, test_loss, metrics) -> None:
  
  console.print(f"[bold white on black]╔══════════════════════════════════════╗[/]")
  console.print(f"[bold white on black]║ [green]Train Loss:[/green] {train_loss:.4f}  [blue]Test Loss:[/blue] {test_loss:.4f} ║[/]")
  console.print(f"[bold white on black]╚══════════════════════════════════════╝[/]")
  console.print(f"[yellow]Metrics:[/yellow] Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
  

def log_progress(title: str, colour: str) -> Progress:
  progress = Progress(
        TextColumn(f"[bold {colour}]{title}[/bold {colour}]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1
    )
  return progress
        
def log_error(message: str, error: Exception, context: Optional[Dict] = None):
    """
    Log an error with a rich panel.
    
    Args:
        message (str): Error message
        error (Exception): The exception that occurred
        context (Optional[Dict]): Additional context information
    """
    context_str = ""
    if context:
        context_str = "\n\n[bold]Context:[/bold]"
        context_str += f"\n{Pretty(context)}"
        
    console.print(Panel(
        f"[bold red]ERROR[/bold red]: {message}\n{str(error)}{context_str}",
        border_style="red",
        title="Error",
        expand=False
    ))