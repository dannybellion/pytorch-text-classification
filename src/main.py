from src.logging.logging import log_panel, log_hyperparameters
from src.training.train import main as train_main
import argparse
from src.config import Hyperparameters


def main():
    """Main function to train the model."""
    
    log_panel(
        colour="magenta",
        body="DistilBERT Loan Default Classification", 
    )

    HYPERPARAMETERS = Hyperparameters(
        num_epochs=6
    )
    
    log_hyperparameters(HYPERPARAMETERS)
    
    args = argparse.Namespace(**vars(HYPERPARAMETERS))
    
    train_main(args)

if __name__ == "__main__":
    main()