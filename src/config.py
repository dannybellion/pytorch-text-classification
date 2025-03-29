from dataclasses import dataclass

@dataclass
class Hyperparameters:
    """Dataclass for storing model hyperparameters and configuration."""
    # Data paths
    data_path: str = "data/loan_default_dataset.json"
    output_dir: str = "models"
    
    # Model configuration
    model_name: str = "huawei-noah/TinyBERT_General_6L_768D"
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 2
    test_size: float = 0.4
    early_stop_threshold: float = 0.1