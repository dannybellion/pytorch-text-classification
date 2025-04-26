# Text Classification Using DistilBERT

This project demonstrates how to fine-tune a DistilBERT model using pytorch for text classification of loan defaults. 

## Project Overview

DistilBERT is a smaller, faster version of BERT that maintains comparable performance while reducing model size and computational requirements. This project uses DistilBERT to predict whether a business will default on a loan based on textual data.

## Features

- Fine-tune DistilBERT for binary text classification
- Dataset of 80 business descriptions with default/no-default labels
- Complete pipeline from data preprocessing to model evaluation
- Easy-to-use interactive interface with hardcoded hyperparameters

## Installation

```bash
# Clone the repository
git clone https://github.com/dannybellion/text-classification.git
cd text-classification

# Install the package and dependencies
uv sync
```

## Dataset

Dummy dataset contains a list of 80 entries, each with:
- `text`: Business description (100-200 words)
- `label`: Binary label indicating loan default status (1 = default, 0 = no default)

## Usage

### Training and Evaluation

You can run the project with:

```bash
uv run -m src.main
```

This will:

**Train the model**
Run through the specified number of epochs, saving the best and most recent models after each epoch. Print out the training and validation loss for each epoch as well as other standard performance metrics.

**Evaluate the model**
Load the best model and evaluate it on the test set. Print out the test loss and other standard performance metrics.

### Making Predictions

To make predictions on new texts, use:

```bash
uv run -m src.predict
```

This will start an interactive mode where you can enter business descriptions and get predictions.

To batch process texts from a file, edit the `PREDICTION_CONFIG` at the top of `src/predict.py` to set the `input_file` parameter.

## Customizing Hyperparameters

To change hyperparameters, edit the `Hyperparameters` dataclass in `src/config.py`:

```python
@dataclass
class Hyperparameters:
    """Dataclass for storing model hyperparameters and configuration."""
    # Data paths
    data_path: str = "data/loan_default_dataset.json"
    output_dir: str = "models"
    
    # Model configuration
    model_name: str = "distilbert/distilbert-base-uncased"
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 2
    test_size: float = 0.4
    early_stop_threshold: float = 0.05
```

## Project Structure

```
text-classification/
├── data/
│   └── loan_default_dataset.json  # Dataset with 80 samples
├── src/
│   ├── config.py                  # Hyperparameters configuration
│   ├── data_preprocessing/        # Data loading and processing
│   ├── training/                  # Model definition and training
│   ├── evaluation/                # Model evaluation
│   ├── logging/                   # Logging utilities
│   ├── predict.py                 # Inference script
│   └── main.py                    # Main entry point
├── tests/                         # Unit tests
├── pyproject.toml                 # Project configuration
└── README.md                      # Project documentation
```
