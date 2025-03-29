# Text Classification Using TinyBERT

This project demonstrates how to fine-tune a TinyBERT model for text classification of loan defaults. 

## Project Overview

TinyBERT is a smaller, faster version of BERT that maintains comparable performance while reducing model size and computational requirements. This project uses TinyBERT to predict whether a business will default on a loan based on textual data.

## Features

- Fine-tune TinyBERT for binary text classification
- Dataset of 50 business descriptions with default/no-default labels
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

The hyperparameters are hardcoded at the top of `src/main.py`. You can run the project with:

```bash
python -m src.main
```

This will:

**Train the model**
Run through the specified number of epochs, saving the best and most recent models after each epoch. Print out the training and validation loss for each epoch as well as other standard performance metrics.

**Evaluate the model**
Load the best model and evaluate it on the test set. Print out the test loss and other standard performance metrics.

### Making Predictions

To make predictions on new texts, use:

```bash
python -m src.predict
```

This will start an interactive mode where you can enter business descriptions and get predictions.

To batch process texts from a file, edit the `PREDICTION_CONFIG` at the top of `src/predict.py` to set the `input_file` parameter.

## Customizing Hyperparameters

To change hyperparameters, edit the `HYPERPARAMETERS` dictionary in `src/main.py`:

```python
HYPERPARAMETERS = {
    # Data paths
    "data_path": "data/loan_default_dataset.json",
    "output_dir": "models",
    "eval_output_dir": "evaluation_results",
    
    # Model configuration
    "model_name": "huawei-noah/TinyBERT_General_6L_768D",
    
    # Training hyperparameters
    "batch_size": 16,
    "learning_rate": 3e-5,
    "num_epochs": 10,
    "test_size": 0.2,
    "val_size": 0.1,
}
```

## Project Structure

```
text-classification/
├── data/
│   └── loan_default_dataset.json  # Dataset with 50 samples
├── src/
│   ├── data_preprocessing/        # Data loading and processing
│   ├── training/                  # Model definition and training
│   ├── evaluation/                # Model evaluation
│   ├── predict.py                 # Inference script
│   └── main.py                    # Main entry point with hardcoded hyperparameters
├── tests/                         # Unit tests
├── pyproject.toml                 # Project configuration
└── README.md                      # Project documentation
```