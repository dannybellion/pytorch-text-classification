# Text Classification Using TinyBERT

This project demonstrates how to fine-tune a TinyBERT model for text classification of loan defaults. 

## Project Overview

TinyBERT is a smaller, faster version of BERT that maintains comparable performance while reducing model size and computational requirements. This project uses TinyBERT to predict whether a business will default on a loan based on textual data.

## Features

- Fine-tune TinyBERT for binary text classification
- Dataset of 50 business descriptions with default/no-default labels
- Complete pipeline from data preprocessing to model evaluation
- Easy-to-use command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-classification.git
cd text-classification

# Install the package and dependencies
pip install -e .
```

## Dataset

The dataset contains 50 entries, each with:
- `text`: Business description (100-200 words)
- `label`: Binary label indicating loan default status (1 = default, 0 = no default)

## Usage

### Training

```bash
python -m src.main train \
    --data_path data/loan_default_dataset.json \
    --output_dir models \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 5
```

### Evaluation

```bash
python -m src.main evaluate \
    --model_path models/best_model.pt \
    --data_path data/loan_default_dataset.json \
    --output_dir evaluation_results
```

### Making Predictions

```bash
python -m src.predict \
    --model_path models/best_model.pt \
    --input_file your_input_file.txt
```

Or for interactive mode:

```bash
python -m src.predict --model_path models/best_model.pt
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
│   └── main.py                    # CLI entry point
├── tests/                         # Unit tests
├── pyproject.toml                 # Project configuration
└── README.md                      # Project documentation
```

## Run Tests

```bash
pytest
```

## License

MIT