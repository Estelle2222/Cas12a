# DNA BERT Model for Cas12a

This repository contains a BERT-based model for analyzing DNA sequences (34bp) with A, T, G, C nucleotides.

## Overview

The model predicts Indel values for DNA sequences using a custom BERT architecture specifically designed for genomic data.

## Features

- **Custom DNA Tokenizer**: Tokenizes DNA sequences using A, T, G, C as individual tokens
- **BERT Architecture**: 4-layer transformer with 4 attention heads optimized for 34bp sequences
- **Regression Task**: Predicts Indel values from DNA sequences
- **Special Tokens**: Uses [CLS], [SEP], [PAD], [UNK], and [MASK] tokens

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The model expects an Excel file (`34bp.xlsx`) with the following columns:
- `sequence`: DNA sequences (34 base pairs, containing only A, T, G, C)
- `Indel`: Target values for prediction

## Usage

### Training the Model

Run the main script to train the BERT model:

```bash
python bert_dna_model.py
```

This will:
1. Load data from `34bp.xlsx`
2. Split data into training (80%) and test (20%) sets
3. Train the BERT model for 10 epochs
4. Evaluate the model on the test set
5. Save the trained model to `./dna_bert_model/`

### Model Architecture

- **Vocabulary Size**: 9 tokens (4 nucleotides + 5 special tokens)
- **Hidden Size**: 128
- **Number of Layers**: 4
- **Attention Heads**: 4
- **Max Sequence Length**: 36 (34bp + [CLS] + [SEP])
- **Task**: Regression (Indel prediction)

### Using the Trained Model

```python
from bert_dna_model import DNABertModel

# Load trained model
model = DNABertModel()
model.load('./dna_bert_model')

# Make predictions
sequences = ['AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT']
predictions = model.predict(sequences)
print(f"Predicted Indel: {predictions[0]}")
```

### Running Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

This will demonstrate:
- Basic model usage for predictions
- Batch prediction on multiple sequences
- Understanding the tokenizer
- Training a model from scratch (optional)

### Custom Training

You can customize training parameters:

```python
from bert_dna_model import DNABertModel

# Initialize model
dna_bert = DNABertModel(vocab_size=9, max_length=36, num_labels=1)

# Prepare data
train_dataset, test_dataset = dna_bert.prepare_data('34bp.xlsx')

# Train with custom parameters
trainer = dna_bert.train(
    train_dataset, 
    test_dataset, 
    output_dir='./custom_model',
    num_epochs=20,
    batch_size=64,
    learning_rate=3e-5
)
```

## Model Outputs

The trained model provides:
- **Predictions**: Indel values for DNA sequences
- **Evaluation Metrics**: MSE, RMSE, and R² score
- **Saved Artifacts**: Model weights and tokenizer vocabulary

## File Structure

```
.
├── 34bp.xlsx                  # Input data
├── bert_dna_model.py          # Main model implementation
├── train_model.py             # Training script
├── example_usage.py           # Usage examples
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── dna_bert_model/           # Trained model directory (after training)
    ├── config.json
    ├── model.safetensors
    └── tokenizer_vocab.json
```

## Tokenizer Details

The custom DNA tokenizer uses the following vocabulary:

| Token  | ID | Description                    |
|--------|----|--------------------------------|
| [PAD]  | 0  | Padding token                  |
| [UNK]  | 1  | Unknown token                  |
| [CLS]  | 2  | Start of sequence token        |
| [SEP]  | 3  | End of sequence token          |
| [MASK] | 4  | Mask token (for MLM pretraining)|
| A      | 5  | Adenine nucleotide             |
| T      | 6  | Thymine nucleotide             |
| G      | 7  | Guanine nucleotide             |
| C      | 8  | Cytosine nucleotide            |

## Training Output

During training, you'll see:
- Training progress with loss values
- Evaluation metrics at the end of each epoch
- Final test set performance (MSE, RMSE, R²)
- Example predictions on sample sequences

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Pandas 2.0+
- scikit-learn 1.3+
- openpyxl 3.0+
