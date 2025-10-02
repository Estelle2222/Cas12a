#!/usr/bin/env python3
"""
BERT Model for DNA Sequences
This script builds and trains a BERT model for DNA sequences (34bp) with A, T, G, C tokens.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os


class DNATokenizer:
    """Custom tokenizer for DNA sequences with A, T, G, C tokens."""
    
    def __init__(self):
        # Define vocabulary: special tokens + nucleotides
        self.vocab = {
            '[PAD]': 0,   # Padding token
            '[UNK]': 1,   # Unknown token
            '[CLS]': 2,   # Classification token (start of sequence)
            '[SEP]': 3,   # Separator token (end of sequence)
            '[MASK]': 4,  # Mask token for MLM
            'A': 5,
            'T': 6,
            'G': 7,
            'C': 8
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def encode(self, sequence, max_length=36):
        """
        Encode DNA sequence to token IDs.
        max_length = 34 (sequence) + 2 ([CLS] and [SEP])
        """
        # Add [CLS] at start and [SEP] at end
        tokens = ['[CLS]'] + list(sequence) + ['[SEP]']
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Pad if necessary
        if len(input_ids) < max_length:
            input_ids += [self.vocab['[PAD]']] * (max_length - len(input_ids))
        else:
            input_ids = input_ids[:max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if id != self.vocab['[PAD]'] else 0 for id in input_ids]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, input_ids):
        """Decode token IDs back to sequence."""
        tokens = [self.id_to_token.get(id, '[UNK]') for id in input_ids]
        # Remove special tokens
        sequence = ''.join([t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']])
        return sequence


class DNADataset(Dataset):
    """PyTorch Dataset for DNA sequences."""
    
    def __init__(self, sequences, labels, tokenizer, max_length=36):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize sequence
        encoded = self.tokenizer.encode(sequence, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class DNABertModel:
    """BERT model wrapper for DNA sequence analysis."""
    
    def __init__(self, vocab_size=9, max_length=36, num_labels=1):
        self.tokenizer = DNATokenizer()
        self.max_length = max_length
        
        # Configure BERT model
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,           # Smaller hidden size for DNA sequences
            num_hidden_layers=4,       # 4 transformer layers
            num_attention_heads=4,     # 4 attention heads
            intermediate_size=512,     # Feedforward network size
            max_position_embeddings=max_length,
            num_labels=num_labels,     # 1 for regression (Indel prediction)
            problem_type="regression"  # Regression task
        )
        
        self.model = BertForSequenceClassification(config)
        
    def prepare_data(self, excel_path, test_size=0.2, random_state=42):
        """Load and prepare data from Excel file."""
        # Load data
        df = pd.read_excel(excel_path)
        
        # Split data
        sequences = df['sequence'].values
        labels = df['Indel'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=random_state
        )
        
        # Create datasets
        train_dataset = DNADataset(X_train, y_train, self.tokenizer, self.max_length)
        test_dataset = DNADataset(X_test, y_test, self.tokenizer, self.max_length)
        
        return train_dataset, test_dataset
    
    def train(self, train_dataset, test_dataset, output_dir='./dna_bert_model', 
              num_epochs=10, batch_size=32, learning_rate=2e-5):
        """Train the BERT model."""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none"  # Disable wandb/tensorboard
        )
        
        # Compute metrics function for evaluation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.squeeze()
            mse = mean_squared_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def predict(self, sequences):
        """Make predictions on new sequences."""
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for seq in sequences:
                encoded = self.tokenizer.encode(seq, max_length=self.max_length)
                input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long)
                attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = outputs.logits.squeeze().item()
                predictions.append(pred)
        
        return predictions
    
    def save(self, path):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        # Save tokenizer vocab
        import json
        with open(os.path.join(path, 'tokenizer_vocab.json'), 'w') as f:
            json.dump(self.tokenizer.vocab, f)
    
    def load(self, path):
        """Load model and tokenizer."""
        self.model = BertForSequenceClassification.from_pretrained(path)
        # Load tokenizer vocab
        import json
        with open(os.path.join(path, 'tokenizer_vocab.json'), 'r') as f:
            self.tokenizer.vocab = json.load(f)
            self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.vocab.items()}


def main():
    """Main function to train BERT model on DNA sequences."""
    
    # File path
    excel_path = '34bp.xlsx'
    
    # Initialize model
    print("Initializing DNA BERT model...")
    dna_bert = DNABertModel(vocab_size=9, max_length=36, num_labels=1)
    
    # Prepare data
    print(f"Loading data from {excel_path}...")
    train_dataset, test_dataset = dna_bert.prepare_data(excel_path)
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Train model
    trainer = dna_bert.train(
        train_dataset, 
        test_dataset, 
        output_dir='./dna_bert_model',
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    eval_results = trainer.evaluate()
    print(f"Test Results: {eval_results}")
    
    # Save model
    print("\nSaving model...")
    dna_bert.save('./dna_bert_model')
    
    # Example prediction
    print("\nExample predictions:")
    sample_sequences = train_dataset.sequences[:5]
    predictions = dna_bert.predict(sample_sequences)
    for seq, pred, true in zip(sample_sequences, predictions, train_dataset.labels[:5]):
        print(f"Sequence: {seq}")
        print(f"Predicted Indel: {pred:.2f}, True Indel: {true:.2f}\n")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
