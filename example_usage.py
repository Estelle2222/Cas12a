#!/usr/bin/env python3
"""
Example usage of the DNA BERT model for predictions.
This script demonstrates how to use a trained model for inference.
"""

from bert_dna_model import DNABertModel

def example_basic_usage():
    """Basic example of loading and using the trained model."""
    print("Example 1: Basic Usage")
    print("-" * 60)
    
    # Load the trained model
    model = DNABertModel()
    model.load('./dna_bert_model')
    
    # Make predictions on new sequences
    sequences = [
        'AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT',
        'AAACTTTAAAAATCTTTTCTGCCAGATCTCCAGA'
    ]
    
    predictions = model.predict(sequences)
    
    for seq, pred in zip(sequences, predictions):
        print(f"Sequence: {seq}")
        print(f"Predicted Indel: {pred:.2f}\n")


def example_batch_prediction():
    """Example of batch prediction on multiple sequences."""
    print("\nExample 2: Batch Prediction")
    print("-" * 60)
    
    import pandas as pd
    
    # Load the trained model
    model = DNABertModel()
    model.load('./dna_bert_model')
    
    # Load some sequences from the original data
    df = pd.read_excel('34bp.xlsx')
    test_sequences = df['sequence'].iloc[:10].tolist()
    true_values = df['Indel'].iloc[:10].tolist()
    
    # Make predictions
    predictions = model.predict(test_sequences)
    
    # Display results
    print(f"{'Sequence':<36} {'True':<8} {'Pred':<8} {'Error':<8}")
    print("-" * 60)
    for seq, true, pred in zip(test_sequences, true_values, predictions):
        error = abs(true - pred)
        print(f"{seq} {true:>7.2f} {pred:>7.2f} {error:>7.2f}")


def example_custom_tokenizer():
    """Example of using the custom DNA tokenizer."""
    print("\nExample 3: Understanding the Tokenizer")
    print("-" * 60)
    
    from bert_dna_model import DNATokenizer
    
    tokenizer = DNATokenizer()
    
    # Example sequence
    sequence = 'ATGCATGCATGCATGCATGCATGCATGCATGCAT'
    
    # Encode the sequence
    encoded = tokenizer.encode(sequence)
    
    print(f"Original sequence: {sequence}")
    print(f"Sequence length: {len(sequence)}bp")
    print(f"\nToken IDs: {encoded['input_ids']}")
    print(f"Attention mask: {encoded['attention_mask']}")
    
    # Show token mapping
    print("\nToken mapping:")
    for i, (token_id, mask) in enumerate(zip(encoded['input_ids'], encoded['attention_mask'])):
        token = tokenizer.id_to_token[token_id]
        status = "active" if mask == 1 else "padding"
        print(f"  Position {i}: ID={token_id}, Token='{token}', Status={status}")
    
    # Decode back
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"\nDecoded sequence: {decoded}")
    print(f"Match: {decoded == sequence}")


def example_model_training():
    """Example of training a model from scratch."""
    print("\nExample 4: Training from Scratch")
    print("-" * 60)
    
    # Initialize new model
    model = DNABertModel(vocab_size=9, max_length=36, num_labels=1)
    
    # Prepare data
    train_dataset, test_dataset = model.prepare_data('34bp.xlsx', test_size=0.2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train (using fewer epochs for this example)
    print("\nTraining model with 3 epochs...")
    trainer = model.train(
        train_dataset, 
        test_dataset, 
        output_dir='./example_model',
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Evaluate
    results = trainer.evaluate()
    print(f"\nTest Results:")
    print(f"  MSE: {results.get('eval_mse', 'N/A'):.4f}")
    print(f"  RMSE: {results.get('eval_rmse', 'N/A'):.4f}")
    print(f"  R²: {results.get('eval_r2', 'N/A'):.4f}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("DNA BERT Model - Usage Examples")
    print("=" * 60)
    
    # Check if model exists
    import os
    if not os.path.exists('./dna_bert_model'):
        print("\nWarning: Trained model not found at './dna_bert_model'")
        print("Please run 'python train_model.py' first to train the model.")
        print("\nRunning tokenizer example only...\n")
        example_custom_tokenizer()
    else:
        # Run all examples
        example_basic_usage()
        example_batch_prediction()
        example_custom_tokenizer()
        
        # Optionally run training example
        # Uncomment the line below to train a new model
        # example_model_training()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
