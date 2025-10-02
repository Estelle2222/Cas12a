#!/usr/bin/env python3
"""
Quick start script for training the DNA BERT model.
"""

from bert_dna_model import DNABertModel

def main():
    """Train BERT model with default settings."""
    
    # File path
    excel_path = '34bp.xlsx'
    
    # Initialize model
    print("Initializing DNA BERT model...")
    print("  - Vocabulary: A, T, G, C + special tokens")
    print("  - Architecture: 4-layer BERT with 4 attention heads")
    print("  - Task: Regression (Indel prediction)")
    
    dna_bert = DNABertModel(vocab_size=9, max_length=36, num_labels=1)
    
    # Prepare data
    print(f"\nLoading data from {excel_path}...")
    train_dataset, test_dataset = dna_bert.prepare_data(excel_path)
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    
    # Train model
    print("\nStarting training...")
    print("  - Epochs: 10")
    print("  - Batch size: 32")
    print("  - Learning rate: 2e-5")
    
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
    print("\nTest Results:")
    print(f"  - MSE: {eval_results.get('eval_mse', 'N/A'):.4f}")
    print(f"  - RMSE: {eval_results.get('eval_rmse', 'N/A'):.4f}")
    print(f"  - R²: {eval_results.get('eval_r2', 'N/A'):.4f}")
    
    # Save model
    print("\nSaving model to ./dna_bert_model...")
    dna_bert.save('./dna_bert_model')
    
    # Example predictions
    print("\n" + "="*60)
    print("Example predictions on training samples:")
    print("="*60)
    sample_sequences = train_dataset.sequences[:5]
    predictions = dna_bert.predict(sample_sequences)
    for i, (seq, pred, true) in enumerate(zip(sample_sequences, predictions, train_dataset.labels[:5]), 1):
        print(f"\nSample {i}:")
        print(f"  Sequence: {seq}")
        print(f"  Predicted Indel: {pred:.2f}")
        print(f"  True Indel: {true:.2f}")
        print(f"  Error: {abs(pred - true):.2f}")
    
    print("\n" + "="*60)
    print("Training complete! Model saved to ./dna_bert_model/")
    print("="*60)
    
    # Usage instructions
    print("\nTo use the trained model:")
    print("  from bert_dna_model import DNABertModel")
    print("  model = DNABertModel()")
    print("  model.load('./dna_bert_model')")
    print("  predictions = model.predict(['AGCGTTTAAAAAACATCGAACGCATCTGCTGCCT'])")
    

if __name__ == "__main__":
    main()
