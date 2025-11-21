import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# ============================================================
# 1. CONFIGURATION
# ============================================================
CSV_FILE = "data/cpf1energy.csv"  
SEQUENCE_COLUMN = "sequence"

MAX_LEN = 36            
BERT_HIDDEN_DIM = 256   
BERT_LAYERS = 4
BERT_HEADS = 8          

LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 32
MASK_PROB = 0.15

# ============================================================
# 2. TOKENIZER
# ============================================================
VOCAB = {
    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
    'A': 5, 'T': 6, 'G': 7, 'C': 8
}
# Reverse vocab for decoding (to print examples)
ID2TOKEN = {v: k for k, v in VOCAB.items()}

# ============================================================
# 3. DATASET
# ============================================================
class DNAMLMDataset(Dataset):
    def __init__(self, sequences, vocab, max_len, mask_prob=0.15):
        self.sequences = sequences
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = ['[CLS]'] + list(seq.upper()) + ['[SEP]']
        input_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        
        # Padding
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.vocab['[PAD]']] * padding_len
        else:
            input_ids = input_ids[:self.max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.full(input_ids.shape, -100, dtype=torch.long)
        
        # Masking Logic
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_tokens_mask = (input_ids == self.vocab['[CLS]']) | \
                              (input_ids == self.vocab['[SEP]']) | \
                              (input_ids == self.vocab['[PAD]'])
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set targets
        labels[masked_indices] = input_ids[masked_indices]
        # Set input to [MASK]
        input_ids[masked_indices] = self.vocab['[MASK]']
        
        attention_mask = (input_ids != self.vocab['[PAD]']).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ============================================================
# 4. CUSTOM PRINTER CALLBACK (So you see progress!)
# ============================================================
class PrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # This runs after every evaluation phase
        acc = metrics.get("eval_accuracy", 0)
        loss = metrics.get("eval_loss", 0)
        epoch = state.epoch
        print(f"\n>>> Epoch {epoch:.1f} | Eval Loss: {loss:.4f} | Eval Accuracy: {acc*100:.2f}%")

# ============================================================
# 5. MAIN EXECUTION
# ============================================================

# Load Data
print(f"Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
sequences = df[SEQUENCE_COLUMN].astype(str).tolist()
valid_sequences = [s for s in sequences if set(s.upper()).issubset(set('ATGC'))]

train_seqs, val_seqs = train_test_split(valid_sequences, test_size=0.1, random_state=42)
train_dataset = DNAMLMDataset(train_seqs, VOCAB, MAX_LEN, MASK_PROB)
val_dataset = DNAMLMDataset(val_seqs, VOCAB, MAX_LEN, MASK_PROB)

# --- VISUAL SANITY CHECK ---
print("\n" + "="*40)
print("   VISUALIZING MASKING PROCESS")
print("="*40)
example = train_dataset[0]
input_ids = example['input_ids'].numpy()
labels = example['labels'].numpy()

print(f"Original Tokens (Reconstructed):")
orig_tokens = []
for i, tid in enumerate(input_ids):
    token = ID2TOKEN[tid]
    # If it's masked, look at the label to find original
    if tid == VOCAB['[MASK]']:
        orig_token = ID2TOKEN[labels[i]]
        token = f"[{orig_token}]" # Highlight masked tokens
    orig_tokens.append(token)
print(" ".join(orig_tokens))

print(f"\nMasked Input (What model sees):")
masked_view = [ID2TOKEN[tid] for tid in input_ids]
print(" ".join(masked_view))
print("="*40 + "\n")

# Initialize Model
config = BertConfig(
    vocab_size=len(VOCAB),
    hidden_size=BERT_HIDDEN_DIM,
    num_hidden_layers=BERT_LAYERS,
    num_attention_heads=BERT_HEADS,
    max_position_embeddings=MAX_LEN,
    pad_token_id=VOCAB['[PAD]'],
)
model = BertForMaskedLM(config)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    active_labels = labels[mask]
    active_preds = predictions[mask]
    return {"accuracy": accuracy_score(active_labels, active_preds)}

# Training Setup
training_args = TrainingArguments(
    output_dir="./results_mlm_pretrain",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none" # Force output to console only
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrinterCallback] # <--- ADDED OUR PRINTER
)

print("Starting Training...")
trainer.train()

trainer.save_model("./bert_mlm_pretrained")
print("\n✅ MLM Pre-training Complete. Model saved.")