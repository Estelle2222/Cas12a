import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
from safetensors.torch import load_file

from transformers import (
    BertConfig,
    BertModel,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# ============================================================
# 1. CONFIGURATION
# ============================================================
CSV_FILE = "data/cpf1energy.csv"
# PRETRAINED_PATH = "./bert_mlm_pretrained/pytorch_model.bin" # <--- PATH TO MLM WEIGHTS
PRETRAINED_PATH = "./bert_mlm_pretrained/model.safetensors"

SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "efficiency"

MAX_LEN = 36
BERT_HIDDEN_DIM = 256
BERT_LAYERS = 4
BERT_HEADS = 8

# Finetuning Hyperparameters
# We use a lower LR because the BERT body is already trained
LEARNING_RATE = 5e-5 
EPOCHS = 30
BATCH_SIZE = 32

# ============================================================
# 2. TOKENIZER
# ============================================================
VOCAB = {
    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
    'A': 5, 'T': 6, 'G': 7, 'C': 8
}
VOCAB_SIZE = len(VOCAB)

# ============================================================
# 3. DATASET (REGRESSION)
# ============================================================
class RegressionDataset(Dataset):
    def __init__(self, sequences, labels, vocab, max_len):
        self.sequences = sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        tokens = ['[CLS]'] + list(seq.upper()) + ['[SEP]']
        input_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.vocab['[PAD]']] * padding_len
        else:
            input_ids = input_ids[:self.max_len]

        # Create Attention Mask (1 for real tokens, 0 for pad)
        attention_mask = [1 if i != self.vocab['[PAD]'] else 0 for i in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

# ============================================================
# 4. MODEL ARCHITECTURE (BERT-CNN-MLP)
# ============================================================
class BertCnnMlpHybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config) # <--- We will load weights into this
        
        # --- The Regression Head (Randomly Initialized) ---
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, out_channels=128, kernel_size=5)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calc flattened size: 36 -> 32 (conv) -> 16 (pool) * 80 ch = 1280
        self.fc1_in_features = 128 * 16 
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.fc1_in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 40)
        self.fc_out = nn.Linear(40, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. BERT Body
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state # (B, 36, 256)

        # 2. Regression Head
        cnn_input = embeddings.permute(0, 2, 1) # (B, 256, 36)
        x = F.relu(self.conv1(cnn_input))
        x = self.pool1(x)
        x = self.flatten(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc_out(x)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, logits) if loss is not None else logits

# ============================================================
# 5. MAIN EXECUTION
# ============================================================

# --- A. Load Data ---
print(f"Loading regression data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
sequences = df[SEQUENCE_COLUMN].astype(str).tolist()
labels_raw = df[LABEL_COLUMN].values.astype(np.float32).reshape(-1, 1)

# Scale labels
scaler = MinMaxScaler()
labels = scaler.fit_transform(labels_raw).flatten()
joblib.dump(scaler, "indel_label_scaler.pkl")

train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences, labels, test_size=0.1, random_state=42
)

train_dataset = RegressionDataset(train_seqs, train_labels, VOCAB, MAX_LEN)
val_dataset = RegressionDataset(val_seqs, val_labels, VOCAB, MAX_LEN)

# --- B. Initialize Model ---
print("Initializing Model Structure...")
config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=BERT_HIDDEN_DIM,
    num_hidden_layers=BERT_LAYERS,
    num_attention_heads=BERT_HEADS,
    max_position_embeddings=MAX_LEN,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = BertCnnMlpHybrid(config)

# # --- C. *** THE TRANSFER MAGIC *** ---

# if os.path.exists(PRETRAINED_PATH):
#     print(f"\n🔍 Found pre-trained weights at {PRETRAINED_PATH}")
#     print("   Loading weights using safetensors...")
    
#     # 1. Load the safetensors file
#     mlm_state_dict = load_file(PRETRAINED_PATH)
    
#     # 2. Filter: Keep only 'bert.' keys, ignore 'cls.' (the MLM head)
#     # This strips away the "Fill-in-the-blank" layer and keeps the "Brain"
#     bert_weights = {}
#     for key, value in mlm_state_dict.items():
#         if key.startswith('bert.'):
#             bert_weights[key] = value
    
#     # 3. Load into our model
#     # strict=False is ESSENTIAL because our model has extra layers (conv1, fc1...)
#     # that are not in the MLM file.
#     missing, unexpected = model.load_state_dict(bert_weights, strict=False)
    
#     print("✅ Weights Loaded Successfully!")
#     print(f"   Initialized from Pre-training: {len(bert_weights)} layers (BERT Body)")
#     print(f"   Randomly Initialized: CNN/MLP Head layers")
# else:
#     raise FileNotFoundError(f"❌ Could not find file: {PRETRAINED_PATH}")

# --- D. Setup Training ---
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)
    return {
        "r2": r2_score(labels, preds),
        "mse": mean_squared_error(labels, preds)
    }

class R2Printer(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"Epoch {state.epoch:.0f} | Eval R²: {metrics['eval_r2']:.4f} | Eval MSE: {metrics['eval_mse']:.4f}")

# training_args = TrainingArguments(
#     output_dir="./results_regression_finetuned",
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE * 2,
#     learning_rate=LEARNING_RATE,
#     warmup_steps=100,
#     weight_decay=0.01,
#     logging_strategy="steps",
#     logging_steps=50,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="r2",
#     greater_is_better=True,
#     fp16=torch.cuda.is_available(),
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
#     callbacks=[R2Printer]
# )
# --- C. LOAD PRE-TRAINED WEIGHTS ---
from safetensors.torch import load_file
PRETRAINED_PATH = "./bert_mlm_pretrained/model.safetensors"

if os.path.exists(PRETRAINED_PATH):
    print(f"\n🔍 Loading Smart Brain from: {PRETRAINED_PATH}")
    mlm_state_dict = load_file(PRETRAINED_PATH)
    bert_weights = {k: v for k, v in mlm_state_dict.items() if k.startswith('bert.')}
    model.load_state_dict(bert_weights, strict=False)
    print("✅ Weights Loaded!")
else:
    raise FileNotFoundError("Pre-trained weights not found.")

# ============================================================
# STAGE 1: FREEZE BERT (Train Head Only)
# ============================================================
print("\n❄️  STAGE 1: Freezing BERT Body. Training Head Only...")

# 1. Freeze BERT
for param in model.bert.parameters():
    param.requires_grad = False

# 2. Verify
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable Params: {trainable_params:,} (Head Only)")
print(f"   Frozen Params:    {total_params - trainable_params:,} (BERT Body)")

# 3. Setup Trainer for Head Training (High LR)
training_args_stage1 = TrainingArguments(
    output_dir="./results_stage1",
    num_train_epochs=10,       # Short training just to align the head
    learning_rate=1e-3,        # High LR for the random head
    per_device_train_batch_size=BATCH_SIZE,
    weight_decay=0.05,
    logging_steps=50,
    save_strategy="no",        # Don't save intermediate checkpoints for this short stage
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args_stage1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[R2Printer]
)

trainer.train()
print("✅ Stage 1 Complete. Head is now aligned.")

# ============================================================
# STAGE 2: THAW BERT (Fine-tune Everything)
# ============================================================
print("\n🔥 STAGE 2: Unfreezing BERT. Fine-tuning Entire Model...")

# 1. Unfreeze BERT
for param in model.bert.parameters():
    param.requires_grad = True

# 2. Setup Trainer for Fine-tuning (Low LR)
training_args_stage2 = TrainingArguments(
    output_dir="./results_stage2",
    num_train_epochs=30,       # Longer training
    learning_rate=5e-5,        # LOW LR to preserve pre-training
    per_device_train_batch_size=BATCH_SIZE,
    weight_decay=0.01,         # Regularization
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="r2",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
)

# We need to re-initialize the trainer to pick up the changed gradients
trainer = Trainer(
    model=model,
    args=training_args_stage2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[R2Printer]
)
# --- E. Train ---
print("\n--- Starting Regression Fine-tuning ---")
trainer.train()
print("\n--- Final Evaluation on Validation Set ---")
eval_results = trainer.evaluate()
eval_r2 = eval_results["eval_r2"]
eval_mse = eval_results["eval_mse"]

print(f"Final Validation R²: {eval_r2:.4f}")
print(f"Final Validation MSE: {eval_mse:.4f}")

# Final training evaluation
train_pred = trainer.predict(train_dataset)
train_preds = np.atleast_1d(np.squeeze(train_pred.predictions))
train_true = np.atleast_1d(np.squeeze(train_pred.label_ids))
train_r2 = r2_score(train_true, train_preds)
train_mse = mean_squared_error(train_true, train_preds)

print(f"\nFinal Training R²: {train_r2:.4f}")
print(f"Final Training MSE: {train_mse:.4f}")

print("\n--- Overfitting Diagnosis ---")
print(f"  Training R²:   {train_r2:.4f}")
print(f"  Validation R²: {eval_r2:.4f}")
print(f"  Difference:    {(train_r2 - eval_r2):.4f}")

# Save Final
trainer.save_model("./final_finetuned_model")
print("\n✅ Final Regression Model Saved!")