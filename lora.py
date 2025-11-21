import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from transformers import BertConfig, BertModel, Trainer, TrainingArguments, TrainerCallback
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. CONFIGURATION
# ============================================================
TEACHER_MODEL_PATH = "./final_finetuned_model/model.safetensors" 
CSV_FILE = "data/hek.csv" 

# LoRA Hyperparameters
LORA_R = 8             # Rank of the low-rank matrices (8 is standard)
LORA_ALPHA = 32         # Scaling factor
LORA_DROPOUT = 0.1

# Training Params (We can use a higher LR because we are training fewer params)
LEARNING_RATE = 1e-3    
EPOCHS = 200
BATCH_SIZE = 16
MAX_LEN = 36
BERT_HIDDEN_DIM = 256
VOCAB_SIZE = 9

# ============================================================
# 2. DATASET (Endogenous Multi-Modal)
# ============================================================
VOCAB = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'A': 5, 'T': 6, 'G': 7, 'C': 8}

class EndogenousDataset(Dataset):
    def __init__(self, sequences, access_features, labels, vocab, max_len):
        self.sequences = sequences
        self.access_features = access_features
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = ['[CLS]'] + list(seq.upper()) + ['[SEP]']
        input_ids = [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.vocab['[PAD]']] * padding_len
        else:
            input_ids = input_ids[:self.max_len]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != 0 else 0 for i in input_ids], dtype=torch.long),
            'accessibility_feature': torch.tensor(self.access_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ============================================================
# 3. MODEL: The Multi-Modal Gating Model
# ============================================================
class BertCnnMlp_MultiModal_Gated(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        
        # --- Teacher Architecture (Sequence Branch) ---
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, out_channels=128, kernel_size=5)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1_in_features = 128 * 16 
        self.dropout = nn.Dropout(0.3) 
        self.fc1 = nn.Linear(self.fc1_in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 40) # Output 40 dim
        
        # --- Gating Branch (Chromatin) ---
        # Expand 1 -> 40 to match sequence features
        self.chromatin_projector = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU()
        )
        
        # --- Final Head ---
        self.fc_final = nn.Linear(40, 1) 

    def forward(self, input_ids, attention_mask, accessibility_feature=None, labels=None):
        # 1. Sequence Branch
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state 
        cnn_input = embeddings.permute(0, 2, 1)
        
        x = F.relu(self.conv1(cnn_input))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        features_40d = F.relu(self.fc3(x)) # (B, 40)
        
        # 2. Chromatin Branch (The Gate)
        acc_input = accessibility_feature.unsqueeze(1)
        gate_values = self.chromatin_projector(acc_input) # (B, 40)
        
        # 3. Multiplication Merge (Gating)
        gated_features = features_40d * gate_values 
        
        # 4. Final Prediction
        logits = self.fc_final(gated_features)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, logits) if loss is not None else logits

# ============================================================
# 4. EXECUTION
# ============================================================

# --- A. Load Data ---
print("Loading Endogenous Data...")
try:
    df = pd.read_csv(CSV_FILE)
except Exception:
    raise FileNotFoundError(f"File not found: {CSV_FILE}")

# Clean and Prep
df = df.dropna(subset=['sequence', 'efficiency', 'accessibility'])
allowed_chars = set('ATGC')
df = df[df['sequence'].apply(lambda x: set(x.upper()).issubset(allowed_chars))]

sequences = df['sequence'].tolist()
# Use raw labels (no scaler)
access_features = df['accessibility'].values.astype(np.float32)
labels = df['efficiency'].values.astype(np.float32)

train_seqs, val_seqs, train_acc, val_acc, train_lbl, val_lbl = train_test_split(
    sequences, access_features, labels, test_size=0.25, random_state=42
)

train_dataset = EndogenousDataset(train_seqs, train_acc, train_lbl, VOCAB, MAX_LEN)
val_dataset = EndogenousDataset(val_seqs, val_acc, val_lbl, VOCAB, MAX_LEN)

# --- B. Initialize Base Model ---
config = BertConfig(
    vocab_size=VOCAB_SIZE, hidden_size=BERT_HIDDEN_DIM, num_hidden_layers=4, num_attention_heads=8,
    max_position_embeddings=MAX_LEN
)
model = BertCnnMlp_MultiModal_Gated(config)

# --- C. Load Teacher Weights ---
if os.path.exists(TEACHER_MODEL_PATH):
    print(f"\n🎓 Loading Teacher Weights from {TEACHER_MODEL_PATH}...")
    teacher_state = load_file(TEACHER_MODEL_PATH)
    # Filter out 'fc_out' to avoid mismatch
    filtered_state = {k: v for k, v in teacher_state.items() if "fc_out" not in k}
    model.load_state_dict(filtered_state, strict=False)
    print("✅ Weights Loaded!")
else:
    raise FileNotFoundError("Teacher model not found!")

# --- D. APPLY LORA (The Magic Step) ---
print("\n💉 Injecting LoRA Adapters...")

# 1. Define LoRA Config
# We target 'query' and 'value' projections in BERT attention layers
peft_config = LoraConfig(
    task_type=None, # Custom model, so no specific task type
    inference_mode=False, 
    r=LORA_R, 
    lora_alpha=LORA_ALPHA, 
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "value"] # These are the names inside BertSelfAttention
)

# 2. Apply LoRA to the BERT component ONLY
# This wraps model.bert into a PeftModel. It effectively freezes the main BERT weights
# and creates small trainable adapter weights.
model.bert = get_peft_model(model.bert, peft_config)
model.bert.print_trainable_parameters()

# --- E. FREEZE/UNFREEZE STRATEGY ---
# We want to:
# 1. Train LoRA Adapters (model.bert is already handling this)
# 2. Freeze the Old Head (conv1, fc1, fc2, fc3) to prevent forgetting
# 3. Train the New Head (chromatin_projector, fc_final)

print("\n🔒 Configuring Trainable Layers:")

# Freeze the "Old" Sequence Head
modules_to_freeze = [model.conv1, model.pool1, model.fc1, model.fc2, model.fc3]
for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False
print("   - Frozen: CNN/MLP Sequence Head")

# Unfreeze the "New" Chromatin Layers
for param in model.chromatin_projector.parameters():
    param.requires_grad = True
for param in model.fc_final.parameters():
    param.requires_grad = True
print("   - Unfrozen: Chromatin Projector & Final Layer")

# Verify total trainable params
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 Total Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")


# --- F. Train ---
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)
    return {"r2": r2_score(labels, preds), "mse": mean_squared_error(labels, preds)}

class R2Printer(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Epoch {state.epoch:.0f} | Eval R²: {metrics.get('eval_r2', 0):.4f}")

training_args = TrainingArguments(
    output_dir="./results_lora_endogenous",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=1e-3, # Higher LR (1e-3) is safe because we use LoRA!
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="r2",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[R2Printer]
)

print("\n🚀 Starting LoRA Finetuning...")
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