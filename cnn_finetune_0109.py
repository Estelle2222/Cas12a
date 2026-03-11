import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
from sklearn.preprocessing import StandardScaler
# --- 1. Dataset ---
class DeepDataset(Dataset):
    def __init__(self, sequences, access_features, labels):
        self.SEQ = []
        self.CA = access_features
        self.labels = labels
        for seq_str in sequences:
            mat = np.zeros((4, 34), dtype=np.float32)
            for i, base in enumerate(seq_str.upper()):
                if i >= 34: break
                if base == 'A': mat[0, i] = 1
                elif base == 'C': mat[1, i] = 1
                elif base == 'G': mat[2, i] = 1
                elif base == 'T': mat[3, i] = 1
            self.SEQ.append(mat)
        self.SEQ = np.array(self.SEQ)
        self.CA = np.array(self.CA, dtype=np.float32).reshape(-1, 1)
        self.labels = np.array(self.labels, dtype=np.float32).reshape(-1, 1)

    def __len__(self): return len(self.SEQ)
    def __getitem__(self, idx): return self.SEQ[idx], self.CA[idx], self.labels[idx]

# --- 2. Architecture (Must match Classifier Body) ---
class DeepCpf1(nn.Module):
    def __init__(self):
        super(DeepCpf1, self).__init__()
        # Body (Matches Pre-trained Classifier)
        self.seq_conv1 = nn.Conv1d(4, 80, 5)
        self.seq_dense1 = nn.Linear(1200, 80)
        self.seq_dense2 = nn.Linear(80, 40)
        self.seq_dense3 = nn.Linear(40, 40)
        
        # New Heads
        self.ca_dense = nn.Linear(1, 40)
        self.output_layer = nn.Linear(40, 1)
        
        # Shared
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.0) 

    def forward(self, seq, ca):
        s = self.relu(self.seq_conv1(seq))
        s = self.pool(s)
        s = self.flatten(s)
        s = self.dropout(s)
        s = self.relu(self.seq_dense1(s))
        s = self.dropout(s)
        s = self.relu(self.seq_dense2(s))
        s = self.dropout(s)
        s_out = self.relu(self.seq_dense3(s))
        
        c_out = self.relu(self.ca_dense(ca))
        merged = s_out * c_out
        m = self.dropout(merged)
        return self.output_layer(m)

# --- 3. Head Swap Logic ---
def load_classifier_body(model, path):
    print("Loading classification body, discarding head...")
    classifier_dict = torch.load(path)
    model_dict = model.state_dict()
    
    # Map layers. Note: We exclude 'output_layer' to let it re-initialize
    weight_map = {
        'conv1.weight': 'seq_conv1.weight', 'conv1.bias': 'seq_conv1.bias',
        'dense1.weight': 'seq_dense1.weight', 'dense1.bias': 'seq_dense1.bias',
        'dense2.weight': 'seq_dense2.weight', 'dense2.bias': 'seq_dense2.bias',
        'dense3.weight': 'seq_dense3.weight', 'dense3.bias': 'seq_dense3.bias'
    }
    
    for old_k, new_k in weight_map.items():
        if old_k in classifier_dict:
            model_dict[new_k] = classifier_dict[old_k]
            
    model.load_state_dict(model_dict)
    return model

# --- 4. Evaluation Helper ---
def evaluate_spearman(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seqs, cas, labels in loader:
            seqs, cas = seqs.to(device), cas.to(device)
            outputs = model(seqs, cas)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())
    
    corr, _ = spearmanr(all_targets, all_preds)
    return corr

# --- 5. Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning with Head Swap on {device}...")

    # Load Data
    df = pd.read_csv("data/Z_gRNA.csv")
    df = df.dropna(subset=['sequence', 'efficiency', 'accessibility'])
    allowed_chars = set('ATGC')
    df = df[df['sequence'].apply(lambda x: set(x.upper()).issubset(allowed_chars))]
    
    sequences = df['sequence'].tolist()
    # Check scaling (0-100)
    access_features = df['accessibility'].values.astype(np.float32).reshape(-1, 1)
    try:
        scaler = joblib.load('my_scaler.pkl')
        
        # 1. Get raw values
        raw_access = df['accessibility'].values.reshape(-1, 1)
        
        # 2. Transform using the loaded scaler
        scaled_values = scaler.transform(raw_access).astype(np.float32)
        
        # 3. SAVE TO NEW COLUMN (Explicitly)
        # This locks the scaled values into the DataFrame for safety
        df['accessibility_scaled'] = scaled_values
        
        print("Success: Added 'accessibility_scaled' column to DataFrame.")
        
    except FileNotFoundError:
        print("CRITICAL ERROR: 'my_scaler.pkl' not found.")
        return
    
    access_features = df['accessibility_scaled'].values    
    # if access_features.max() <= 1.0: access_features *= 100.0
    labels = df['efficiency'].values.astype(np.float32)

    # 5-Fold Cross-Validation recommended, but for quick check we use single split
    X_train, X_test, y_train, y_test, ca_train, ca_test = train_test_split(
        sequences, labels, access_features, test_size=0.2, random_state=45
    )
    
    train_loader = DataLoader(DeepDataset(X_train, ca_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(DeepDataset(X_test, ca_test, y_test), batch_size=16, shuffle=False)

    model = DeepCpf1().to(device)
    
    # === LOAD WEIGHTS FROM CLASSIFIER ===
    try:
        model = load_classifier_body(model, "seq_classifier_weights.pth")
    except FileNotFoundError:
        print("Error: 'seq_classifier_weights.pth' not found.")
        return

    # === INITIALIZE NEW HEADS ===
    with torch.no_grad():
        # Initialize Chromatin branch (Critical Fix)
        nn.init.uniform_(model.ca_dense.weight, 0.0, 0.1)
        nn.init.constant_(model.ca_dense.bias, 1.0)
        # Initialize Regression Head
        nn.init.xavier_uniform_(model.output_layer.weight)
        nn.init.zeros_(model.output_layer.bias)

    # === FREEZE BODY ===
    print("Freezing body layers...")
    for name, param in model.named_parameters():
        if "seq_conv1" in name or "seq_dense1" in name or "seq_dense2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # Train seq_dense3 + chromatin + output

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting Fine-tuning...")
    best_spearman = -1.0
    
    for epoch in range(200):
        model.train()
        total_loss = 0
        for seqs, cas, lbls in train_loader:
            seqs, cas, lbls = seqs.to(device), cas.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(seqs, cas)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            val_spearman = evaluate_spearman(model, test_loader, device)
            train_spearman = evaluate_spearman(model, train_loader, device)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Test Spearman: {val_spearman:.4f} | Train Spearman:{train_spearman:.4f}")
            if val_spearman > best_spearman:
                best_spearman = val_spearman

    print(f"\nBest Spearman Correlation: {best_spearman:.4f}")
    torch.save(model.state_dict(), "Z_gRNA_deepcpf1_model.pth")
    print("Saved to Z_deepcpf1_model.pth")
    print(sum(p.numel() for p in model.parameters()))

if __name__ == '__main__':
    main()