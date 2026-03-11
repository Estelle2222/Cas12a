import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- 1. Dataset with Auto-Binarization ---
class ClassificationDataset(Dataset):
    def __init__(self, sequences, labels):
        self.SEQ = []
        self.labels = []
        
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
        self.labels = np.array(labels, dtype=np.float32).reshape(-1, 1)

    def __len__(self): return len(self.SEQ)
    def __getitem__(self, idx): return self.SEQ[idx], self.labels[idx]

# --- 2. Classification Architecture ---
class SeqClassifier(nn.Module):
    def __init__(self):
        super(SeqClassifier, self).__init__()
        self.conv1 = nn.Conv1d(4, 80, 5)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(1200, 80)
        self.dense2 = nn.Linear(80, 40)
        self.dense3 = nn.Linear(40, 40)
        
        # Classification Head (Logits)
        self.output_layer = nn.Linear(40, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.relu(self.dense3(x))
        x = self.dropout(x)
        return self.output_layer(x)

# --- 3. Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pre-training Classifier on {device}...")

    # Load Data
    df = pd.read_csv("data/cpf1energy.csv")
    df = df.dropna(subset=['sequence', 'efficiency'])
    
    # === CRITICAL: Create Binary Labels ===
    # The paper used top 20% as the threshold for 'Good' (1) vs 'Bad' (0)
    threshold = df['efficiency'].quantile(0.80)
    print(f"Binarizing data: Efficiency > {threshold:.2f} is Class 1 (Good)")
    
    df['binary_label'] = (df['efficiency'] > threshold).astype(float)
    
    sequences = df['sequence'].tolist()
    labels = df['binary_label'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(ClassificationDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(ClassificationDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    model = SeqClassifier().to(device)
    # BCEWithLogitsLoss is the standard for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate (Using AUC which is better than Accuracy for imbalanced data)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seqs, lbls in test_loader:
            seqs = seqs.to(device)
            outputs = torch.sigmoid(model(seqs)) # Convert logits to prob
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(lbls.numpy())
            
    print(f"Test AUC Score: {roc_auc_score(all_targets, all_preds):.4f}")
    
    torch.save(model.state_dict(), "seq_classifier_weights.pth")
    print("Saved classification weights.")

if __name__ == '__main__':
    main()