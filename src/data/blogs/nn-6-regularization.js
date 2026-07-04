export const nn_6_regularization = {
  title: "Regularization: Dropout, Batch Norm ও Overfitting রোধ",
  description: "নিউরাল নেটওয়ার্কে overfitting সনাক্ত ও রোধ করতে Dropout, Batch Normalization, L1/L2 regularization, EarlyStopping এবং Learning Rate Scheduling বিস্তারিত।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 11,
  slug: "nn-regularization-dropout",
  content: `
    <h3>১. Overfitting কী ও কীভাবে সনাক্ত করবেন</h3>
    <p>Overfitting হলো যখন model training data খুব ভালো শেখে কিন্তু new data-তে কাজ করে না। Training accuracy বেশি, validation accuracy কম — এটাই overfitting-এর চিহ্ন।</p>
    <pre><code>import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Synthetic dataset: overfitting demonstrate করার জন্য
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Baseline model (overfit করবে)
class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)   # raw logit — BCEWithLogitsLoss নিজেই sigmoid করে
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

def build_overfit_model():
    return OverfitModel()

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                 lr=0.001, weight_decay=0.0):
    """Manual training loop — model.fit()-এর PyTorch equivalent"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'accuracy': [], 'val_accuracy': []}
    n = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = ((torch.sigmoid(model(X_train)) >= 0.5).float() == y_train).float().mean().item()
            val_acc   = ((torch.sigmoid(model(X_val))   >= 0.5).float() == y_val).float().mean().item()
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

    return history

baseline = build_overfit_model()
history_overfit = train_model(baseline, X_train, y_train, X_val, y_val, epochs=100)

train_acc = history_overfit['accuracy'][-1]
val_acc   = history_overfit['val_accuracy'][-1]
print(f"Baseline — Train: {train_acc:.3f}, Val: {val_acc:.3f}")
# Train: 0.99, Val: 0.72 → OVERFITTING!
</code></pre>

    <h3>২. Dropout: Random Neuron বন্ধ করা</h3>
    <p>Dropout training-এর সময় randomly কিছু neurons-এর output শূন্য করে দেয়। এটি neurons-কে forced করে independently শিখতে, একে অপরের উপর নির্ভর না করতে।</p>
    <p>Rate=0.3 মানে প্রতিটি batch-এ 30% neurons বন্ধ থাকবে। Inference-এর সময় সব neurons active থাকে কিন্তু output (1-rate) দিয়ে scale হয়।</p>
    <pre><code>class DropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.fc5(x)

def build_dropout_model(dropout_rate=0.3):
    return DropoutModel(dropout_rate)

dropout_model = build_dropout_model(dropout_rate=0.3)
history_dropout = train_model(dropout_model, X_train, y_train, X_val, y_val, epochs=100)

train_acc = history_dropout['accuracy'][-1]
val_acc   = history_dropout['val_accuracy'][-1]
print(f"Dropout(0.3) — Train: {train_acc:.3f}, Val: {val_acc:.3f}")
# Train: 0.88, Val: 0.85 → Gap অনেক কম!

# Dropout rate guidelines:
# Input layer: 0.1 - 0.2
# Hidden layers: 0.3 - 0.5
# Too high (>0.7): underfitting হবে
</code></pre>

    <h3>৩. Batch Normalization</h3>
    <p>Batch Normalization প্রতিটি layer-এর input normalize করে (mean≈0, std≈1)। এটি training accelerate করে, higher learning rate সম্ভব করে এবং slight regularization effect আছে।</p>
    <pre><code>class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))    # normalize layer output
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        return self.fc5(x)
# Note: BN -> Activation order (BN আগে, Activation পরে) কিছুটা ভালো কাজ করে

def build_batchnorm_model():
    return BatchNormModel()

bn_model = build_batchnorm_model()
history_bn = train_model(bn_model, X_train, y_train, X_val, y_val, epochs=50, lr=0.01)
print("BN model converges faster with higher lr!")
</code></pre>

    <h3>৪. L1 ও L2 Regularization</h3>
    <p>L1/L2 regularization weight penalty loss function-এ যোগ করে বড় weights discourage করে।</p>
    <p>L2 (Ridge): <strong>Total Loss = CE + λ·Σw²</strong> — weights ছোট রাখে</p>
    <p>L1 (Lasso): <strong>Total Loss = CE + λ·Σ|w|</strong> — sparse weights (অনেক zero)</p>
    <pre><code>class L2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def build_l2_model():
    return L2Model()

# L2: optimizer-এর weight_decay দিয়ে হয় — per-layer regularizer লাগে না
# L1: PyTorch-এ built-in flag নেই — loss-এ manually যোগ করতে হয়

l2_model = build_l2_model()
history_l2 = train_model(l2_model, X_train, y_train, X_val, y_val, epochs=100, weight_decay=0.001)
train_acc = history_l2['accuracy'][-1]
val_acc   = history_l2['val_accuracy'][-1]
print(f"L2 Reg — Train: {train_acc:.3f}, Val: {val_acc:.3f}")

def l1_penalty(model, l1_lambda=0.001):
    return l1_lambda * sum(p.abs().sum() for p in model.parameters())
# loss = criterion(model(xb), yb) + l1_penalty(model)   # training loop-এর ভিতরে যোগ করো
</code></pre>

    <h3>৫. EarlyStopping ও Learning Rate Scheduling</h3>
    <pre><code># Combined: সব best practices একসাথে
class BestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

best_model = BestModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001, weight_decay=0.001)

# ReduceLROnPlateau: plateau-তে lr কমায়
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

best_val_loss, patience, patience_counter = float('inf'), 10, 0
best_state = None

for epoch in range(200):
    best_model.train()
    perm = torch.randperm(X_train.shape[0])
    for i in range(0, X_train.shape[0], 32):
        idx = perm[i:i+32]
        optimizer.zero_grad()
        loss = criterion(best_model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

    best_model.eval()
    with torch.no_grad():
        val_loss = criterion(best_model(X_val), y_val).item()
    scheduler.step(val_loss)

    if val_loss &lt; best_val_loss - 0.001:      # min_delta=0.001-এর মতো
        best_val_loss = val_loss
        best_state = best_model.state_dict()   # Best model save (ModelCheckpoint-এর মতো)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:       # EarlyStopping(patience=10)-এর মতো
            print(f"Early stopping at epoch {epoch+1}")
            break

best_model.load_state_dict(best_state)         # restore_best_weights=True
</code></pre>
    <table>
      <thead><tr><th>Technique</th><th>প্রভাব</th><th>কখন ব্যবহার</th><th>Parameter</th></tr></thead>
      <tbody>
        <tr><td>Dropout</td><td>Random neurons মুছে co-adaptation রোধ</td><td>Fully connected layers</td><td>rate: 0.2-0.5</td></tr>
        <tr><td>Batch Norm</td><td>Layer inputs normalize, training speed up</td><td>Deep networks, Conv layers</td><td>momentum, epsilon</td></tr>
        <tr><td>L2 Reg</td><td>Weight magnitude penalty</td><td>Weight regularization</td><td>lambda: 0.001-0.01</td></tr>
        <tr><td>L1 Reg</td><td>Sparse weights</td><td>Feature selection</td><td>lambda: 0.001</td></tr>
        <tr><td>EarlyStopping</td><td>Overfitting শুরু হলে থামায়</td><td>সবসময়</td><td>patience: 5-20</td></tr>
        <tr><td>ReduceLROnPlateau</td><td>Plateau-তে lr কমায়</td><td>Learning stagnation</td><td>factor: 0.5, patience: 5</td></tr>
      </tbody>
    </table>

    <h3>৬. Comparison Plot</h3>
    <pre><code>fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Val accuracy comparison
axes[0].plot(history_overfit['val_accuracy'], label='Baseline (Overfit)', color='red')
axes[0].plot(history_dropout['val_accuracy'], label='Dropout(0.3)', color='blue')
axes[0].plot(history_l2['val_accuracy'], label='L2 Reg', color='green')
axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Train-Val gap (overfitting indicator)
gap_baseline = np.array(history_overfit['accuracy']) - np.array(history_overfit['val_accuracy'])
gap_dropout  = np.array(history_dropout['accuracy']) - np.array(history_dropout['val_accuracy'])

axes[1].plot(gap_baseline, label='Baseline Gap', color='red')
axes[1].plot(gap_dropout, label='Dropout Gap', color='blue')
axes[1].set_title("Train-Val Accuracy Gap (Overfitting Indicator)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Gap")
axes[1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("regularization_comparison.png")
plt.show()
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম কীভাবে overfitting রোধ করতে হয়:</p>
    <ul>
      <li>Dropout (nn.Dropout): random neurons বন্ধ করে co-adaptation রোধ</li>
      <li>Batch Normalization (nn.BatchNorm1d): normalize + accelerate training</li>
      <li>L2: optimizer-এর weight_decay দিয়ে; L1: loss-এ manually যোগ করে — দুটোই weight-কে ছোট রাখে</li>
      <li>Manual early stopping: best val_loss track করে state_dict save, শেষে restore করা</li>
      <li>torch.optim.lr_scheduler.ReduceLROnPlateau: plateau-তে learning rate কমায়</li>
    </ul>
    <p>পরবর্তী ও শেষ ব্লগে Fashion MNIST দিয়ে একটি সম্পূর্ণ end-to-end project করব।</p>
  `,
};
