export const nn_7_project = {
  title: "প্রজেক্ট: Fashion MNIST পোশাক শ্রেণীবিন্যাস",
  description: "Fashion MNIST dataset দিয়ে সম্পূর্ণ end-to-end deep learning প্রজেক্ট — data exploration, MLP, CNN build, তুলনা, confusion matrix এবং best practices সহ।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "nn-project-fashion-mnist",
  content: `
    <h3>১. Fashion MNIST: Dataset পরিচয়</h3>
    <p>Fashion MNIST হলো MNIST-এর একটি কঠিন version। ৭০,০০০ grayscale image (28×28) দশটি পোশাক category-তে। MNIST digits-এর মতো simple নয় — real-world image classification challenge।</p>
    <p>Classes: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Sandal (5), Shirt (6), Sneaker (7), Bag (8), Ankle boot (9)।</p>
    <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

torch.manual_seed(42)
np.random.seed(42)

# Class names
CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Data load
transform = transforms.Compose([transforms.ToTensor()])
train_full = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
test_set   = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("Full training set:", len(train_full))
print("Test set:", len(test_set))
</code></pre>

    <h3>২. Data Exploration ও Visualization</h3>
    <pre><code># Sample images visualization
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_full[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(CLASS_NAMES[label], fontsize=9)
    ax.axis('off')
plt.suptitle("Fashion MNIST Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig("fashion_samples.png")
plt.show()

# Class distribution
train_labels = train_full.targets.numpy()
fig, ax = plt.subplots(figsize=(10, 4))
class_counts = np.bincount(train_labels)
bars = ax.bar(CLASS_NAMES, class_counts, color=plt.cm.tab10.colors)
ax.set_title("Class Distribution (Training Set)")
ax.set_ylabel("Count")
plt.xticks(rotation=30, ha='right')
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(count), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

print("Class distribution (balanced):")
for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {i}: {name:15s} — {count} samples")
</code></pre>

    <h3>৩. Preprocessing</h3>
    <pre><code># Preprocessing
# 1. Normalize 0-255 -> 0-1
X_train_full = train_full.data.float() / 255.0    # (60000, 28, 28)
y_train_full = train_full.targets
X_test       = test_set.data.float() / 255.0
y_test       = test_set.targets

# 2. Train/Validation split
val_size = 5000
X_val   = X_train_full[:val_size]
y_val   = y_train_full[:val_size]
X_train = X_train_full[val_size:]
y_train = y_train_full[val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 3. Model অনুযায়ী reshape
X_train_flat = X_train.reshape(-1, 784)
X_val_flat   = X_val.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

X_train_cnn = X_train.unsqueeze(1)   # (55000, 1, 28, 28)
X_val_cnn   = X_val.unsqueeze(1)
X_test_cnn  = X_test.unsqueeze(1)

# Common early-stopping helper — EarlyStopping + ModelCheckpoint-এর বদলে
class EarlyStopper:
    def __init__(self, model_name, patience=8):
        self.model_name = model_name
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss &lt; self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), f'{self.model_name}_best.pt')
            return False       # training চালিয়ে যাও
        self.counter += 1
        return self.counter >= self.patience   # True -> থামো
</code></pre>

    <h3>৪. তিনটি Model Build ও Train</h3>
    <pre><code># Model 1: Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)

def build_simple_mlp():
    return SimpleMLP()

# Model 2: Deep MLP with BN
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

def build_deep_mlp():
    return DeepMLP()

# Model 3: CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = self.flatten(x)
        x = self.dropout3(self.relu(self.bn3(self.fc1(x))))
        return self.fc2(x)

def build_cnn():
    return CNN()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# সব model train করার জন্য common training loop
def train_and_evaluate(name, build_fn, X_tr, X_v, X_te, weight_decay=0.0, epochs=50, batch_size=64):
    model = build_fn()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    stopper = EarlyStopper(name, patience=8)

    train_loader = DataLoader(TensorDataset(X_tr, y_train), batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_val).item()
        scheduler.step(val_loss)
        if stopper.step(val_loss, model):
            break

    model.load_state_dict(torch.load(f'{name}_best.pt'))   # restore_best_weights
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_test).float().mean().item()

    return {'model': model, 'test_acc': acc, 'n_params': count_params(model), 'train_time': train_time}

results = {}
models_config = [
    ('simple_mlp', build_simple_mlp, X_train_flat, X_val_flat, X_test_flat, 0.0),
    ('deep_mlp',   build_deep_mlp,   X_train_flat, X_val_flat, X_test_flat, 0.001),
    ('cnn',        build_cnn,        X_train_cnn,  X_val_cnn,  X_test_cnn,  0.0),
]

for name, build_fn, X_tr, X_v, X_te, wd in models_config:
    print(f"\nTraining {name}...")
    r = train_and_evaluate(name, build_fn, X_tr, X_v, X_te, weight_decay=wd)
    results[name] = r
    print(f"  Params: {r['n_params']:,} | Test Acc: {r['test_acc']:.4f} | Time: {r['train_time']:.1f}s")
</code></pre>

    <h3>৫. Model Comparison ও Confusion Matrix</h3>
    <pre><code># Comparison table print
print("\n" + "="*70)
print(f"{'Model':<15} {'Params':>10} {'Test Acc':>10} {'Time (s)':>10}")
print("="*70)
for name, r in results.items():
    print(f"{name:<15} {r['n_params']:>10,} {r['test_acc']:>10.4f} {r['train_time']:>10.1f}")

# Confusion Matrix for best model (CNN)
best_model = results['cnn']['model']
best_model.eval()
with torch.no_grad():
    y_pred = best_model(X_test_cnn).argmax(1).numpy()
y_test_np = y_test.numpy()
cm = confusion_matrix(y_test_np, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("CNN Confusion Matrix — Fashion MNIST Test Set")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("fashion_confusion_matrix.png")
plt.show()
</code></pre>

    <h3>৬. Per-Class Accuracy ও Error Analysis</h3>
    <pre><code># Per-class accuracy
print("\nPer-Class Accuracy (CNN):")
print("-"*45)
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = (y_test_np == i)
    class_acc = np.mean(y_pred[class_mask] == y_test_np[class_mask])
    print(f"  {i}: {class_name:<15} — {class_acc:.3f} ({class_acc*100:.1f}%)")

# Most confused pairs
print("\nMost Confused Pairs:")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    true_class = CLASS_NAMES[idx[0]]
    pred_class = CLASS_NAMES[idx[1]]
    count = cm_no_diag[idx]
    print(f"  True: {true_class:<15} → Predicted: {pred_class:<15} ({count} errors)")
    cm_no_diag[idx] = 0

# Classification report
print("\nFull Classification Report:")
print(classification_report(y_test_np, y_pred, target_names=CLASS_NAMES))
</code></pre>
    <table>
      <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>Test Accuracy</th><th>Training Time</th></tr></thead>
      <tbody>
        <tr><td>Simple MLP</td><td>784→256→128→10 + Dropout</td><td>~234K</td><td>~88.5%</td><td>~30s</td></tr>
        <tr><td>Deep MLP + BN</td><td>784→512→256→128→10 + BN + Dropout + L2</td><td>~530K</td><td>~89.8%</td><td>~60s</td></tr>
        <tr><td>CNN</td><td>Conv32→Conv32→Pool→Conv64→Conv64→Pool→256→10</td><td>~620K</td><td>~92.5%</td><td>~180s</td></tr>
      </tbody>
    </table>

    <h3>৭. Best Practices সারসংক্ষেপ</h3>
    <pre><code"># Final best practices checklist

# 1. Data Preprocessing
print("Preprocessing:")
print("  - Normalize: 0-255 -> 0-1 (or standardize)")
print("  - Always use validation set (separate from test)")
print("  - Check class balance")

# 2. Architecture
print("\nArchitecture:")
print("  - Start simple, add complexity as needed")
print("  - CNN for images, Dense for tabular")
print("  - BatchNorm before activation in deep nets")
print("  - ReLU hidden layers, Softmax output (multi-class)")

# 3. Training
print("\nTraining:")
print("  - Adam optimizer (lr=0.001) as default")
print("  - Mini-batch size: 32-128")
print("  - Use EarlyStopping + restore_best_weights")
print("  - ReduceLROnPlateau for better convergence")

# 4. Regularization
print("\nRegularization (if overfit):")
print("  - Dropout: 0.2-0.5 for Dense, 0.25 for Conv")
print("  - L2: lambda=0.001 on Dense weights")
print("  - Data augmentation for images")

# 5. Evaluation
print("\nEvaluation:")
print("  - Confusion matrix for class-level analysis")
print("  - Per-class accuracy: find weak spots")
print("  - Analyze worst errors for insights")
</code></pre>
    <table>
      <thead><tr><th>Best Practice</th><th>কেন গুরুত্বপূর্ণ</th><th>Default পছন্দ</th></tr></thead>
      <tbody>
        <tr><td>Normalize input</td><td>Stable gradients, faster convergence</td><td>0-1 বা StandardScaler</td></tr>
        <tr><td>He/Xavier init</td><td>Symmetry breaking, gradient flow</td><td>PyTorch default (Kaiming uniform)</td></tr>
        <tr><td>Adam optimizer</td><td>Adaptive lr, momentum combined</td><td>lr=0.001</td></tr>
        <tr><td>Batch Normalization</td><td>Faster training, higher lr possible</td><td>Deep networks-এ সবসময়</td></tr>
        <tr><td>EarlyStopping</td><td>Overfitting রোধ, compute save</td><td>patience=10, restore_best=True</td></tr>
        <tr><td>Validation set</td><td>Unbiased performance estimate</td><td>10-20% of training data</td></tr>
        <tr><td>Confusion matrix</td><td>Class-level insights</td><td>Final evaluation-এ সবসময়</td></tr>
      </tbody>
    </table>
    <p>এই সিরিজে আমরা পার্সেপট্রন থেকে শুরু করে সম্পূর্ণ deep learning pipeline শিখলাম। পরবর্তী সিরিজে Recurrent Neural Networks (RNN/LSTM) এবং Natural Language Processing নিয়ে আলোচনা করব।</p>
  `,
};
