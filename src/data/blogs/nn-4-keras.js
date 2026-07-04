export const nn_4_keras = {
  title: "PyTorch দিয়ে নিউরাল নেটওয়ার্ক",
  description: "PyTorch দিয়ে neural network তৈরি (nn.Module, manual training loop), MNIST handwritten digit classification এবং model save/load সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-keras-tensorflow",
  content: `
    <h3>১. PyTorch কেন? NumPy থেকে PyTorch-এ যাওয়া</h3>
    <p>আগের ব্লগে আমরা NumPy দিয়ে neural network implement করেছিলাম — ৩০০+ লাইন কোড। PyTorch দিয়ে একই কাজ অনেক কম লাইনে করা যায়, অথচ training loop-এর উপর সম্পূর্ণ নিয়ন্ত্রণ থাকে।</p>
    <p>PyTorch-এর সুবিধা:</p>
    <ul>
      <li>GPU support: <code>.to(device)</code> দিয়ে automatic</li>
      <li>Automatic differentiation (autograd): backprop নিজেই করে</li>
      <li>Built-in optimizers ও losses</li>
      <li>Training loop plain Python — পড়া ও debug করা সহজ</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# PyTorch version check
print("PyTorch version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
</code></pre>

    <h3>২. MNIST Dataset: Handwritten Digits</h3>
    <p>MNIST হলো ML-এর "Hello World" — ৭০,০০০ handwritten digit image (0-9)। প্রতিটি image 28×28 grayscale pixel।</p>
    <pre><code># MNIST load — torchvision নিজেই download করে, ToTensor 0-255 কে 0-1-এ normalize করে
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Training:", len(train_dataset))  # 60000
print("Test:", len(test_dataset))       # 10000

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)      # (128, 1, 28, 28)
print("Label sample:", labels[:10].tolist())

# Sample images visualize
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.suptitle("MNIST Sample Images")
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()
</code></pre>

    <h3>৩. nn.Module দিয়ে Model তৈরি</h3>
    <p>PyTorch-এ model একটি Python class — layer গুলো <code>__init__</code>-এ declare করা হয় আর forward computation <code>forward()</code>-এ explicitly লেখা হয়।</p>
    <pre><code># Model 1: Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)    # 10 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)              # raw logits — softmax loss-এর ভিতরেই আছে

model = SimpleMLP().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print("Total params:", total_params)    # 109,386
# fc1: 784*128 + 128 = 100480
# fc2: 128*64 + 64   = 8256
# fc3: 64*10 + 10    = 650

# criterion ও optimizer আলাদাভাবে define — compile()-এর বদলে
criterion = nn.CrossEntropyLoss()       # softmax + NLL একসাথে, integer label সরাসরি নেয়
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Note: one-hot label লাগবে না — CrossEntropyLoss সরাসরি integer label নেয়
</code></pre>

    <h3>৪. Training: Manual Training Loop</h3>
    <p>PyTorch-এ <code>.fit()</code> নেই — নিজেই loop লিখতে হয়: forward pass, loss হিসাব, backward pass, weight update।</p>
    <pre><code>def evaluate(model, loader):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total

best_val_loss = float('inf')
patience, patience_counter = 5, 0
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_loss, train_acc = running_loss / total, correct / total
    val_loss, val_acc = evaluate(model, test_loader)
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss &lt; best_val_loss:               # ModelCheckpoint(save_best_only=True)-এর মতো
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_mnist_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:       # EarlyStopping(patience=5)-এর মতো
            print(f"Early stopping at epoch {epoch+1}")
            break

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")  # ~0.98

model.eval()
sample_x, sample_y = next(iter(test_loader))
with torch.no_grad():
    logits = model(sample_x[:5].to(device))
    y_pred = logits.argmax(1).cpu()
print("Predicted:", y_pred.tolist())
print("Actual:   ", sample_y[:5].tolist())
</code></pre>

    <h3>৫. Training Curves ও Analysis</h3>
    <pre><code>def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history['train_acc'], label='Train Accuracy', color='blue')
    axes[0].plot(history['val_acc'], label='Val Accuracy', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[1].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

plot_training_history(history)

# Confusion matrix (top errors)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu()
        all_preds.append(preds); all_labels.append(yb)
y_pred_all = torch.cat(all_preds).numpy()
y_test_all = torch.cat(all_labels).numpy()

cm = confusion_matrix(y_test_all, y_pred_all)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix — MNIST Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_test_all, y_pred_all))
</code></pre>

    <h3>৬. Model Save ও Load</h3>
    <pre><code># Save করার দুটি সাধারণ উপায়

# 1. Weights only (recommended)
torch.save(model.state_dict(), 'mnist_model.pt')

# 2. পুরো model object (কম portable)
torch.save(model, 'mnist_model_full.pt')

# Load করা
loaded_model = SimpleMLP().to(device)
loaded_model.load_state_dict(torch.load('mnist_model.pt'))
loaded_model.eval()

test_loss, test_acc = evaluate(loaded_model, test_loader)
print(f"Loaded model accuracy: {test_acc:.4f}")

# Model architecture দেখা
print(loaded_model)

# Inference: single image
single_image, _ = test_dataset[0]
single_image = single_image.unsqueeze(0).to(device)   # batch dimension যোগ: (1, 1, 28, 28)
with torch.no_grad():
    logits = loaded_model(single_image)
    probs = torch.softmax(logits, dim=1)
    predicted_class = probs.argmax(1).item()
    confidence = probs[0][predicted_class].item()
print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
</code></pre>
    <table>
      <thead><tr><th>Concept</th><th>ব্যবহার</th><th>Return</th></tr></thead>
      <tbody>
        <tr><td>training loop</td><td>Training</td><td>history dict (loss, accuracy per epoch) — নিজে বানাতে হয়</td></tr>
        <tr><td>evaluate() (custom)</td><td>Test set performance</td><td>(loss, accuracy)</td></tr>
        <tr><td>model(x), torch.no_grad()-এর ভিতরে</td><td>Probability/output</td><td>logits tensor</td></tr>
        <tr><td>print(model)</td><td>Architecture overview</td><td>Print to stdout</td></tr>
        <tr><td>torch.save(model.state_dict(), path)</td><td>Persistence</td><td>Saves to disk</td></tr>
        <tr><td>model.load_state_dict(torch.load(path))</td><td>Load saved model</td><td>Model-কে in-place populate করে</td></tr>
      </tbody>
    </table>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা PyTorch দিয়ে শিখলাম:</p>
    <ul>
      <li>nn.Module: __init__-এ layer, forward()-এ computation</li>
      <li>criterion ও optimizer: compile()-এর loss ও optimizer argument-এর বদলে</li>
      <li>Manual training loop: zero_grad() → forward → loss → backward() → step()</li>
      <li>MNIST: ToTensor → DataLoader → Linear layers → CrossEntropyLoss → 98%+ accuracy</li>
      <li>Manual early stopping: best val_loss track করে checkpoint save, patience শেষ হলে থামা</li>
      <li>Model save/load: state_dict দিয়ে production deployment-এর জন্য</li>
    </ul>
    <p>পরবর্তী ব্লগে CNN (Convolutional Neural Network) দিয়ে image classification করব — MLP-এর চেয়ে অনেক বেশি accurate।</p>
  `,
};
