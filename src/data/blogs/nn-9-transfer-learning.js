export const nn_9_transfer_learning = {
  title: "Transfer Learning: Pre-trained Model দিয়ে দ্রুত Image Classification",
  description: "ImageNet pre-trained VGG16, MobileNetV2 দিয়ে feature extraction ও fine-tuning — কম data-তে high accuracy, data augmentation এবং PyTorch (torchvision.models) দিয়ে সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-transfer-learning",
  content: `
    <h3>১. Transfer Learning কী ও কেন</h3>
    <p>ImageNet-এ train করা model (1.2M images, 1000 classes) ইতিমধ্যে edges, textures, shapes চেনে। আমরা সেই <strong>শেখা features পুনরায় ব্যবহার</strong> করি নতুন task-এ — শূন্য থেকে শেখার দরকার নেই।</p>
    <pre><code># Transfer Learning-এর দুটো approach:
#
# 1. Feature Extraction (Frozen Base):
#    - Pre-trained model-এর weights freeze করো
#    - শুধু নতুন top layers train করো
#    - কম data থাকলে এটাই ভালো
#
# 2. Fine-Tuning (Partial Unfreeze):
#    - Base model-এর শেষ কিছু layer unfreeze করো
#    - পুরো model-কে নতুন data দিয়ে low learning rate-এ re-train করো
#    - বেশি data থাকলে ও higher accuracy দরকার হলে ব্যবহার করো</code></pre>
    <table>
      <thead><tr><th>Model</th><th>ImageNet Accuracy</th><th>Parameters</th><th>Speed</th><th>ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>VGG16</td><td>92.7%</td><td>138M</td><td>ধীর</td><td>Research, feature extraction</td></tr>
        <tr><td>ResNet50</td><td>93.0%</td><td>25.6M</td><td>মাঝারি</td><td>General purpose</td></tr>
        <tr><td>MobileNetV2</td><td>91.0%</td><td>3.4M</td><td>দ্রুত</td><td>Mobile, edge device</td></tr>
        <tr><td>EfficientNetB0</td><td>93.3%</td><td>5.3M</td><td>দ্রুত</td><td>State-of-the-art efficiency</td></tr>
      </tbody>
    </table>

    <h3>২. Feature Extraction — Frozen Base Model</h3>
    <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np

# নিজের dataset ব্যবহার করলে: transforms.Compose([...]) + datasets.ImageFolder + DataLoader
# এখানে demonstrate করার জন্য CIFAR-10 ব্যবহার করছি (10-class, 32x32 images)
IMG_SIZE   = (32, 32)
BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# MobileNetV2: features অংশ শুধু → classifier head বাদ
base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
base_model = base_model.features        # ImageNet classifier head বাদ
for p in base_model.parameters():
    p.requires_grad = False             # weights freeze

# নতুন classifier head যোগ — explicit nn.Module class
class TransferModel(nn.Module):
    def __init__(self, base, dropout_rate=0.3):
        super().__init__()
        self.base = base
        self.pool = nn.AdaptiveAvgPool2d(1)   # GlobalAveragePooling2D-এর মতো, feature map → vector
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 256)       # MobileNetV2 features output 1280 channel
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)         # 10 CIFAR classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        x = self.flatten(self.pool(x))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = TransferModel(base_model)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}")</code></pre>

    <h3>৩. Fine-Tuning — Base Model Unfreeze</h3>
    <pre><code>def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total

val_accs1 = []
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    val_accs1.append(evaluate(model, test_loader))

print(f"Feature extraction val accuracy: {max(val_accs1):.4f}")

# Fine-tuning: base model-এর শেষ 30 sub-module unfreeze করো
for p in base_model.parameters():
    p.requires_grad = True              # প্রথমে সব unfreeze

base_layers = list(base_model.children())
fine_tune_from = len(base_layers) - 30
for layer in base_layers[:fine_tune_from]:
    for p in layer.parameters():
        p.requires_grad = False         # বাকি সব আবার freeze

trainable = sum(1 for p in base_model.parameters() if p.requires_grad)
total_p   = sum(1 for p in base_model.parameters())
print(f"Trainable param tensors: {trainable} / {total_p}")

# Fine-tuning-এ learning rate অনেক কম রাখতে হবে
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)  # default-এর ১০ ভাগের ১ ভাগ

best_val_acc, patience, patience_counter, best_state = 0.0, 3, 0, None
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    val_acc = evaluate(model, test_loader)
    if val_acc > best_val_acc:
        best_val_acc, patience_counter = val_acc, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:    # EarlyStopping(patience=3)
            break

model.load_state_dict(best_state)           # restore_best_weights=True
print(f"Fine-tuning val accuracy: {best_val_acc:.4f}")</code></pre>

    <h3>৪. Data Augmentation</h3>
    <pre><code>from torchvision import transforms

# Training data-তে artificial variation তৈরি করো → overfitting কমে
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),                       # RandomFlip("horizontal")
    transforms.RandomRotation(10),                            # RandomRotation(0.1)
    transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),        # RandomZoom(0.1)
    transforms.ColorJitter(contrast=0.1),                      # RandomContrast(0.1)
    transforms.ToTensor(),
])

# Augmentation Dataset-এর transform-এ বসে — প্রতি epoch-এ on-the-fly apply হয়
aug_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
aug_train_loader = DataLoader(aug_train_set, batch_size=64, shuffle=True)

aug_base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
for p in aug_base.parameters():
    p.requires_grad = False    # base freeze

augmented_model = TransferModel(aug_base, dropout_rate=0.4)</code></pre>

    <h3>৫. Scratch vs Transfer Learning তুলনা</h3>
    <pre><code>class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 6, 128)   # 32->30->15->13->6, padding ছাড়া conv/pool
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

scratch_model = ScratchCNN()
scratch_optimizer = torch.optim.Adam(scratch_model.parameters(), lr=0.001)

scratch_val_accs = []
for epoch in range(10):
    scratch_model.train()
    for xb, yb in train_loader:
        scratch_optimizer.zero_grad()
        loss = criterion(scratch_model(xb), yb)
        loss.backward()
        scratch_optimizer.step()
    scratch_val_accs.append(evaluate(scratch_model, test_loader))

results = {
    'Scratch CNN (10 epochs)':          max(scratch_val_accs),
    'Transfer — Feature Extraction':    max(val_accs1),
    'Transfer — Fine-tuned':            best_val_acc,
}
for name, acc in results.items():
    print(f"{name:40s}: {acc:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Feature Extraction</td><td>Base model freeze (requires_grad = False), শুধু top layers train — কম data-তে দ্রুত ফলাফল</td></tr>
        <tr><td>Fine-Tuning</td><td>Base model partially unfreeze, খুব কম learning rate — accuracy আরও বাড়ে</td></tr>
        <tr><td>base_model.features</td><td>ImageNet-এর 1000-class head বাদ দিয়ে নিজের head যোগ করো</td></tr>
        <tr><td>nn.AdaptiveAvgPool2d(1)</td><td>Feature map → 1D vector — Flatten-এর চেয়ে overfitting কম (GlobalAveragePooling2D-এর মতো)</td></tr>
        <tr><td>Data Augmentation</td><td>কম data-তে অপরিহার্য — rotation, flip, zoom দিয়ে variety বাড়ায়</td></tr>
        <tr><td>MobileNetV2 (torchvision.models)</td><td>Production ও mobile-এ সেরা choice — ছোট, দ্রুত, accurate</td></tr>
      </tbody>
    </table>
  `,
};
