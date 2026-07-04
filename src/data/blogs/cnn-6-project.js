export const cnn_6_project = {
  title: "CIFAR-10 প্রজেক্ট: Custom CNN, ResNet ও Transfer Learning — Grad-CAM সহ",
  description: "CIFAR-10-এ তিনটি model তুলনা — Simple CNN, ResNet skip connections ও MobileNetV2 transfer learning। Data augmentation, ReduceLROnPlateau, EarlyStopping, confusion matrix এবং Grad-CAM visualization দিয়ে model-এর মনোজগৎ বোঝা।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "cnn-project-image-classification",
  content: `
    <h3>১. CIFAR-10: Data Loading ও Preprocessing</h3>
    <p>CIFAR-10 dataset-এ 60,000টি 32×32 RGB image এবং 10টি class: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck।</p>
    <p>Training: 50,000 | Test: 10,000 | প্রতি class: 6,000 images।</p>
    <pre><code>import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Data load
transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {len(train_set)}")    # 50000, each (3, 32, 32)
print(f"Test samples:     {len(test_set)}")     # 10000, each (3, 32, 32)

sample_img, _ = train_set[0]
print(f"Pixel value range: {sample_img.min():.3f} - {sample_img.max():.3f}")   # already [0.0, 1.0]

# transforms.ToTensor() already scales [0, 255] → [0.0, 1.0] and permutes to (C, H, W)

# Per-channel normalization (ImageNet mean/std — for pretrained models)
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
x_norm = imagenet_normalize(sample_img)

# Dataset statistics
labels = np.array(train_set.targets)
print(f"\nPer-class distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(labels == i)
    print(f"  {name}: {count} train samples")
</code></pre>

    <h3>২. Data Augmentation Pipeline</h3>
    <p>Data augmentation training data artificially বাড়ায় এবং overfitting কমায়।</p>
    <pre><code>import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# torchvision transforms দিয়ে augmentation pipeline
def build_augmentation_pipeline():
    """
    Training-এ augment, inference-এ শুধু ToTensor (identity augmentation)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),                    # বাম-ডান flip
        transforms.RandomRotation(10),                        # ±10% rotation-এর কাছাকাছি
        transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),   # ±10% zoom
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift
        transforms.ColorJitter(contrast=0.2),                 # Contrast adjust
        transforms.ToTensor(),
    ])

augmentation = build_augmentation_pipeline()

# Cutout augmentation (manual): random rectangle zero করা
def cutout(image, n_holes=1, length=8):
    """Cutout regularization — image: (C, H, W) tensor"""
    h, w = image.shape[1:]
    mask = torch.ones((h, w), dtype=torch.float32)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0

    return image * mask.unsqueeze(0)

# Augmented dataset
train_set_aug  = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=augmentation)
test_set_plain = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_ds = DataLoader(train_set_aug,  batch_size=128, shuffle=True)
test_ds  = DataLoader(test_set_plain, batch_size=128, shuffle=False)
</code></pre>

    <h3>৩. Model 1: Simple CNN (Baseline)</h3>
    <pre><code>import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """3 Conv blocks + Flatten + Linear"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)     # 16x16x32
        self.drop1 = nn.Dropout(0.2)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)     # 8x8x64
        self.drop2 = nn.Dropout(0.3)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn6   = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)     # 4x4x128

        # Classifier
        self.flatten = nn.Flatten()             # 4*4*128 = 2048
        self.fc1 = nn.Linear(2048, 256)
        self.relu7 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.relu7(self.fc1(x))
        x = self.drop3(x)
        return self.fc2(x)

model1 = SimpleCNN()
model1_params = sum(p.numel() for p in model1.parameters())
print(f"Simple CNN params: {model1_params:,}")
</code></pre>

    <h3>৪. Model 2: ResNet-Style Skip Connections</h3>
    <pre><code>import torch
import torch.nn as nn

class ResidualBlockCIFAR(nn.Module):
    """ResNet block adapted for CIFAR-10 (32x32 input)"""
    def __init__(self, in_channels, filters, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

        self.has_shortcut = stride != 1 or in_channels != filters
        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.shortcut_bn   = nn.BatchNorm2d(filters)

        self.relu_out = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x)) if self.has_shortcut else x
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu_out(y + shortcut)

class ResNetCIFAR(nn.Module):
    """ResNet-20 style for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU()

        # Stage 1: 32x32
        self.stage1a = ResidualBlockCIFAR(64, 64)
        self.stage1b = ResidualBlockCIFAR(64, 64)

        # Stage 2: 16x16 (stride=2 downsamples)
        self.stage2a = ResidualBlockCIFAR(64, 128, stride=2)
        self.stage2b = ResidualBlockCIFAR(128, 128)

        # Stage 3: 8x8
        self.stage3a = ResidualBlockCIFAR(128, 256, stride=2)
        self.stage3b = ResidualBlockCIFAR(256, 256)

        # Stage 4: 4x4
        self.stage4a = ResidualBlockCIFAR(256, 512, stride=2)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))

        x = self.stage1a(x)
        x = self.stage1b(x)

        x = self.stage2a(x)
        x = self.stage2b(x)

        x = self.stage3a(x)
        x = self.stage3b(x)

        x = self.stage4a(x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

model2 = ResNetCIFAR()
model2_params = sum(p.numel() for p in model2.parameters())
print(f"ResNet CIFAR params: {model2_params:,}")
</code></pre>

    <h3>৫. Model 3: MobileNetV2 Transfer Learning</h3>
    <pre><code>import torch
import torch.nn as nn
import torchvision

class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2 transfer learning:
    1. Feature extraction (base frozen)
    2. Fine-tuning (last layers unfrozen)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Pre-trained base (ImageNet weights)
        base_model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.base_model = base_model.features
        for p in self.base_model.parameters():
            p.requires_grad = False   # Phase 1: Feature extraction

        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1      = nn.Linear(1280, 128)
        self.relu     = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2      = nn.Linear(128, num_classes)

    def forward(self, x):
        # CIFAR-10 images ছোট (32x32) → upscale করি
        x = nn.functional.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)  # MobileNetV2 ভালো কাজ করে ≥96x96 তে
        x = self.base_model(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

model3 = MobileNetV2Transfer()
total_params     = sum(p.numel() for p in model3.parameters())
trainable_params = sum(p.numel() for p in model3.parameters() if p.requires_grad)
print(f"MobileNetV2 Transfer params (total):     {total_params:,}")
print(f"MobileNetV2 Transfer params (trainable): {trainable_params:,}")

# Phase 1: Feature extraction training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)
patience  = 5   # early stopping patience

# Phase 1 training (5 epochs sufficient with frozen base)
print("Phase 1: Feature extraction...")
# for epoch in range(10):
#     ... standard training loop, tracking best val loss for early stopping ...

# Phase 2: Fine-tuning
print("Phase 2: Fine-tuning last layers...")
for p in model3.base_model.parameters():
    p.requires_grad = True
# শুধু শেষ কয়েকটি child layer fine-tune (বাকি সব আবার freeze)
base_children = list(model3.base_model.children())
for layer in base_children[:-4]:
    for p in layer.parameters():
        p.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model3.parameters()), lr=1e-5)  # Lower LR for fine-tuning
# for epoch in range(10):
#     ... continue training loop for fine-tuning phase ...
</code></pre>

    <h3>৬. Training সব Model ও Comparison</h3>
    <pre><code>import time

def train_model(model, train_ds, test_ds, epochs=50, patience=15, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6, verbose=True)

    best_val_acc, epochs_no_improve, best_state = 0.0, 0, None
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_ds:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_ds:
                out = model(xb)
                val_loss_sum += criterion(out, yb).item()
                correct += (out.argmax(dim=1) == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total
        scheduler.step(val_loss_sum)

        if val_acc > best_val_acc:
            best_val_acc, best_state, epochs_no_improve = val_acc, model.state_dict(), 0
            torch.save(best_state, f'best_{model_name}.pt')   # checkpoint the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:   # early stopping
                break

    model.load_state_dict(best_state)   # restore best weights
    train_time = time.time() - start_time
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'test_accuracy': best_val_acc,
        'train_time_min': train_time / 60,
        'params': total_params
    }

# Train all models
results = {}
for m, name in [(model1, "SimpleCNN"), (model2, "ResNetCIFAR")]:
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    results[name] = train_model(m, train_ds, test_ds, epochs=100, model_name=name)

# Comparison table print
print("\n" + "="*70)
print(f"{'Model':&lt;20} {'Params':>12} {'Time(min)':>10} {'Test Acc':>10}")
print("="*70)
for name, res in results.items():
    print(f"{name:&lt;20} {res['params']:>12,} {res['train_time_min']:>10.1f} {res['test_accuracy']:>10.4f}")
</code></pre>

    <h3>৭. Grad-CAM: Model কোথায় তাকাচ্ছে?</h3>
    <p><strong>Grad-CAM (Gradient-weighted Class Activation Mapping):</strong> একটি image-এর কোন region-এ model মনোযোগ দিয়ে prediction করছে তা visualize করে।</p>
    <p><strong>Algorithm:</strong></p>
    <ol>
      <li>Target class-এর score compute করো</li>
      <li>Last conv layer-এর activation-এর সাপেক্ষে gradient নাও</li>
      <li>Gradient-এর global average pool → importance weights</li>
      <li>Weighted sum of activation maps → heatmap</li>
      <li>ReLU apply → original image-এ overlay</li>
    </ol>
    <pre><code>import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def grad_cam(model, img_tensor, target_layer, pred_index=None):
    """
    Grad-CAM implementation using forward/backward hooks
    img_tensor: (1, C, H, W) preprocessed image
    target_layer: the last convolutional nn.Module layer
    pred_index: class index (None = top prediction)
    """
    activations, gradients = [], []
    fwd_handle = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bwd_handle = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    model.eval()
    predictions = model(img_tensor)
    if pred_index is None:
        pred_index = predictions[0].argmax().item()
    model.zero_grad()
    predictions[0, pred_index].backward()
    fwd_handle.remove(); bwd_handle.remove()

    # Class score-এর সাপেক্ষে last conv layer-এর gradient
    conv_outputs = activations[0][0]        # (C, H, W)
    grads        = gradients[0][0]          # (C, H, W)

    # Global average pooling of gradients → importance weights
    pooled_grads = grads.mean(dim=(1, 2))

    # Weighted combination of feature maps
    heatmap = (conv_outputs * pooled_grads[:, None, None]).sum(dim=0)

    # ReLU: negative contributions সরাও
    heatmap = torch.relu(heatmap) / (heatmap.max() + 1e-8)

    return heatmap.detach().numpy(), pred_index

def overlay_gradcam(img, heatmap, alpha=0.4):
    """Grad-CAM heatmap original image-এ overlay করো — img: (H, W, C) numpy array, [0,1] range"""
    # Heatmap resize to image size
    heatmap_t = torch.from_numpy(heatmap)[None, None]
    heatmap_resized = nn.functional.interpolate(
        heatmap_t, size=img.shape[:2], mode='bilinear', align_corners=False
    )[0, 0].numpy()

    # Colormap apply
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # RGB only

    # Overlay
    overlaid = alpha * heatmap_colored + (1 - alpha) * img
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid, heatmap_resized

# Usage example
def visualize_grad_cam(model, test_set, class_names, num_samples=5):
    # Last conv layer খুঁজে বের করা (model-এর শেষ nn.Conv2d)
    last_conv_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    print(f"Using last conv layer: {last_conv_layer}")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    for i in range(num_samples):
        img_tensor, true_idx = test_set[i]
        true_label = class_names[true_idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Grad-CAM compute
        heatmap, pred_class = grad_cam(model, img_tensor.unsqueeze(0), last_conv_layer)
        pred_label = class_names[pred_class]
        with torch.no_grad():
            confidence = model(img_tensor.unsqueeze(0)).softmax(dim=1)[0, pred_class].item()

        overlaid, _ = overlay_gradcam(img_np, heatmap)

        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original\nTrue: {true_label}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title(f"Overlay\nPred: {pred_label} ({confidence:.2f})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Call visualization
# visualize_grad_cam(model1, test_set, CLASS_NAMES)

# Per-class accuracy ও confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_detailed(model, test_loader, class_names):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(yb.tolist())

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=class_names))

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

# evaluate_detailed(model1, test_ds, CLASS_NAMES)

# Final comparison table
print("\n" + "="*60)
print("CIFAR-10 Model Comparison")
print("="*60)
comparison = [
    ("Simple CNN (3 blocks)",     1_116_970,  45, 80),
    ("ResNet with skip conn.",    2_830_026,  55, 87),
    ("MobileNetV2 (fine-tuned)", 2_300_000,  35, 91),
]
print(f"{'Model':&lt;28} {'Params':>10} {'Train(min)':>11} {'Test Acc':>9}")
print("-"*60)
for name, params, t, acc in comparison:
    print(f"{name:&lt;28} {params:>10,} {t:>11} {acc:>8}%")
</code></pre>
  `,
};
