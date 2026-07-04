export const cnn_2_pooling_batchnorm = {
  title: "Pooling, Batch Normalization ও Dropout: CNN-কে শক্তিশালী করার কৌশল",
  description: "MaxPooling, AveragePooling ও Global Average Pooling-এর গণিত, Batch Normalization-এর কাজের প্রক্রিয়া ও CNN-এ placement, SpatialDropout2D বনাম Dropout এবং CIFAR-10-এ BN-এর প্রভাব তুলনা।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 11,
  slug: "cnn-pooling-batchnorm",
  content: `
    <h3>১. Pooling: Spatial Downsampling</h3>
    <p>Pooling layer spatial dimensions (height × width) কমায়, channel count অপরিবর্তিত রাখে। এর দুটি উদ্দেশ্য: parameter count কমানো এবং translation invariance বাড়ানো।</p>
    <p><strong>Output size after pooling:</strong></p>
    <p><strong>Output = ⌊(W - pool_size) / stride⌋ + 1</strong></p>
    <p>সাধারণত pool_size=2, stride=2 ব্যবহার হয় → output = W/2 (exact halving)।</p>
    <table>
      <thead>
        <tr><th>Pooling Type</th><th>Operation</th><th>Advantage</th><th>Disadvantage</th></tr>
      </thead>
      <tbody>
        <tr><td>MaxPooling2D</td><td>Region-এর maximum নেয়</td><td>Strongest feature preserve, translation invariant</td><td>Position information হারায়</td></tr>
        <tr><td>AveragePooling2D</td><td>Region-এর mean নেয়</td><td>Smooth feature aggregation</td><td>Weak features dilute হয়</td></tr>
        <tr><td>GlobalAveragePooling2D</td><td>পুরো feature map-এর mean</td><td>Flatten replace করে, fewer params</td><td>Fine spatial info হারায়</td></tr>
        <tr><td>GlobalMaxPooling2D</td><td>পুরো feature map-এর max</td><td>Most prominent feature capture</td><td>Spatial context নেই</td></tr>
      </tbody>
    </table>
    <pre><code>import numpy as np

def max_pool_2d(feature_map, pool_size=2, stride=2):
    """Manual MaxPooling2D"""
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = feature_map[
                i*stride : i*stride + pool_size,
                j*stride : j*stride + pool_size
            ]
            output[i, j] = np.max(region)    # MaxPool
            # output[i, j] = np.mean(region) # AvgPool হলে mean নিতাম

    return output

# উদাহরণ
fm = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 1, 0],
    [4, 1, 3, 2]
], dtype=float)

result = max_pool_2d(fm, pool_size=2, stride=2)
print("Feature Map (4x4):")
print(fm)
print("\nAfter MaxPool (2x2, stride=2):")
print(result)
# [[6, 4], [4, 3]] — প্রতিটি 2x2 region-এর max
</code></pre>

    <h3>২. Global Average Pooling (GAP): Modern CNN-এর Flatten বিকল্প</h3>
    <p>Traditional CNN-এ শেষে Flatten → Dense ব্যবহার হতো। Modern CNN (ResNet, MobileNet, EfficientNet) Global Average Pooling ব্যবহার করে।</p>
    <p><strong>GAP:</strong> প্রতিটি feature map (H×W) → একটি সংখ্যা (সেই map-এর mean)।</p>
    <p>যদি শেষ feature tensor shape হয় (7, 7, 512), তাহলে GAP দিয়ে (512,) vector পাওয়া যায়।</p>
    <ul>
      <li>Flatten-এর তুলনায় অনেক কম parameters</li>
      <li>Overfitting কম হয়</li>
      <li>Different input sizes handle করতে পারে (fully convolutional)</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn

# Flatten vs GAP comparison
class ModelWithFlatten(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()      # 8*8*128 = 8192 neurons
        self.fc1 = nn.Linear(8192, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)

class ModelWithGAP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()   # (128,) vector
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)

m_flat = ModelWithFlatten()
m_gap  = ModelWithGAP()

flat_params = sum(p.numel() for p in m_flat.parameters())
gap_params  = sum(p.numel() for p in m_gap.parameters())
print(f"Flatten model params: {flat_params:,}")
print(f"GAP model params:     {gap_params:,}")
# GAP model অনেক কম params — Linear(8192→256) বাদ দেওয়া হয়েছে
</code></pre>

    <h3>৩. Batch Normalization: CNN-এ কীভাবে কাজ করে</h3>
    <p>Batch Normalization (BN) প্রতিটি mini-batch-এ activation normalize করে। CNN-এ প্রতিটি channel-এর সব spatial locations জুড়ে normalize হয়।</p>
    <p><strong>BN Formula:</strong></p>
    <p><strong>x̂ = (x - μ_B) / √(σ²_B + ε)</strong></p>
    <p><strong>y = γ · x̂ + β</strong></p>
    <p>এখানে:</p>
    <ul>
      <li><strong>μ_B</strong> = batch mean (channel-wise)</li>
      <li><strong>σ²_B</strong> = batch variance</li>
      <li><strong>ε</strong> = numerical stability-এর জন্য ছোট constant (1e-5)</li>
      <li><strong>γ, β</strong> = learnable scale ও shift parameters</li>
    </ul>
    <p><strong>CNN-এ BN Placement:</strong> সর্বোত্তম pattern হলো <code>Conv2D → BatchNorm → ReLU</code>।</p>
    <pre><code>import torch
import torch.nn as nn

# Standard CNN block with BN
class ConvBNReLU(nn.Module):
    """Conv → BatchNorm → ReLU block"""
    def __init__(self, in_channels, filters, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, filters, kernel_size,
            stride=stride, padding=kernel_size // 2,
            bias=False    # BN ব্যবহারে bias redundant, তাই False
        )
        self.bn   = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# BN-এর learnable parameters: 4 per channel (gamma, beta, moving_mean, moving_var)
# 64 filters → 4 * 64 = 256 params (2 trainable + 2 non-trainable)

class BNDemoModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBNReLU(3, 32)
        self.block2 = ConvBNReLU(32, 64)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)

model = BNDemoModel()
print(model)
</code></pre>

    <h3>৪. BN-এর সুবিধা ও Training প্রভাব</h3>
    <p>Batch Normalization কেন এত জনপ্রিয়:</p>
    <ul>
      <li><strong>Faster convergence:</strong> Higher learning rate ব্যবহার করা যায় (0.1 vs 0.001)</li>
      <li><strong>Internal Covariate Shift কমায়:</strong> প্রতিটি layer stable distribution দেখে</li>
      <li><strong>Regularization effect:</strong> Dropout-এর মতো কিছুটা noise inject করে</li>
      <li><strong>Gradient flow উন্নত করে:</strong> Deep network-এ vanishing gradient কমায়</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 load
transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

class BNComparisonCNN(nn.Module):
    def __init__(self, use_batchnorm=True, num_classes=10):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=not use_batchnorm)
        self.bn1   = nn.BatchNorm2d(32) if use_batchnorm else None
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=not use_batchnorm)
        self.bn2   = nn.BatchNorm2d(64) if use_batchnorm else None
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm: x = self.bn1(x)
        x = self.pool1(self.relu1(x))

        x = self.conv2(x)
        if self.use_batchnorm: x = self.bn2(x)
        x = self.pool2(self.relu2(x))

        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)

def train_and_eval(model, lr, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                correct += (model(xb).argmax(dim=1) == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total
    return val_acc

# Without BN — lower LR needed
model_no_bn = BNComparisonCNN(use_batchnorm=False)
acc_no_bn = train_and_eval(model_no_bn, lr=0.001)

# With BN — higher LR ব্যবহার করা যায়
model_bn = BNComparisonCNN(use_batchnorm=True)
acc_bn = train_and_eval(model_bn, lr=0.01)   # 10x বেশি LR

print(f"Without BN - Final val accuracy: {acc_no_bn:.4f}")
print(f"With BN    - Final val accuracy: {acc_bn:.4f}")
# With BN সাধারণত 3-5% বেশি accuracy দেবে এবং দ্রুত converge করবে
</code></pre>

    <h3>৫. Dropout in CNNs: Regular vs SpatialDropout2D</h3>
    <p>CNN-এ regular Dropout এবং SpatialDropout2D দুটি আলাদা কাজ করে।</p>
    <p><strong>Regular Dropout:</strong> Individual activation randomly zero করে। কিন্তু CNN-এ spatially correlated activations-এ এটি কম effective।</p>
    <p><strong>SpatialDropout2D:</strong> পুরো feature map (channel) randomly zero করে। Feature maps-এর মধ্যে redundancy কমায়, stronger regularization।</p>
    <table>
      <thead>
        <tr><th>Dropout Type</th><th>কী zero করে</th><th>কোথায় ব্যবহার</th><th>Effect</th></tr>
      </thead>
      <tbody>
        <tr><td>Dropout(rate)</td><td>Individual pixels</td><td>Dense layers, FC layers</td><td>Mild regularization</td></tr>
        <tr><td>SpatialDropout2D(rate)</td><td>Entire feature map</td><td>Conv layers-এর পর</td><td>Strong regularization</td></tr>
        <tr><td>Dropout after GAP</td><td>Individual neurons</td><td>GAP → Dropout → Dense</td><td>Modern best practice</td></tr>
      </tbody>
    </table>
    <pre><code>import torch
import torch.nn as nn

class CNNWithDropout(nn.Module):
    def __init__(self, spatial_dropout_rate=0.2, dropout_rate=0.5, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1   = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(32)
        self.relu1   = nn.ReLU()
        self.spdrop1 = nn.Dropout2d(spatial_dropout_rate)   # Feature map-level dropout
        self.pool1   = nn.MaxPool2d(2)

        # Block 2
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.relu2   = nn.ReLU()
        self.spdrop2 = nn.Dropout2d(spatial_dropout_rate)
        self.pool2   = nn.MaxPool2d(2)

        # Classifier
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)             # Regular dropout before Linear
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.spdrop1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.pool2(self.spdrop2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

model = CNNWithDropout()
print(model)

# Note: Dropout is only active during training — call model.train()
# Inference-এ model.eval() call করলে automatically disabled হয়
</code></pre>

    <h3>৬. Complete CNN Block: সবকিছু একসাথে</h3>
    <p>Production-grade CNN block: Conv2D → BatchNorm → ReLU → MaxPool — এই pattern বারবার repeat করা হয়।</p>
    <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class CNNBlock(nn.Module):
    """Standard CNN block: Conv→BN→ReLU→(MaxPool)"""
    def __init__(self, in_channels, filters, pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU()

        self.do_pool = pool
        if pool:
            self.pool  = nn.MaxPool2d(2)
            self.spdrop = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.do_pool:
            x = self.pool(x)
            x = self.spdrop(x)
        return x

class FullCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = CNNBlock(3,   32, pool=True)   # → 16x16x32
        self.block2 = CNNBlock(32,  64, pool=True)   # → 8x8x64
        self.block3 = CNNBlock(64, 128, pool=True)   # → 4x4x128

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()              # → (128,)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

# Load data
transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

model = FullCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

best_val_acc, epochs_no_improve, best_state = 0.0, 0, None
patience = 10   # early stopping patience

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total, val_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            val_loss_sum += criterion(out, yb).item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total   += yb.size(0)
    val_acc = correct / total
    scheduler.step(val_loss_sum)

    if val_acc > best_val_acc:
        best_val_acc, best_state, epochs_no_improve = val_acc, model.state_dict(), 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:   # early stopping
            break

model.load_state_dict(best_state)   # restore best weights

# এই model CIFAR-10-এ ~80-85% accuracy দেওয়া উচিত
print(f"Test accuracy: {best_val_acc:.4f}")
</code></pre>
  `,
};
