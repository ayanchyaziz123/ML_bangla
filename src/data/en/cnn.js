export const cnnEn = [
  {
    title: "CNN Architecture Math: Convolution, Stride, Padding & Parameter Count",
    description: "Mathematical foundations of convolution operations, kernels, feature maps, stride & padding output formulas, parameter counting, receptive field, and why CNNs excel at image tasks.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 13,
    slug: "cnn-architecture-math",
    content: `
      <h3>1. The Convolution Operation</h3>
      <p>A convolution slides a learnable <strong>kernel (filter)</strong> across the input, computing a dot product at each position to produce a <strong>feature map</strong>.</p>
      <pre><code># (I * K)[i,j] = sum_m sum_n  I[i+m, j+n] * K[m,n]
#
# I = input image  (H x W x C_in)
# K = kernel       (F x F x C_in)
# Output           (H_out x W_out x 1)  per filter
#
# Output size formula:
#   H_out = floor((H - F + 2*P) / S) + 1
#   W_out = floor((W - F + 2*P) / S) + 1
#
# P = padding,  S = stride,  F = filter size

import numpy as np

def conv2d_naive(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    H, W   = image.shape
    F      = kernel.shape[0]
    H_out  = (H - F) // stride + 1
    W_out  = (W - F) // stride + 1
    output = np.zeros((H_out, W_out))
    for i in range(0, H_out):
        for j in range(0, W_out):
            output[i, j] = np.sum(
                image[i*stride:i*stride+F, j*stride:j*stride+F] * kernel
            )
    return output

img    = np.random.randn(8, 8)
kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])   # vertical edge detector
out    = conv2d_naive(img, kernel, stride=1, padding=0)
print(f"Input: {img.shape}, Kernel: {kernel.shape}, Output: {out.shape}")</code></pre>

      <h3>2. Parameters, Depth & Multiple Filters</h3>
      <pre><code># Parameters per Conv layer:
#   (F * F * C_in + 1) * C_out
#   └─ kernel size ─┘   bias     └─ num filters ─┘
#
# Example: Conv2d(32 filters, 3x3) on RGB input (3 channels)
#   = (3 * 3 * 3 + 1) * 32 = 896 parameters
#
# Each filter produces one feature map → output has C_out channels

import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # in_channels=1, 28x28 input
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

model = SimpleConvNet()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")
# Layer 1: (3*3*1+1)*32  =  320 params
# Layer 2: (3*3*32+1)*64 = 18,496 params</code></pre>

      <h3>3. Stride & Padding</h3>
      <pre><code>import torch
import torch.nn as nn

# Stride=2: skip every other position → output half the size
# padding=1: add zeros around input → output same spatial size ('same', for 3x3 stride 1)
# padding=0 (default): no padding → output shrinks ('valid')

examples = [
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),   # 'valid' — shrinks by 2 each dim
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # 'same' — same spatial size
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # halves spatial size
]

inp = torch.randn(1, 1, 28, 28)
for layer in examples:
    out = layer(inp)
    print(f"stride={layer.stride}, padding={layer.padding} → {tuple(out.shape)}")</code></pre>
      <table>
        <thead><tr><th>Parameter</th><th>Effect</th><th>Common Values</th></tr></thead>
        <tbody>
          <tr><td>filters</td><td>Number of feature maps produced</td><td>32, 64, 128, 256</td></tr>
          <tr><td>kernel_size</td><td>Spatial extent of filter</td><td>3×3 (most common), 5×5, 1×1</td></tr>
          <tr><td>stride</td><td>Step size when sliding → controls output size</td><td>1 (default), 2 (downsampling)</td></tr>
          <tr><td>padding='same'</td><td>Preserve spatial dimensions</td><td>Use with stride=1</td></tr>
          <tr><td>padding='valid'</td><td>No padding, output shrinks</td><td>Use when size reduction is OK</td></tr>
        </tbody>
      </table>

      <h3>4. Receptive Field</h3>
      <pre><code># Receptive field: how large an area in the input does one output neuron "see"
#
# After 1 conv layer (3x3, stride 1): receptive field = 3x3
# After 2 conv layers (3x3, stride 1): receptive field = 5x5
# After 3 conv layers (3x3, stride 1): receptive field = 7x7
# After MaxPool(2x2) + conv: receptive field grows faster
#
# Deep networks with small filters = large receptive field + fewer params
# VGG insight: two 3x3 layers ≡ one 5x5 layer in receptive field,
#              but fewer parameters and one extra ReLU nonlinearity

# 1x1 convolution: cross-channel mixing, dimension reduction
# (used in Inception, ResNet bottleneck)
conv1x1 = nn.Conv2d(1, 64, kernel_size=1)
x = torch.relu(conv1x1(inp))   # 1x1 conv
print(f"1x1 conv output: {tuple(x.shape)}")</code></pre>

      <h3>5. Why CNNs Work for Images</h3>
      <table>
        <thead><tr><th>Property</th><th>What it means</th><th>Benefit</th></tr></thead>
        <tbody>
          <tr><td>Weight sharing</td><td>Same filter scans entire image</td><td>Far fewer params than Dense (28×28→Dense: 784 per neuron)</td></tr>
          <tr><td>Local connectivity</td><td>Each neuron sees only a local region</td><td>Exploit spatial locality of image features</td></tr>
          <tr><td>Translation invariance</td><td>Feature detector works anywhere in image</td><td>Detect cat in top-left or bottom-right</td></tr>
          <tr><td>Hierarchical features</td><td>Early layers: edges → later: textures → deep: objects</td><td>Automatic feature engineering</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Pooling, Batch Normalization & Dropout in CNNs",
    description: "MaxPooling vs AveragePooling vs GlobalAveragePooling math, Batch Normalization placement in CNN blocks, SpatialDropout2D, and comparison of training with vs without BatchNorm on CIFAR-10.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 11,
    slug: "cnn-pooling-batchnorm",
    content: `
      <h3>1. Pooling Layers</h3>
      <p>Pooling reduces spatial dimensions — fewer parameters downstream, some translation invariance, prevents overfitting.</p>
      <pre><code># Output size after pooling (no padding):
#   H_out = (H - pool_size) / stride + 1
#
# MaxPool(2x2, stride=2): halves H and W  →  most common
# AvgPool(2x2, stride=2): mean instead of max

import torch
import torch.nn as nn

# Demonstrate max vs avg pooling on a tiny feature map
feat_map = torch.tensor([[1, 3, 2, 4],
                         [5, 6, 1, 2],
                         [3, 1, 4, 7],
                         [2, 8, 3, 1]], dtype=torch.float32).reshape(1,1,4,4)

max_pool = nn.MaxPool2d(2, stride=2)(feat_map)
avg_pool = nn.AvgPool2d(2, stride=2)(feat_map)
gap      = nn.AdaptiveAvgPool2d(1)(feat_map)

print("MaxPool:", max_pool.squeeze().numpy())   # [[6,4],[8,7]]
print("AvgPool:", avg_pool.squeeze().numpy())   # [[3.75,2.25],[3.5,3.75]]
print("GAP:    ", gap.squeeze().numpy())        # single value per channel</code></pre>
      <table>
        <thead><tr><th>Pooling Type</th><th>Operation</th><th>Use When</th></tr></thead>
        <tbody>
          <tr><td>MaxPooling2D</td><td>Takes maximum in each window</td><td>Feature detection — was it present?</td></tr>
          <tr><td>AveragePooling2D</td><td>Mean of each window</td><td>Smooth backgrounds, texture classification</td></tr>
          <tr><td>GlobalAveragePooling2D</td><td>One mean value per entire feature map</td><td>Modern CNNs — replaces Flatten+Dense, less overfit</td></tr>
          <tr><td>Stride=2 Conv</td><td>Learnable downsampling</td><td>ResNet, modern architectures prefer over MaxPool</td></tr>
        </tbody>
      </table>

      <h3>2. Batch Normalization</h3>
      <pre><code># BatchNorm normalizes each feature map across the batch:
#   x_hat = (x - mean) / sqrt(var + epsilon)
#   y = gamma * x_hat + beta    (gamma, beta are learnable)
#
# Placement: Conv → BN → ReLU  (original paper)
#            Conv → ReLU → BN  (sometimes used)
#
# Benefits:
#   1. Faster training (higher lr possible)
#   2. Reduces internal covariate shift
#   3. Slight regularization (reduces need for dropout)
#   4. Less sensitive to weight initialization

class CNNWithBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)

model_with_bn = CNNWithBN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_with_bn.parameters(), lr=0.001)</code></pre>

      <h3>3. Dropout in CNNs</h3>
      <pre><code">import torch
import torch.nn as nn

# Regular Dropout: randomly zeroes individual activations
# Dropout2d: randomly zeroes entire feature maps (channels)
# → Better for CNNs because adjacent pixels are correlated

class CNNWithDropout(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1   = nn.ReLU()
        self.spdrop1 = nn.Dropout2d(0.25)   # drop entire feature maps
        self.pool1   = nn.MaxPool2d(2)

        self.conv2   = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2   = nn.ReLU()
        self.spdrop2 = nn.Dropout2d(0.25)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.5)             # regular dropout before Linear
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.spdrop1(self.relu1(self.conv1(x))))
        x = self.gap(self.spdrop2(self.relu2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

model = CNNWithDropout()</code></pre>

      <h3>4. BN Impact on CIFAR-10</h3>
      <pre><code">import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

class ComparisonCNN(nn.Module):
    def __init__(self, use_bn=True, num_classes=10):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32) if use_bn else None
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64) if use_bn else None
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.pool1(self.relu1(x))

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.pool2(self.relu2(x))

        x = self.gap(x)
        x = self.flatten(x)
        return self.fc(x)

def train_and_eval(use_bn, epochs=15, lr=0.01):
    model = ComparisonCNN(use_bn=use_bn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
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
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        best_val_acc = max(best_val_acc, correct / total)
    return best_val_acc

acc_no_bn = train_and_eval(use_bn=False)
acc_bn    = train_and_eval(use_bn=True)
print(f"Without BN: {acc_no_bn:.4f}")
print(f"With BN:    {acc_bn:.4f}")</code></pre>
    `,
  },
  {
    title: "Famous CNN Architectures: LeNet to MobileNet — An Evolution",
    description: "LeNet-5, AlexNet, VGG, ResNet (skip connections), Inception, and MobileNet — key innovations, parameter counts, and how each solved the limitations of its predecessor.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 14,
    slug: "cnn-famous-architectures",
    content: `
      <h3>1. Architecture Timeline</h3>
      <table>
        <thead><tr><th>Architecture</th><th>Year</th><th>ImageNet Top-5 Err</th><th>Parameters</th><th>Key Innovation</th></tr></thead>
        <tbody>
          <tr><td>LeNet-5</td><td>1998</td><td>N/A (MNIST)</td><td>60K</td><td>First practical CNN</td></tr>
          <tr><td>AlexNet</td><td>2012</td><td>15.3%</td><td>60M</td><td>ReLU, Dropout, GPU training</td></tr>
          <tr><td>VGG-16</td><td>2014</td><td>7.3%</td><td>138M</td><td>Depth with 3×3 filters only</td></tr>
          <tr><td>GoogLeNet</td><td>2014</td><td>6.7%</td><td>6.8M</td><td>Inception module (parallel branches)</td></tr>
          <tr><td>ResNet-50</td><td>2015</td><td>3.6%</td><td>25.6M</td><td>Skip connections, 152 layers</td></tr>
          <tr><td>MobileNetV2</td><td>2018</td><td>~7%</td><td>3.4M</td><td>Depthwise separable convolutions</td></tr>
          <tr><td>EfficientNetB0</td><td>2019</td><td>~5%</td><td>5.3M</td><td>Compound scaling (width/depth/resolution)</td></tr>
        </tbody>
      </table>

      <h3>2. ResNet — Solving Vanishing Gradient with Skip Connections</h3>
      <pre><code>import torch
import torch.nn as nn

# ResNet insight: learning residual F(x) = H(x) - x is easier than H(x) directly
# Skip connection: output = F(x) + x   (identity shortcut)
# If gradient vanishes through F(x), it still flows through the identity path

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(filters)
        self.relu  = nn.ReLU()

        # Adjust shortcut if shape changed
        self.has_shortcut = downsample or in_channels != filters
        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, filters, 1, stride=stride)
            self.shortcut_bn   = nn.BatchNorm2d(filters)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x)) if self.has_shortcut else x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu(y + shortcut)   # F(x) + x
        return y

# Build a small ResNet
class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.stem_bn   = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU()
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 64, downsample=True)
        self.block3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SmallResNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Custom ResNet params: {total_params:,}")</code></pre>

      <h3>3. Inception Module — Parallel Multi-Scale Filters</h3>
      <pre><code">import torch
import torch.nn.functional as F

def inception_module(x, in_channels, f1, f3, f5, fpool):
    # Branch 1: 1x1 conv
    b1 = F.relu(nn.Conv2d(in_channels, f1, 1)(x))
    # Branch 2: 1x1 → 3x3
    b2 = F.relu(nn.Conv2d(in_channels, f3, 1)(x))
    b2 = F.relu(nn.Conv2d(f3, f3, 3, padding=1)(b2))
    # Branch 3: 1x1 → 5x5
    b3 = F.relu(nn.Conv2d(in_channels, f5, 1)(x))
    b3 = F.relu(nn.Conv2d(f5, f5, 5, padding=2)(b3))
    # Branch 4: MaxPool → 1x1
    b4 = nn.MaxPool2d(3, stride=1, padding=1)(x)
    b4 = F.relu(nn.Conv2d(in_channels, fpool, 1)(b4))
    # Concatenate along channel axis
    return torch.cat([b1, b2, b3, b4], dim=1)</code></pre>

      <h3>4. MobileNet — Efficient Convolutions</h3>
      <pre><code"># Standard Conv2D:      H*W*C_in*C_out*F*F  multiplications
# Depthwise Separable:  H*W*C_in*F*F + H*W*C_in*C_out  (much less!)
# Reduction factor ≈ 1/C_out + 1/F^2  → ~8-9x fewer operations

# In PyTorch, use groups=in_channels for depthwise, then 1x1 for pointwise:
class DepthwiseSeparableConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise = nn.Conv2d(64, 64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

dws_layer = DepthwiseSeparableConv()

# Load pre-trained MobileNetV2 and run inference
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

weights = MobileNet_V2_Weights.IMAGENET1K_V1
base = models.mobilenet_v2(weights=weights)
base.eval()
preprocess = weights.transforms()   # scales/normalizes to ImageNet stats
categories = weights.meta["categories"]

dummy_img = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8)
with torch.no_grad():
    preds = base(preprocess(dummy_img))
top3 = torch.topk(preds.softmax(dim=1), 3)
print([categories[i] for i in top3.indices[0]])</code></pre>

      <h3>5. Pre-trained ResNet50 Inference</h3>
      <pre><code">import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

weights = ResNet50_Weights.IMAGENET1K_V2
resnet = resnet50(weights=weights)
resnet.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]

# Load and preprocess an image
img = Image.open('dog.jpg').convert('RGB')
x   = preprocess(img).unsqueeze(0)   # add batch dimension

with torch.no_grad():
    preds = resnet(x)
top3 = torch.topk(preds.softmax(dim=1), 3)
print('Predicted:', [categories[i] for i in top3.indices[0]])</code></pre>
    `,
  },
  {
    title: "Object Detection: From R-CNN to YOLO",
    description: "Image classification vs detection vs segmentation, R-CNN family evolution (R-CNN, Fast R-CNN, Faster R-CNN), YOLO's single-pass architecture, anchor boxes, NMS, and mAP evaluation.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 13,
    slug: "cnn-object-detection",
    content: `
      <h3>1. Three Computer Vision Tasks</h3>
      <table>
        <thead><tr><th>Task</th><th>Output</th><th>Question answered</th><th>Example</th></tr></thead>
        <tbody>
          <tr><td>Image Classification</td><td>Single class label</td><td>What is in this image?</td><td>"cat"</td></tr>
          <tr><td>Object Detection</td><td>Bounding boxes + class labels</td><td>What and WHERE?</td><td>"cat at [x,y,w,h]"</td></tr>
          <tr><td>Semantic Segmentation</td><td>Per-pixel class label</td><td>What class is each pixel?</td><td>Pixel-level "cat" mask</td></tr>
          <tr><td>Instance Segmentation</td><td>Per-instance pixel mask</td><td>Each individual object's pixels?</td><td>Separate mask per cat</td></tr>
        </tbody>
      </table>

      <h3>2. R-CNN Family Evolution</h3>
      <pre><code># R-CNN (2014): Regions with CNN features
#   1. Selective Search → ~2000 region proposals
#   2. Warp each region → 227x227
#   3. CNN feature extraction (per region — SLOW)
#   4. SVM classification + bbox regression
#   Problem: ~47s per image, each region processed separately
#
# Fast R-CNN (2015):
#   1. CNN on ENTIRE image once → feature map
#   2. Project region proposals onto feature map (RoI Pooling)
#   3. Classify + regress from pooled features
#   Speed: ~2s per image
#
# Faster R-CNN (2016):
#   1. CNN on entire image
#   2. Region Proposal Network (RPN) — replaces Selective Search
#      RPN shares convolutional features with detector
#   3. RoI Pooling → classify + regress
#   Speed: ~200ms per image
#   Still two-stage: propose then classify</code></pre>

      <h3>3. YOLO — You Only Look Once</h3>
      <pre><code"># YOLO (single-stage): one forward pass for everything
#
# Divide image into S×S grid (e.g., 13×13 for YOLOv3)
# Each cell predicts B bounding boxes + confidence + C class probabilities
#
# Bounding box prediction: [x, y, w, h, confidence]
#   x, y: center of box relative to cell
#   w, h: relative to image size (log-space prediction)
#   confidence = P(object) * IoU(pred, truth)
#
# Anchor boxes: predefined aspect ratios per cell
#   Each cell predicts offsets FROM anchor boxes, not absolute coords
#
# Loss = λ_coord * bbox_loss + objectness_loss + class_loss
#
# YOLOv8 with ultralytics (easiest to use):
from ultralytics import YOLO

model = YOLO('yolov8n.pt')        # load nano model
results = model('image.jpg')      # run inference
results[0].show()                 # display with boxes

# Train on custom data:
# model.train(data='coco128.yaml', epochs=50, imgsz=640)</code></pre>

      <h3>4. IoU & Non-Maximum Suppression</h3>
      <pre><code"># IoU (Intersection over Union):
#   IoU = area(A ∩ B) / area(A ∪ B)
#   IoU > 0.5: good detection  |  IoU > 0.75: strict  |  IoU = 1.0: perfect

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

# NMS: remove overlapping duplicate detections
def nms(boxes, scores, iou_threshold=0.5):
    order    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep     = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if compute_iou(boxes[i], boxes[j]) &lt; iou_threshold]
    return keep</code></pre>

      <h3>5. Evaluation: mAP</h3>
      <table>
        <thead><tr><th>Metric</th><th>Definition</th><th>Notes</th></tr></thead>
        <tbody>
          <tr><td>IoU</td><td>Overlap between predicted and ground-truth box</td><td>Threshold usually 0.5</td></tr>
          <tr><td>Precision</td><td>TP / (TP + FP)</td><td>Of all detections, how many are correct?</td></tr>
          <tr><td>Recall</td><td>TP / (TP + FN)</td><td>Of all objects, how many found?</td></tr>
          <tr><td>AP (per class)</td><td>Area under Precision-Recall curve</td><td>Averaged over IoU thresholds</td></tr>
          <tr><td>mAP</td><td>Mean AP across all classes</td><td>Primary detection benchmark</td></tr>
        </tbody>
      </table>
      <table>
        <thead><tr><th>Model</th><th>Speed (ms)</th><th>mAP COCO</th><th>Type</th></tr></thead>
        <tbody>
          <tr><td>Faster R-CNN</td><td>~200</td><td>~37</td><td>Two-stage</td></tr>
          <tr><td>YOLOv5s</td><td>~6</td><td>37.4</td><td>One-stage</td></tr>
          <tr><td>YOLOv8n</td><td>~3</td><td>37.3</td><td>One-stage</td></tr>
          <tr><td>YOLOv8x</td><td>~14</td><td>53.9</td><td>One-stage</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Semantic Segmentation: U-Net to Mask R-CNN",
    description: "Fully convolutional networks, transposed convolutions for upsampling, U-Net encoder-decoder with skip connections built in PyTorch, Mask R-CNN, and mIoU evaluation.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 12,
    slug: "cnn-semantic-segmentation",
    content: `
      <h3>1. Segmentation Task Types</h3>
      <table>
        <thead><tr><th>Task</th><th>Output</th><th>Distinguishes instances?</th><th>Architecture</th></tr></thead>
        <tbody>
          <tr><td>Semantic Segmentation</td><td>Per-pixel class label</td><td>No — all cats same label</td><td>FCN, U-Net, DeepLab</td></tr>
          <tr><td>Instance Segmentation</td><td>Per-pixel + per-instance</td><td>Yes — each cat separate</td><td>Mask R-CNN</td></tr>
          <tr><td>Panoptic Segmentation</td><td>Semantic + Instance combined</td><td>Yes</td><td>Panoptic FPN</td></tr>
        </tbody>
      </table>

      <h3>2. Fully Convolutional Network (FCN)</h3>
      <pre><code># Key idea: replace all Dense layers with Conv layers
# → can accept any input size, output spatial map instead of single label
#
# Problem: Conv + MaxPool reduces spatial resolution → blurry output
# Solution: Upsample back to original resolution
#
# Transposed Convolution (deconv): learnable upsampling
#   → opposite of conv: spreads input, pads with zeros, applies filter
#   nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=..., output_padding=...)
#     stride=2 → doubles spatial dimensions

import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: downsample
        self.enc1  = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)     # 64x64
        self.enc2  = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)     # 32x32

        # Decoder: upsample
        self.up1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)  # 64x64
        self.up2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 128x128
        self.out_conv = nn.Conv2d(32, 1, 1)   # binary segmentation

    def forward(self, x):
        x = torch.relu(self.enc1(x))
        x = self.pool1(x)
        x = torch.relu(self.enc2(x))
        x = self.pool2(x)
        x = torch.relu(self.up1(x))
        x = torch.relu(self.up2(x))
        return torch.sigmoid(self.out_conv(x))

fcn = SimpleFCN()
inp = torch.randn(1, 3, 128, 128)
out = fcn(inp)
print(f"Input: {tuple(inp.shape)} → Output: {tuple(out.shape)}")</code></pre>

      <h3>3. U-Net — Skip Connections for Sharp Boundaries</h3>
      <pre><code">class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        # Encoder (contracting path)
        self.c1_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.c1_relu1 = nn.ReLU()
        self.c1_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.c1_relu2 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2)   # 64x64

        self.c2_conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.c2_relu1 = nn.ReLU()
        self.c2_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c2_relu2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(2)   # 32x32

        # Bottleneck
        self.bn_conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_relu1 = nn.ReLU()
        self.bn_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_relu2 = nn.ReLU()

        # Decoder (expanding path) with skip connections
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)   # 64x64
        self.d1_conv1 = nn.Conv2d(64, 32, 3, padding=1)   # 64 = 32(up1) + 32(skip c2)
        self.d1_relu1 = nn.ReLU()
        self.d1_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.d1_relu2 = nn.ReLU()

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)   # 128x128
        self.d2_conv1 = nn.Conv2d(32, 16, 3, padding=1)   # 32 = 16(up2) + 16(skip c1)
        self.d2_relu1 = nn.ReLU()
        self.d2_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.d2_relu2 = nn.ReLU()

        self.out_conv = nn.Conv2d(16, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x):
        c1 = self.c1_relu1(self.c1_conv1(x))
        c1 = self.c1_relu2(self.c1_conv2(c1))
        p1 = self.p1(c1)

        c2 = self.c2_relu1(self.c2_conv1(p1))
        c2 = self.c2_relu2(self.c2_conv2(c2))
        p2 = self.p2(c2)

        bn = self.bn_relu1(self.bn_conv1(p2))
        bn = self.bn_relu2(self.bn_conv2(bn))

        u1 = self.up1(bn)
        u1 = torch.cat([u1, c2], dim=1)   # skip connection from encoder
        d1 = self.d1_relu1(self.d1_conv1(u1))
        d1 = self.d1_relu2(self.d1_conv2(d1))

        u2 = self.up2(d1)
        u2 = torch.cat([u2, c1], dim=1)   # skip connection
        d2 = self.d2_relu1(self.d2_conv1(u2))
        d2 = self.d2_relu2(self.d2_conv2(d2))

        out = self.out_conv(d2)
        return torch.sigmoid(out) if self.num_classes == 1 else torch.softmax(out, dim=1)

model = UNet(in_channels=1, num_classes=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)</code></pre>

      <h3>4. Training U-Net with Custom Data</h3>
      <pre><code"># Synthetic segmentation data for demonstration
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def make_segmentation_data(n=200, img_size=128):
    X = np.zeros((n, 1, img_size, img_size), dtype='float32')
    Y = np.zeros((n, 1, img_size, img_size), dtype='float32')
    for i in range(n):
        # random circle
        cx, cy = np.random.randint(20, img_size-20, 2)
        r      = np.random.randint(10, 30)
        xx, yy = np.ogrid[:img_size, :img_size]
        mask   = ((xx-cx)**2 + (yy-cy)**2) &lt;= r**2
        X[i, 0] = np.random.randn(img_size, img_size) * 0.1
        X[i, 0][mask] += 1.0   # bright circle
        Y[i, 0][mask]  = 1.0
    return torch.from_numpy(X), torch.from_numpy(Y)

X, Y = make_segmentation_data(400)
X_tr, X_te, Y_tr, Y_te = X[:320], X[320:], Y[:320], Y[320:]
train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=16, shuffle=True)

model = UNet(in_channels=1, num_classes=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_te), Y_te)
    print(f"Epoch {epoch+1}: val_loss={val_loss.item():.4f}")</code></pre>

      <h3>5. Evaluation Metrics</h3>
      <pre><code"># Pixel Accuracy: correct pixels / total pixels (misleading on imbalanced data)
# IoU (Jaccard Index): intersection / union per class
# mIoU: mean IoU across all classes (primary segmentation metric)

def compute_iou(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).float()
    intersection = (y_true * y_pred_bin).sum()
    union        = y_true.sum() + y_pred_bin.sum() - intersection
    return (intersection / (union + 1e-8)).item()

model.eval()
with torch.no_grad():
    y_pred = model(X_te)
ious = [compute_iou(Y_te[i,0], y_pred[i,0]) for i in range(len(X_te))]
print(f"Mean IoU: {np.mean(ious):.4f}")</code></pre>
    `,
  },
  {
    title: "CIFAR-10 Project: Custom CNN, ResNet & Transfer Learning with Grad-CAM",
    description: "End-to-end project comparing a simple CNN, ResNet with skip connections, and MobileNetV2 transfer learning on CIFAR-10 — data augmentation, LR scheduling, confusion matrix, and Grad-CAM visualization.",
    date: "২৩ মে, ২০২৬",
    category: "Convolutional Neural Network",
    readTime: 14,
    slug: "cnn-project-image-classification",
    content: `
      <h3>1. Dataset & Setup</h3>
      <pre><code>import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
print(f"Train: {len(train_set)}, Test: {len(test_set)}")

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_set[i]
    ax.imshow(img.permute(1, 2, 0)); ax.set_title(class_names[label]); ax.axis('off')
plt.tight_layout(); plt.show()</code></pre>

      <h3>2. Data Augmentation Pipeline</h3>
      <pre><code">augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(32, scale=(0.85, 1.0)),
    transforms.ColorJitter(contrast=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Visualize augmented images
sample_img, _ = train_set[0]
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax in axes:
    img = augment(sample_img)
    ax.imshow(img.permute(1, 2, 0)); ax.axis('off')
plt.suptitle('Augmented versions of one image'); plt.show()</code></pre>

      <h3>3. Three Models</h3>
      <pre><code"># Note: augmentation (previous block) is applied via the dataset's transform
# pipeline rather than embedded as a model layer — the idiomatic PyTorch way.

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.conv3(x))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels, filters, 1) if in_channels != filters else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(filters)
        self.relu  = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + shortcut)

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 32, 3, padding=1)
        self.stem_bn   = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU()

        in_ch = 32
        self.res_blocks = nn.ModuleList()
        for f in [32, 64, 128]:
            self.res_blocks.append(ResidualBlock(in_ch, f))
            in_ch = f

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class MobileNetTransfer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in base.parameters():
            p.requires_grad = False   # freeze base
        self.base = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = self.base(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)</code></pre>

      <h3>4. Training & Comparison</h3>
      <pre><code">from torch.utils.data import DataLoader, random_split

train_size = int(0.9 * len(train_set))
val_size   = len(train_set) - train_size
train_subset, val_subset = random_split(train_set, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset, batch_size=64)
test_loader  = DataLoader(test_set, batch_size=64)

def train_model(model, epochs=50, patience=8):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-6)

    best_val_acc, epochs_no_improve, best_state = 0.0, 0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
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
    return model, best_val_acc

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total   += yb.size(0)
    return correct / total

models_dict = {'Simple CNN': SimpleCNN(), 'ResNet': ResNetCNN(), 'MobileNetV2': MobileNetTransfer()}
results = {}
for name, m in models_dict.items():
    m, val_acc = train_model(m, epochs=50)
    test_acc = evaluate(m)
    params = sum(p.numel() for p in m.parameters())
    results[name] = {'val_acc': val_acc, 'test_acc': test_acc, 'params': params}

for name, r in results.items():
    print(f"{name:15s}: test={r['test_acc']:.4f}, params={r['params']:,}")</code></pre>

      <h3>5. Grad-CAM Visualization</h3>
      <pre><code"># Grad-CAM: gradient of class score w.r.t. last conv layer activations
# Positive gradients → regions important for this class

def grad_cam(model, img, class_idx, target_layer):
    activations, gradients = [], []
    fwd_handle = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bwd_handle = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    model.eval()
    out = model(img.unsqueeze(0))
    model.zero_grad()
    out[0, class_idx].backward()
    fwd_handle.remove(); bwd_handle.remove()

    conv_out = activations[0][0]                 # (C, H, W)
    grads    = gradients[0][0]                    # (C, H, W)
    weights  = grads.mean(dim=(1, 2))             # global average pool
    cam      = torch.relu((conv_out * weights[:, None, None]).sum(dim=0))
    cam      = cam / (cam.max() + 1e-8)
    cam      = torch.nn.functional.interpolate(
        cam[None, None], size=img.shape[1:], mode='bilinear', align_corners=False
    ).squeeze().detach().numpy()
    return cam

import cv2
img_tensor, label = test_set[42]
model           = models_dict['Simple CNN']
last_conv_layer = model.conv3   # last Conv2d layer in SimpleCNN
cam             = grad_cam(model, img_tensor, label, last_conv_layer)

img_np    = img_tensor.permute(1, 2, 0).numpy()
heatmap   = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay   = cv2.addWeighted(np.uint8(img_np * 255), 0.6, heatmap, 0.4, 0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(img_np); axes[0].set_title('Original')
axes[1].imshow(cam, cmap='jet'); axes[1].set_title('Grad-CAM')
axes[2].imshow(overlay[:,:,::-1]); axes[2].set_title('Overlay')
[ax.axis('off') for ax in axes]; plt.tight_layout(); plt.show()</code></pre>

      <h3>Summary</h3>
      <table>
        <thead><tr><th>Model</th><th>Test Accuracy</th><th>Parameters</th><th>Training Time</th></tr></thead>
        <tbody>
          <tr><td>Simple CNN</td><td>~78%</td><td>~200K</td><td>Fast</td></tr>
          <tr><td>ResNet (custom)</td><td>~85%</td><td>~400K</td><td>Medium</td></tr>
          <tr><td>MobileNetV2 (transfer)</td><td>~88%</td><td>~3.4M (frozen)</td><td>Fast (frozen base)</td></tr>
        </tbody>
      </table>
    `,
  },
];
