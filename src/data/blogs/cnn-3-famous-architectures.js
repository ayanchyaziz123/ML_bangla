export const cnn_3_famous_architectures = {
  title: "বিখ্যাত CNN আর্কিটেকচার: LeNet থেকে MobileNet — বিবর্তনের গল্প",
  description: "LeNet-5, AlexNet, VGG, ResNet, Inception ও MobileNet-এর architecture, key innovations ও parameter count। ResNet-এর skip connection কেন vanishing gradient সমাধান করে, PyTorch-এ pre-trained ResNet50 দিয়ে inference।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "cnn-famous-architectures",
  content: `
    <h3>১. CNN Architecture-এর বিবর্তন: একটি সংক্ষিপ্ত ইতিহাস</h3>
    <p>১৯৯৮ সাল থেকে ২০১৭ সাল পর্যন্ত CNN architecture-এ বিপ্লব ঘটেছে। প্রতিটি নতুন architecture একটি নির্দিষ্ট সমস্যা সমাধান করেছে।</p>
    <table>
      <thead>
        <tr><th>Architecture</th><th>Year</th><th>ImageNet Top-5 Acc</th><th>Parameters</th><th>Key Innovation</th></tr>
      </thead>
      <tbody>
        <tr><td>LeNet-5</td><td>1998</td><td>N/A (MNIST)</td><td>~60K</td><td>First practical CNN</td></tr>
        <tr><td>AlexNet</td><td>2012</td><td>84.6%</td><td>~60M</td><td>ReLU, Dropout, GPU</td></tr>
        <tr><td>VGG-16</td><td>2014</td><td>92.7%</td><td>138M</td><td>Very deep, 3×3 only</td></tr>
        <tr><td>GoogLeNet</td><td>2014</td><td>93.3%</td><td>~6.8M</td><td>Inception module</td></tr>
        <tr><td>ResNet-50</td><td>2015</td><td>96.4%</td><td>~25M</td><td>Skip connections</td></tr>
        <tr><td>MobileNetV2</td><td>2018</td><td>91.0%</td><td>~3.4M</td><td>Depthwise separable conv</td></tr>
      </tbody>
    </table>

    <h3>২. LeNet-5 ও AlexNet: CNN-এর জন্ম</h3>
    <p><strong>LeNet-5 (Yann LeCun, 1998):</strong> প্রথম কার্যকর CNN। মাত্র 5টি layer (2 Conv + 3 FC), tanh/sigmoid activation, average pooling। MNIST-এ 99%+ accuracy। ৬০K parameters।</p>
    <p><strong>AlexNet (Krizhevsky, 2012):</strong> ImageNet-এ বিপ্লব এনেছিল। 2012 ILSVRC competition-এ second place-এর চেয়ে 10.9% কম error। মূল উদ্ভাবন:</p>
    <ul>
      <li><strong>ReLU activation:</strong> tanh-এর চেয়ে 6x দ্রুত convergence</li>
      <li><strong>Dropout:</strong> Overfitting রোধে (p=0.5)</li>
      <li><strong>GPU training:</strong> দুটি GTX 580 তে parallel training</li>
      <li><strong>Data augmentation:</strong> Random crops, flips, color jitter</li>
      <li><strong>Local Response Normalization (LRN)</strong></li>
    </ul>
    <pre><code>import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """Original LeNet-5 architecture"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.tanh1 = nn.Tanh()                      # 32x32x1 → 28x28x6
        self.pool1 = nn.AvgPool2d(2, stride=2)       # → 14x14x6
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()                      # → 10x10x16
        self.pool2 = nn.AvgPool2d(2, stride=2)       # → 5x5x16
        self.flatten = nn.Flatten()                    # → 400
        self.fc1 = nn.Linear(400, 120)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.flatten(x)
        x = self.tanh3(self.fc1(x))
        x = self.tanh4(self.fc2(x))
        return self.fc3(x)

class AlexNet(nn.Module):
    """Simplified AlexNet (single GPU, no LRN)"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.relu1 = nn.ReLU()                       # → 55x55x96
        self.pool1 = nn.MaxPool2d(3, stride=2)        # → 27x27x96
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.relu2 = nn.ReLU()                       # → 27x27x256
        self.pool2 = nn.MaxPool2d(3, stride=2)        # → 13x13x256
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=2)        # → 6x6x256
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout1(self.relu6(self.fc1(x)))
        x = self.dropout2(self.relu7(self.fc2(x)))
        return self.fc3(x)

lenet = LeNet5()
lenet_params = sum(p.numel() for p in lenet.parameters())
print(f"LeNet-5 params: {lenet_params:,}")

alex = AlexNet()
alex_params = sum(p.numel() for p in alex.parameters())
print(f"AlexNet params: {alex_params:,}")
</code></pre>

    <h3>৩. VGG: Depth ও Simplicity</h3>
    <p><strong>VGG (Oxford, 2014):</strong> মূল idea — শুধু 3×3 conv filters ব্যবহার করো, কিন্তু network অনেক deep করো।</p>
    <p>কেন 3×3 filters? দুটি 3×3 conv = একটি 5×5 conv-এর receptive field, কিন্তু:</p>
    <ul>
      <li>Parameters: 2×(3×3×C×C) = 18C² বনাম 5×5×C×C = 25C² — ২৮% কম</li>
      <li>বেশি non-linearity (দুটি ReLU বনাম একটি)</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """VGG block: num_convs × Conv(3x3) → MaxPool"""
    def __init__(self, in_channels, num_convs, filters):
        super().__init__()
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        ch = in_channels
        for _ in range(num_convs):
            self.convs.append(nn.Conv2d(ch, filters, 3, padding=1))
            self.relus.append(nn.ReLU())
            ch = filters
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        for conv, relu in zip(self.convs, self.relus):
            x = relu(conv(x))
        return self.pool(x)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.block1 = VGGBlock(3,    2,  64)   # Block 1: → 112x112x64
        self.block2 = VGGBlock(64,   2, 128)   # Block 2: → 56x56x128
        self.block3 = VGGBlock(128,  3, 256)   # Block 3: → 28x28x256
        self.block4 = VGGBlock(256,  3, 512)   # Block 4: → 14x14x512
        self.block5 = VGGBlock(512,  3, 512)   # Block 5: → 7x7x512

        self.flatten = nn.Flatten()                    # 7*7*512 = 25088
        self.fc1 = nn.Linear(25088, 4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

model = VGG16()
total_params = sum(p.numel() for p in model.parameters())
print(f"VGG-16 params: {total_params:,}")
# ~138 million — mostly Linear layers-এ (FC layers 102M params নেয়!)
</code></pre>

    <h3>৪. ResNet: Skip Connection ও Vanishing Gradient সমাধান</h3>
    <p>VGG-এর পর deeper network তৈরি করলে accuracy <em>কমে যেত</em> — degradation problem। ResNet (He et al., 2015) এর সমাধান করে <strong>residual/skip connections</strong> দিয়ে।</p>
    <p><strong>Skip Connection:</strong> output = F(x) + x</p>
    <p>Network শুধু residual F(x) শেখে (x থেকে কতটুকু পরিবর্তন হবে), পুরো mapping নয়। যদি optimal function = identity হয়, তাহলে F(x) = 0 শিখলেই হয় — সহজ।</p>
    <p><strong>Gradient flow:</strong> d(Loss)/dx = d(Loss)/d(F(x)+x) = gradient × (1 + dF/dx) — gradient সরাসরি পৌঁছায়।</p>
    <p><strong>ResNet-50 Bottleneck Block:</strong> 1×1 → 3×3 → 1×1 (channels: C → C/4 → C/4 → C)</p>
    <pre><code>import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ResNet basic residual block (for ResNet-18/34)"""
    def __init__(self, in_channels, filters, stride=1):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.relu  = nn.ReLU()

        # Shortcut: dimension match করতে projection লাগতে পারে
        self.has_shortcut = stride != 1 or in_channels != filters
        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.shortcut_bn   = nn.BatchNorm2d(filters)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x)) if self.has_shortcut else x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # Skip connection: F(x) + x
        return self.relu(y + shortcut)

class BottleneckBlock(nn.Module):
    """ResNet-50 bottleneck block: 1x1 → 3x3 → 1x1"""
    def __init__(self, in_channels, filters, stride=1):
        super().__init__()
        expanded = filters * 4  # Bottleneck expansion

        # 1x1 conv: reduce channels
        self.conv1 = nn.Conv2d(in_channels, filters, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        # 3x3 conv
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        # 1x1 conv: expand channels back
        self.conv3 = nn.Conv2d(filters, expanded, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(expanded)
        self.relu  = nn.ReLU()

        # Projection shortcut
        self.has_shortcut = stride != 1 or in_channels != expanded
        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels, expanded, 1, stride=stride, bias=False)
            self.shortcut_bn   = nn.BatchNorm2d(expanded)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x)) if self.has_shortcut else x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + shortcut)

class ResNet18(nn.Module):
    """Small ResNet-18 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU()

        # 4 stages
        self.stages = nn.ModuleList()
        in_ch = 64
        for filters, stride in [(64,1), (128,2), (256,2), (512,2)]:
            self.stages.append(ResidualBlock(in_ch, filters, stride=stride))
            self.stages.append(ResidualBlock(filters, filters, stride=1))
            in_ch = filters

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

resnet18 = ResNet18()
total_params = sum(p.numel() for p in resnet18.parameters())
print(f"Custom ResNet-18 params: {total_params:,}")
</code></pre>

    <h3>৫. Inception Module ও GoogLeNet</h3>
    <p><strong>GoogLeNet/Inception (Google, 2014):</strong> মূল idea — আলাদা আলাদা kernel size (1×1, 3×3, 5×5) parallel-এ চালাও, তারপর concatenate করো। Network নিজেই শিখবে কোন scale-এ কী feature দরকার।</p>
    <p>1×1 conv-এর ভূমিকা: channel reduction (bottleneck) — computational cost কমায়।</p>
    <pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F

def inception_module(x, in_channels, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool):
    """
    Inception module with dimensionality reduction
    f_1x1: 1x1 branch filters
    f_3x3_reduce: 1x1 reduction before 3x3
    f_3x3: 3x3 branch filters
    f_5x5_reduce: 1x1 reduction before 5x5
    f_5x5: 5x5 branch filters
    f_pool: 1x1 after MaxPool branch
    """
    # Branch 1: 1x1 conv
    branch1 = F.relu(nn.Conv2d(in_channels, f_1x1, 1)(x))

    # Branch 2: 1x1 → 3x3
    branch2 = F.relu(nn.Conv2d(in_channels, f_3x3_reduce, 1)(x))
    branch2 = F.relu(nn.Conv2d(f_3x3_reduce, f_3x3, 3, padding=1)(branch2))

    # Branch 3: 1x1 → 5x5
    branch3 = F.relu(nn.Conv2d(in_channels, f_5x5_reduce, 1)(x))
    branch3 = F.relu(nn.Conv2d(f_5x5_reduce, f_5x5, 5, padding=2)(branch3))

    # Branch 4: MaxPool → 1x1
    branch4 = nn.MaxPool2d(3, stride=1, padding=1)(x)
    branch4 = F.relu(nn.Conv2d(in_channels, f_pool, 1)(branch4))

    # Concatenate all branches along channel axis
    output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
    return output

# উদাহরণ: Inception 3a module (GoogLeNet)
inputs = torch.randn(1, 192, 28, 28)
x = inception_module(inputs, 192, 64, 96, 128, 16, 32, 32)
print(f"Inception output shape: {tuple(x.shape)}")
# 64 + 128 + 32 + 32 = 256 channels
</code></pre>

    <h3>৬. MobileNet: Depthwise Separable Convolution</h3>
    <p><strong>MobileNet (Google, 2017):</strong> Mobile ও embedded device-এর জন্য lightweight CNN। মূল idea: standard convolution-কে দুটি step-এ ভাগ করো।</p>
    <p><strong>Depthwise Separable Convolution:</strong></p>
    <ul>
      <li><strong>Depthwise Conv:</strong> প্রতিটি input channel-এ আলাদা একটি filter (no cross-channel mixing)</li>
      <li><strong>Pointwise Conv (1×1):</strong> Channel mixing — linear combination of all channels</li>
    </ul>
    <p><strong>Parameter reduction:</strong> Standard: F²×C_in×C_out | Depthwise Sep: F²×C_in + C_in×C_out</p>
    <p>Reduction factor ≈ 1/C_out + 1/F² ≈ 8-9x কম parameters (3×3 kernel, 256 filters-এ)।</p>
    <pre><code>import torch
import torch.nn as nn

class DepthwiseSeparableBlock(nn.Module):
    """MobileNet-style depthwise separable conv block"""
    def __init__(self, in_channels, filters, stride=1):
        super().__init__()
        # Depthwise conv: প্রতি channel-এ আলাদা 3x3 filter
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        # Pointwise conv: 1x1, channel mixing
        self.pointwise = nn.Conv2d(in_channels, filters, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x

class MobileNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU()

        configs, in_ch = [(64,1), (128,2), (128,1), (256,2), (256,1)], 32
        self.blocks = nn.ModuleList()
        for filters, stride in configs:
            self.blocks.append(DepthwiseSeparableBlock(in_ch, filters, stride=stride))
            in_ch = filters

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

mob = MobileNetSmall()
mob_params = sum(p.numel() for p in mob.parameters())
print(f"MobileNet-small params: {mob_params:,}")

# Compare: standard CNN vs MobileNet-style
standard_params = sum([3*3*3*32, 3*3*32*64, 3*3*64*128, 3*3*128*256])
mobile_params   = sum([3*3*3 + 3*32, 3*3*32 + 32*64, 3*3*64 + 64*128, 3*3*128 + 128*256])
print(f"\nConv params (4 layers): {standard_params:,}")
print(f"DepthSep params (4 layers): {mobile_params:,}")
print(f"Reduction: {standard_params/mobile_params:.1f}x")
</code></pre>

    <h3>৭. Pre-trained ResNet50 দিয়ে Inference</h3>
    <p>PyTorch/torchvision-এ ImageNet-pre-trained ResNet50 সরাসরি ব্যবহার করা যায়।</p>
    <pre><code>import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Pre-trained ResNet50 load (ImageNet weights)
weights  = ResNet50_Weights.IMAGENET1K_V2
resnet   = resnet50(weights=weights)   # Classifier head সহ
resnet.eval()

params = sum(p.numel() for p in resnet.parameters())
print(f"ResNet50 params: {params:,}")

# একটি random image দিয়ে inference
dummy_image = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8)

# ResNet50 preprocessing
preprocess  = weights.transforms()
categories  = weights.meta["categories"]
preprocessed = preprocess(dummy_image)
with torch.no_grad():
    predictions = resnet(preprocessed)

# Top-5 predictions decode করি
top5 = torch.topk(predictions.softmax(dim=1), 5)
print("\nTop-5 Predictions:")
for rank, (idx, confidence) in enumerate(zip(top5.indices[0], top5.values[0]), 1):
    print(f"  {rank}. {categories[idx]}: {confidence:.4f}")

# Feature extractor হিসেবে ব্যবহার (transfer learning)
class ResNetFeatureExtractor(nn.Module):
    """resnet50-এর সব layer রাখে, শুধু শেষের fc classifier বাদ দিয়ে"""
    def __init__(self, base_model):
        super().__init__()
        self.conv1   = base_model.conv1
        self.bn1     = base_model.bn1
        self.relu    = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1  = base_model.layer1
        self.layer2  = base_model.layer2
        self.layer3  = base_model.layer3
        self.layer4  = base_model.layer4
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)

feature_extractor = ResNetFeatureExtractor(resnet)
with torch.no_grad():
    features = feature_extractor(preprocessed).flatten(1)   # 2048-dim features
print(f"\nFeature vector shape: {tuple(features.shape)}")  # (1, 2048)
</code></pre>
  `,
};
