export const cnn_1_architecture = {
  title: "CNN আর্কিটেকচারের গণিত: Convolution, Stride, Padding ও Parameter Count",
  description: "Convolution operation-এর গাণিতিক ভিত্তি, kernel/filter, feature map, stride ও padding-এর output size formula, parameter count গণনা, receptive field এবং কেন CNN image-এর জন্য আদর্শ — সম্পূর্ণ math ও কোড সহ।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "cnn-architecture-math",
  content: `
    <h3>১. Convolution Operation-এর গণিত</h3>
    <p>Convolution হলো দুটি function-এর মধ্যে একটি গাণিতিক অপারেশন। Image processing-এ আমরা 2D discrete convolution ব্যবহার করি। Input image <strong>I</strong> এবং kernel <strong>K</strong>-এর convolution:</p>
    <p><strong>(I ★ K)[i, j] = Σₘ Σₙ I[i+m, j+n] · K[m, n]</strong></p>
    <p>এখানে:</p>
    <ul>
      <li><strong>I[i+m, j+n]</strong> — input image-এর একটি patch</li>
      <li><strong>K[m, n]</strong> — kernel/filter-এর weight</li>
      <li><strong>i, j</strong> — output feature map-এর position</li>
      <li><strong>m, n</strong> — kernel-এর row ও column index</li>
    </ul>
    <p>একটি 5×5 input-এ 3×3 kernel apply করলে প্রতিটি output pixel হয় input-এর 9টি pixel-এর weighted sum। Kernel-এর প্রতিটি weight আলাদা pattern detect করতে শেখে — যেমন horizontal edge, vertical edge, corner ইত্যাদি।</p>
    <pre><code>import numpy as np

# Manual 2D convolution (cross-correlation, যা CNN-এ ব্যবহার হয়)
def manual_conv2d(image, kernel, stride=1, padding=0):
    """
    image:  (H, W) numpy array
    kernel: (kH, kW) numpy array
    return: feature map
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # Padding যোগ করা
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
        H, W = image.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    feature_map = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map

# উদাহরণ: Vertical edge detection kernel
image = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0]
], dtype=float)

vertical_edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

output = manual_conv2d(image, vertical_edge_kernel)
print("Feature Map (Vertical Edge):")
print(output)
# Output: vertical edge টা bright হবে যেখানে transition আছে
</code></pre>

    <h3>২. Kernel, Feature Map এবং Multiple Filters</h3>
    <p><strong>Kernel/Filter</strong> হলো একটি ছোট weight matrix (যেমন 3×3, 5×5) যা input-এর উপর slide করে। প্রতিটি kernel একটি নির্দিষ্ট pattern শেখে।</p>
    <p><strong>Feature Map</strong> হলো একটি kernel-এর output — input-এ সেই pattern কোথায় কতটুকু আছে তার spatial map।</p>
    <p>একটি Conv layer-এ <em>C_out</em> সংখ্যক kernel থাকে → <em>C_out</em> সংখ্যক feature map তৈরি হয়। এগুলো stack করলে output tensor-এর depth (channel dimension) হয় C_out।</p>
    <table>
      <thead>
        <tr><th>Kernel Type</th><th>Pattern Detected</th><th>উদাহরণ Use</th></tr>
      </thead>
      <tbody>
        <tr><td>Horizontal Edge [[-1,-1,-1],[0,0,0],[1,1,1]]</td><td>Horizontal edges</td><td>Layer 1</td></tr>
        <tr><td>Vertical Edge [[-1,0,1],[-1,0,1],[-1,0,1]]</td><td>Vertical edges</td><td>Layer 1</td></tr>
        <tr><td>Laplacian [[0,1,0],[1,-4,1],[0,1,0]]</td><td>Blobs/corners</td><td>Layer 1-2</td></tr>
        <tr><td>Learned (deep layer)</td><td>Complex textures, parts</td><td>Layer 3+</td></tr>
      </tbody>
    </table>
    <pre><code>import torch
import torch.nn as nn

# RGB image-এ multiple filter: C_in=3 (RGB), C_out=32 filters
class ModelCheck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        # Output shape: (28, 28, 32) — 32টি feature map
        x = self.relu2(self.conv2(x))
        # Output shape: (28, 28, 64) — 64টি feature map
        return x

model_check = ModelCheck()
print(model_check)
total_params = sum(p.numel() for p in model_check.parameters())
print(f"Total params: {total_params:,}")
# দেখবেন প্রতিটি layer-এর parameter count input channel অনুযায়ী বাড়ছে
</code></pre>

    <h3>৩. Stride এবং Output Size Formula</h3>
    <p><strong>Stride (S)</strong> নির্ধারণ করে kernel কতটুকু পদক্ষেপে slide করবে। Stride=1 মানে প্রতিটি position-এ একটু সরে, Stride=2 মানে দুটো করে সরে।</p>
    <p>Output size formula:</p>
    <p><strong>Output Size = ⌊(W - F + 2P) / S⌋ + 1</strong></p>
    <p>এখানে:</p>
    <ul>
      <li><strong>W</strong> = input width (বা height)</li>
      <li><strong>F</strong> = filter size (kernel_size)</li>
      <li><strong>P</strong> = padding amount</li>
      <li><strong>S</strong> = stride</li>
    </ul>
    <p>উদাহরণ: W=28, F=3, P=0, S=1 → (28-3+0)/1 + 1 = <strong>26</strong></p>
    <p>উদাহরণ: W=28, F=3, P=1, S=1 → (28-3+2)/1 + 1 = <strong>28</strong> (same padding)</p>
    <pre><code>def output_size(W, F, P, S):
    """Output size formula for Conv/Pool layers"""
    return (W - F + 2*P) // S + 1

# বিভিন্ন configuration test
configs = [
    (28, 3, 0, 1, "28x28, 3x3 kernel, no pad, stride 1"),
    (28, 3, 1, 1, "28x28, 3x3 kernel, same pad, stride 1"),
    (28, 3, 0, 2, "28x28, 3x3 kernel, no pad, stride 2"),
    (32, 5, 2, 1, "32x32, 5x5 kernel, same pad, stride 1"),
    (224, 11, 0, 4, "AlexNet first layer: 224x224, 11x11, stride 4"),
]

for W, F, P, S, desc in configs:
    out = output_size(W, F, P, S)
    print(f"{desc}")
    print(f"  Output size: {out}x{out}\n")
</code></pre>

    <h3>৪. Padding: Same vs Valid</h3>
    <p><strong>Valid Padding (P=0)</strong>: কোনো padding নেই। Output size input-এর চেয়ে ছোট হয়। Border pixels কম গুরুত্ব পায়।</p>
    <p><strong>Same Padding</strong>: Output size = Input size (stride=1 হলে)। Input-এর চারদিকে zeros যোগ করা হয়। P = ⌊F/2⌋।</p>
    <table>
      <thead>
        <tr><th>Padding</th><th>Formula (P)</th><th>Output Size (S=1)</th><th>কখন ব্যবহার</th></tr>
      </thead>
      <tbody>
        <tr><td>valid</td><td>P = 0</td><td>W - F + 1</td><td>Size কমাতে চাইলে</td></tr>
        <tr><td>same</td><td>P = ⌊F/2⌋</td><td>W (same as input)</td><td>Size maintain করতে</td></tr>
        <tr><td>causal</td><td>Left only</td><td>W (1D sequences)</td><td>Time-series, NLP</td></tr>
      </tbody>
    </table>
    <pre><code>import torch
import torch.nn as nn

# Padding comparison
x = torch.randn(1, 1, 28, 28)

conv_valid = nn.Conv2d(1, 32, kernel_size=3, padding=0)   # 'valid'
conv_same  = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 'same' (for 3x3, stride 1)

out_valid = conv_valid(x)
out_same  = conv_same(x)

print(f"Input shape:        {tuple(x.shape)}")
print(f"Valid padding out:  {tuple(out_valid.shape)}")   # (1, 32, 26, 26)
print(f"Same  padding out:  {tuple(out_same.shape)}")    # (1, 32, 28, 28)
</code></pre>

    <h3>৫. Parameter Count: Conv Layer-এ কতটি Weight?</h3>
    <p>একটি Conv2D layer-এর মোট parameter:</p>
    <p><strong>Params = F × F × C_in × C_out + C_out</strong></p>
    <p>এখানে প্রথম অংশ weights, শেষ C_out হলো biases (প্রতি filter-এ একটি bias)।</p>
    <p>উদাহরণ: 3×3 kernel, 3 input channels (RGB), 64 filters:</p>
    <p>Params = 3 × 3 × 3 × 64 + 64 = 1728 + 64 = <strong>1,792</strong></p>
    <p>তুলনায়: একটি Dense layer (150,528 → 64) এর params = 150,528 × 64 + 64 = <strong>9,633,856</strong> — ৫,০০০ গুণ বেশি!</p>
    <pre><code>def conv_params(F, C_in, C_out, bias=True):
    """Conv2D layer-এর parameter count"""
    weights = F * F * C_in * C_out
    biases  = C_out if bias else 0
    return weights + biases

# VGG-16 প্রথম কয়েক layer-এর parameter count
layers_config = [
    (3, 3,   64, "Conv1_1: 3x3, RGB→64"),
    (3, 64,  64, "Conv1_2: 3x3, 64→64"),
    (3, 64, 128, "Conv2_1: 3x3, 64→128"),
    (3, 128,128, "Conv2_2: 3x3, 128→128"),
]

total = 0
for F, C_in, C_out, name in layers_config:
    p = conv_params(F, C_in, C_out)
    total += p
    print(f"{name}: {p:,} params")

print(f"\nTotal (first 4 conv layers): {total:,}")
</code></pre>

    <h3>৬. Receptive Field: একটি Neuron কতটুকু দেখে?</h3>
    <p><strong>Receptive Field</strong> হলো একটি output neuron-এর সাথে সংযুক্ত input pixels-এর area। Deep network-এ deep layer-এর neurons অনেক বড় area দেখে।</p>
    <p>Receptive field formula (stacked layers):</p>
    <p><strong>RF_l = RF_{l-1} + (F_l - 1) × ∏_{k=1}^{l-1} S_k</strong></p>
    <p>উদাহরণ: দুটি 3×3 conv (stride=1) → RF = 1 + 2 + 2 = <strong>5×5</strong></p>
    <p>তিনটি 3×3 conv → RF = <strong>7×7</strong></p>
    <p>এই কারণে VGG-16 ছোট (3×3) kernel ব্যবহার করে — ৩টি 3×3 conv = ১টি 7×7 conv-এর receptive field, কিন্তু কম parameters ও বেশি non-linearity।</p>
    <pre><code>def receptive_field(layers_config):
    """
    layers_config: list of (kernel_size, stride) tuples
    Returns: receptive field size
    """
    rf = 1
    stride_product = 1
    for F, S in layers_config:
        rf += (F - 1) * stride_product
        stride_product *= S
    return rf

# তিনটি 3x3 conv, stride=1
rf_3x3x3 = receptive_field([(3,1), (3,1), (3,1)])
print(f"3x 3x3 conv RF: {rf_3x3x3}x{rf_3x3x3}")  # 7x7

# একটি 7x7 conv, stride=1
rf_7x7 = receptive_field([(7,1)])
print(f"1x 7x7 conv RF: {rf_7x7}x{rf_7x7}")       # 7x7

# AlexNet style: 11x11 stride=4, 5x5 stride=2
rf_alexnet = receptive_field([(11,4), (5,2)])
print(f"AlexNet (2 layers) RF: {rf_alexnet}x{rf_alexnet}")
</code></pre>

    <h3>৭. Weight Sharing ও Translation Invariance</h3>
    <p>CNN-এর তিনটি key property image-এর জন্য আদর্শ করে তোলে:</p>
    <ul>
      <li><strong>Weight Sharing:</strong> একই kernel পুরো image-এ slide করে — same weights সর্বত্র। এটি parameter count নাটকীয়ভাবে কমায় এবং overfitting রোধ করে।</li>
      <li><strong>Translation Invariance:</strong> বিড়ালটি ছবির বাম বা ডানে থাকলেও একই kernel সেই feature detect করবে। Model position-independent pattern শেখে।</li>
      <li><strong>Local Connectivity:</strong> প্রতিটি neuron শুধু local neighborhood দেখে, পুরো image নয়। Image-এ local structure (edge, corner, texture) থাকে।</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn

# Full CNN architecture দেখি — shapes ট্র্যাক করে
class CNNWithShapes(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()      # 32x32x32
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2 = nn.ReLU()      # 32x32x32
        self.pool1 = nn.MaxPool2d(2)   # 16x16x32

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()      # 16x16x64
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu4 = nn.ReLU()      # 16x16x64
        self.pool2 = nn.MaxPool2d(2)   # 8x8x64

        # Classifier head
        self.flatten = nn.Flatten()   # 8*8*64 = 4096
        self.fc1 = nn.Linear(4096, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        return self.fc2(x)

model = CNNWithShapes()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")

# Parameter count manually verify করি:
# Conv1: 3*3*3*32 + 32 = 896
# Conv2: 3*3*32*32 + 32 = 9,248
# Conv3: 3*3*32*64 + 64 = 18,496
# Conv4: 3*3*64*64 + 64 = 36,928
# Linear1: 4096*256 + 256 = 1,048,832
# Linear2: 256*10 + 10 = 2,570
# Total: ~1,116,970
</code></pre>
  `,
};
