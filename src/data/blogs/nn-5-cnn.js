export const nn_5_cnn = {
  title: "CNN দিয়ে Image Classification",
  description: "Convolutional Neural Network-এর অন্তর্নিহিত ধারণা, Conv2D ও MaxPooling, MNIST এবং CIFAR-10-এ MLP বনাম CNN তুলনা এবং Transfer Learning-এর পরিচয়।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "nn-cnn-image-classification",
  content: `
    <h3>১. CNN কেন? MLP-এর সীমাবদ্ধতা</h3>
    <p>MLP দিয়ে image classification করা যায় কিন্তু কয়েকটি বড় সমস্যা আছে:</p>
    <ul>
      <li><strong>Parameters বেশি:</strong> 224×224 RGB image → 150,528 inputs। একটি hidden layer (1000 neurons) → 150 million parameters!</li>
      <li><strong>Spatial information হারায়:</strong> Flatten করলে pixel-এর position relationship নষ্ট হয়</li>
      <li><strong>Translation invariant নয়:</strong> ছবির বাম-ডানে বিড়াল থাকলে আলাদা feature হিসেবে দেখে</li>
    </ul>
    <p>CNN এই সমস্যাগুলো সমাধান করে local pattern sharing এবং spatial hierarchy দিয়ে।</p>
    <pre><code>import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# CNN intuition: image -> edges -> shapes -> parts -> objects
# Layer 1 (Conv): simple edges (horizontal, vertical, diagonal)
# Layer 2 (Conv): corners, curves combinations
# Layer 3 (Conv): eyes, wheels, textures
# Final layers: "dog", "car", "face"
</code></pre>

    <h3>২. Convolution Operation: কীভাবে কাজ করে</h3>
    <p>একটি filter/kernel (যেমন 3×3 matrix) image-এর উপর sliding window হিসেবে চলে। প্রতিটি position-এ element-wise multiplication এবং sum — এটাই convolution।</p>
    <table>
      <thead><tr><th>Parameter</th><th>অর্থ</th><th>সাধারণ মান</th><th>প্রভাব</th></tr></thead>
      <tbody>
        <tr><td>filters</td><td>কতটি feature map</td><td>32, 64, 128</td><td>বেশি = বেশি feature শেখে</td></tr>
        <tr><td>kernel_size</td><td>Filter আকার</td><td>(3,3), (5,5)</td><td>বড় = larger receptive field</td></tr>
        <tr><td>padding</td><td>'valid' বা 'same'</td><td>'same'</td><td>same = output size same রাখে</td></tr>
        <tr><td>strides</td><td>Filter কতটা সরবে</td><td>(1,1)</td><td>বড় stride = smaller output</td></tr>
        <tr><td>activation</td><td>Non-linearity</td><td>'relu'</td><td>Negative values মুছে</td></tr>
      </tbody>
    </table>
    <pre><code"># Manual convolution visualization
import numpy as np

# 5x5 image (grayscale)
image = np.array([
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,1,1,0],
    [0,1,1,0,0]
], dtype=float)

# Edge detection filter (Sobel-like)
vertical_edge_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Manual convolution
def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[i,j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

feature_map = convolve2d(image, vertical_edge_filter)
print("Feature map shape:", feature_map.shape)  # (3, 3)
print("Feature map:\n", feature_map)
</code></pre>

    <h3>৩. MaxPooling: Spatial Downsampling</h3>
    <p>MaxPooling প্রতিটি window থেকে maximum value নেয়। এটি:</p>
    <ul>
      <li>Feature map-এর size কমায় (computational cost কমে)</li>
      <li>Translation invariance দেয়: object একটু সরে গেলেও same feature</li>
      <li>Dominant feature retain করে</li>
    </ul>
    <pre><code># MaxPooling example
pool_input = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
])

# 2x2 MaxPooling with stride 2
# Top-left: max(1,3,5,6) = 6
# Top-right: max(2,4,7,8) = 8
# Bottom-left: max(3,2,1,2) = 3
# Bottom-right: max(1,0,3,4) = 4
print("After MaxPooling:")
print(np.array([[6,8],[3,4]]))
# Shape: 4x4 -> 2x2 (halved!)
</code></pre>

    <h3>৪. CNN Architecture: MNIST Classification</h3>
    <pre><code># MNIST dataset (28x28 grayscale) — torchvision দিয়ে load
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])  # 0-255 -> 0-1, shape (1,28,28)
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

images, _ = next(iter(train_loader))
print("CNN input shape:", images.shape)  # (128, 1, 28, 28)

# CNN model — nn.Module class
class CNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool  = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)   # 7*7*64 = 3136
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))    # 14x14 -> 7x7
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)                         # raw logits

cnn_model = CNNMnist()
print(cnn_model)
# Conv2d(32): 32 filters, 3x3 kernel, 320 params
# Conv2d(64): 64 filters, 18496 params
# Linear(128): 3136*128 + 128 = 401536 params
# Total: ~422K params (MLP-এর 109K-এর চেয়ে বেশি, কিন্তু accuracy অনেক ভালো!)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
</code></pre>

    <h3>৫. Training ও MLP vs CNN তুলনা</h3>
    <pre><code>device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model.to(device)

for epoch in range(10):
    cnn_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn_model(xb), yb)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total

cnn_acc = evaluate(cnn_model, test_loader)
print(f"CNN Test Accuracy: {cnn_acc:.4f}")  # ~0.993

# MLP তুলনা (একই data, নিজেই flatten করে nn.Flatten)
class MLPMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

mlp_model = MLPMnist().to(device)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

for epoch in range(10):
    mlp_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        mlp_optimizer.zero_grad()
        loss = criterion(mlp_model(xb), yb)
        loss.backward()
        mlp_optimizer.step()

mlp_acc = evaluate(mlp_model, test_loader)
print(f"MLP Test Accuracy: {mlp_acc:.4f}")  # ~0.980

print(f"\nCNN improvement: {(cnn_acc - mlp_acc)*100:.2f}%")
</code></pre>
    <table>
      <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>MNIST Accuracy</th></tr></thead>
      <tbody>
        <tr><td>Simple MLP</td><td>784→128→64→10</td><td>~109K</td><td>~97.8%</td></tr>
        <tr><td>Deep MLP</td><td>784→256→128→64→10</td><td>~236K</td><td>~98.2%</td></tr>
        <tr><td>Simple CNN</td><td>Conv32→Pool→Conv64→Pool→128→10</td><td>~422K</td><td>~99.1%</td></tr>
        <tr><td>Deep CNN</td><td>2x(Conv→Pool)→Conv→128→10</td><td>~750K</td><td>~99.4%</td></tr>
      </tbody>
    </table>

    <h3>৬. CIFAR-10 ও Transfer Learning পরিচয়</h3>
    <pre><code># CIFAR-10: 32x32 RGB, 10 classes
from torchvision import datasets as tv_datasets, transforms as tv_transforms
import torchvision.models as models

cifar_transform = tv_transforms.Compose([tv_transforms.ToTensor()])
cifar_train = tv_datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# CIFAR-10-এর জন্য CNN (আরও complex)
class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

cifar_cnn = CifarCNN()
cifar_criterion = nn.CrossEntropyLoss()
cifar_optimizer = torch.optim.Adam(cifar_cnn.parameters(), lr=0.001)

# Transfer Learning (সংক্ষিপ্ত পরিচয়):
# ImageNet pre-trained model শুরু থেকে train না করে fine-tune করা
base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
base_model = base_model.features        # convolutional অংশ শুধু
for p in base_model.parameters():
    p.requires_grad = False             # Freeze pre-trained weights

class TransferModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)   # VGG16 features, 32x32 input -> (512, 1, 1)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

transfer_model = TransferModel(base_model)
print("Transfer learning model: VGG16 base + custom head")
print("Pre-trained on 1.2M ImageNet images!")
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম:</p>
    <ul>
      <li>CNN-এর intuition: local patterns → hierarchical features</li>
      <li>nn.Conv2d: filters, feature maps, padding, stride</li>
      <li>nn.MaxPool2d: spatial downsampling, translation invariance</li>
      <li>MNIST CNN: 99%+ accuracy (MLP ~98%)</li>
      <li>Transfer Learning: torchvision.models-এর ImageNet pre-trained VGG16, ResNet, EfficientNet</li>
    </ul>
    <p>পরবর্তী ব্লগে Regularization techniques — Dropout, Batch Normalization, L1/L2 — দিয়ে overfitting কমানো শিখব।</p>
  `,
};
