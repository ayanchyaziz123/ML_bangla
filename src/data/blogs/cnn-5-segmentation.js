export const cnn_5_segmentation = {
  title: "Semantic Segmentation: U-Net থেকে Mask R-CNN — Pixel-Level Prediction",
  description: "FCN, Transposed Convolution, U-Net-এর encoder-decoder architecture ও skip connections, PyTorch-এ binary segmentation U-Net implementation, Mask R-CNN এবং mIoU metric — pixel-by-pixel classification-এর সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "cnn-semantic-segmentation",
  content: `
    <h3>১. Segmentation-এর বিভিন্ন ধরন</h3>
    <p>Segmentation task-এ প্রতিটি pixel-কে classify করতে হয়। তিনটি প্রধান ধরন:</p>
    <table>
      <thead>
        <tr><th>Type</th><th>Output</th><th>Instance আলাদা?</th><th>Example Application</th></tr>
      </thead>
      <tbody>
        <tr><td>Semantic Segmentation</td><td>Per-pixel class (same class same color)</td><td>না</td><td>Road vs Sky vs Car label</td></tr>
        <tr><td>Instance Segmentation</td><td>Per-object mask + class</td><td>হ্যাঁ</td><td>Car-1 mask, Car-2 mask আলাদা</td></tr>
        <tr><td>Panoptic Segmentation</td><td>Semantic + Instance একসাথে</td><td>হ্যাঁ</td><td>Self-driving full scene understanding</td></tr>
      </tbody>
    </table>
    <p>Medical imaging (tumor detection), autonomous driving (road parsing), satellite imagery (land cover) — এই ক্ষেত্রগুলোয় segmentation অত্যন্ত গুরুত্বপূর্ণ।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# Segmentation output format visualization
H, W = 256, 256
num_classes = 3  # Background, Person, Car

# Prediction: (H, W, num_classes) — softmax probabilities
pred_logits = np.random.randn(H, W, num_classes)

# Softmax
exp = np.exp(pred_logits - pred_logits.max(axis=-1, keepdims=True))
pred_probs = exp / exp.sum(axis=-1, keepdims=True)

# Argmax → class map: (H, W)
pred_class_map = np.argmax(pred_probs, axis=-1)

print(f"Prediction probs shape: {pred_probs.shape}")
print(f"Class map shape:        {pred_class_map.shape}")
print(f"Unique classes predicted: {np.unique(pred_class_map)}")

# Per-pixel accuracy
def pixel_accuracy(pred_map, true_map):
    return np.mean(pred_map == true_map)

# Binary segmentation (foreground vs background)
true_binary = np.random.randint(0, 2, (H, W))
pred_binary = np.random.randint(0, 2, (H, W))
print(f"Pixel accuracy: {pixel_accuracy(pred_binary, true_binary):.4f}")
</code></pre>

    <h3>২. FCN: Fully Convolutional Network</h3>
    <p><strong>FCN (Long et al., 2015):</strong> Classification CNN-এর Dense layer-গুলো 1×1 Conv দিয়ে replace করো → output হয় spatial prediction map।</p>
    <p>সমস্যা: MaxPooling-এর কারণে spatial resolution কমে যায়। Solution: bilinear upsampling দিয়ে original resolution restore।</p>
    <p><strong>FCN-32s, FCN-16s, FCN-8s:</strong> কতটা upsample করা হচ্ছে তার উপর নির্ভর করে coarse/fine prediction।</p>
    <pre><code>import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    Simplified FCN-like architecture
    Dense/Linear layers replace করা হয়েছে Conv2d দিয়ে
    """
    def __init__(self, num_classes=21):
        super().__init__()
        # Encoder (VGG-like)
        self.enc1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc1_relu1 = nn.ReLU()
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc1_relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)   # /2

        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_relu1 = nn.ReLU()
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc2_relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)   # /4

        self.enc3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc3_relu1 = nn.ReLU()
        self.enc3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc3_relu2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)   # /8

        # "Fully connected" layers as 1x1 convolutions
        self.fc_conv1 = nn.Conv2d(256, 512, 1)
        self.fc_relu1 = nn.ReLU()
        self.fc_drop1 = nn.Dropout(0.5)
        self.fc_conv2 = nn.Conv2d(512, 512, 1)
        self.fc_relu2 = nn.ReLU()
        self.fc_drop2 = nn.Dropout(0.5)

        # Score map
        self.score = nn.Conv2d(512, num_classes, 1)   # (H/8, W/8, num_classes)

        # Upsample 8x back to original size
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.enc1_relu1(self.enc1_conv1(x))
        x = self.enc1_relu2(self.enc1_conv2(x))
        x = self.pool1(x)

        x = self.enc2_relu1(self.enc2_conv1(x))
        x = self.enc2_relu2(self.enc2_conv2(x))
        x = self.pool2(x)

        x = self.enc3_relu1(self.enc3_conv1(x))
        x = self.enc3_relu2(self.enc3_conv2(x))
        x = self.pool3(x)

        x = self.fc_drop1(self.fc_relu1(self.fc_conv1(x)))
        x = self.fc_drop2(self.fc_relu2(self.fc_conv2(x)))

        x = self.score(x)
        x = self.upsample(x)
        # Final activation
        return torch.softmax(x, dim=1)

fcn = FCN(num_classes=21)
total_params = sum(p.numel() for p in fcn.parameters())
print(f"FCN params: {total_params:,}")
test_input = torch.randn(1, 3, 512, 512)
output = fcn(test_input)
print(f"Input:  {tuple(test_input.shape)}")
print(f"Output: {tuple(output.shape)}")  # (1, 21, 512, 512)
</code></pre>

    <h3>৩. Transposed Convolution: Learnable Upsampling</h3>
    <p><strong>Transposed Convolution (Deconvolution):</strong> Regular conv-এর বিপরীত — ছোট feature map থেকে বড় output। Learnable weights দিয়ে upsampling।</p>
    <p>Stride=2 transposed conv → output size ≈ 2× input size।</p>
    <p>Bilinear upsampling (fixed) vs Transposed Conv (learnable): সাধারণত transposed conv ভালো result দেয় segmentation-এ।</p>
    <pre><code>import torch
import torch.nn as nn

# Bilinear upsampling vs Transposed Conv comparison
x_small = torch.randn(1, 64, 8, 8)

# Fixed bilinear upsampling
bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(x_small)

# Learnable transposed convolution
transposed_conv = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
)(x_small)

print(f"Input shape:         {tuple(x_small.shape)}")
print(f"Bilinear output:     {tuple(bilinear_up.shape)}")       # (1, 64, 16, 16)
print(f"TransposedConv out:  {tuple(transposed_conv.shape)}")   # (1, 64, 16, 16)

# Transposed Conv params
params = 2 * 2 * 64 * 64 + 64  # kernel + bias
print(f"\nTransposed Conv params: {params:,}")
print(f"Bilinear params: 0 (no learning)")

# U-Net-এ ConvTranspose2d + Concatenation pattern
def upsample_block(x, skip_connection, in_channels, filters):
    """U-Net decoder block"""
    # Upsample
    x = nn.ConvTranspose2d(in_channels, filters, 2, stride=2)(x)
    # Concatenate with skip connection from encoder
    x = torch.cat([x, skip_connection], dim=1)
    # Two conv layers
    x = torch.relu(nn.Conv2d(filters * 2, filters, 3, padding=1)(x))
    x = torch.relu(nn.Conv2d(filters, filters, 3, padding=1)(x))
    return x

print("\nUpsample block created successfully")
</code></pre>

    <h3>৪. U-Net: Encoder-Decoder with Skip Connections</h3>
    <p><strong>U-Net (Ronneberger et al., 2015):</strong> Medical image segmentation-এর জন্য তৈরি, এখন সব ধরনের segmentation-এ ব্যবহার হয়।</p>
    <p>Architecture দুই ভাগ:</p>
    <ul>
      <li><strong>Encoder (Contracting Path):</strong> Conv → Conv → MaxPool — spatial size কমে, channels বাড়ে। Features capture করে।</li>
      <li><strong>Decoder (Expanding Path):</strong> UpConv → Concatenate (encoder skip) → Conv → Conv — spatial size বাড়ে, channels কমে।</li>
    </ul>
    <p><strong>Skip Connections:</strong> Encoder-এর feature map সরাসরি corresponding decoder layer-এ concatenate হয়। Fine-grained spatial information preserve করে।</p>
    <pre><code>import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    """U-Net encoder block: Conv → Conv → (Dropout) → MaxPool"""
    def __init__(self, in_channels, filters, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        skip = x                 # Skip connection save
        x = self.pool(skip)
        return x, skip                        # Return both

class DecoderBlock(nn.Module):
    """U-Net decoder block: TransConv → Concat(skip) → Conv → Conv"""
    def __init__(self, in_channels, skip_channels, filters):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, filters, 2, stride=2)
        self.conv1 = nn.Conv2d(filters + skip_channels, filters, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)       # Skip connection merge
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

class UNet(nn.Module):
    """
    U-Net for binary segmentation (num_classes=1)
    বা multi-class (num_classes > 1)
    """
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        # Encoder (Contracting Path)
        self.enc1 = EncoderBlock(in_channels, 64)              # 256→128, skip1: (64,256,256)
        self.enc2 = EncoderBlock(64,        128)               # 128→64,  skip2: (128,128,128)
        self.enc3 = EncoderBlock(128,       256, dropout_rate=0.3)  # 64→32, skip3
        self.enc4 = EncoderBlock(256,       512, dropout_rate=0.3)  # 32→16, skip4

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bottleneck_relu1 = nn.ReLU()
        self.bottleneck_dropout = nn.Dropout(0.3)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bottleneck_relu2 = nn.ReLU()

        # Decoder (Expanding Path)
        self.dec4 = DecoderBlock(1024, 512, 512)               # 16→32
        self.dec3 = DecoderBlock(512,  256, 256)               # 32→64
        self.dec2 = DecoderBlock(256,  128, 128)               # 64→128
        self.dec1 = DecoderBlock(128,   64,  64)               # 128→256

        # Output layer
        self.out_conv = nn.Conv2d(64, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck_relu1(self.bottleneck_conv1(x))
        x = self.bottleneck_dropout(x)
        x = self.bottleneck_relu2(self.bottleneck_conv2(x))

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        out = self.out_conv(x)
        return torch.sigmoid(out) if self.num_classes == 1 else torch.softmax(out, dim=1)

# Binary segmentation model (1-channel output)
unet = UNet(in_channels=1, num_classes=1)
print(unet)

test = torch.randn(2, 1, 256, 256)
out  = unet(test)
print(f"\nU-Net Input:  {tuple(test.shape)}")
print(f"U-Net Output: {tuple(out.shape)}")   # (2, 1, 256, 256)
</code></pre>

    <h3>৫. U-Net Training: Binary Segmentation</h3>
    <pre><code>import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient: 2|A∩B| / (|A|+|B|)"""
    y_true_f = y_true.reshape(-1).float()
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """Combined BCE + Dice loss — segmentation-এ popular"""
    bce  = nn.functional.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def iou_score(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """IoU (Jaccard index) metric"""
    y_pred_bin = (y_pred > threshold).float()
    y_true_f   = y_true.reshape(-1).float()
    y_pred_f   = y_pred_bin.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Synthetic training data (real project-এ actual masks দরকার)
def generate_synthetic_data(n=100, img_size=256):
    """Simple synthetic segmentation data: circle in image"""
    images = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    masks  = np.zeros((n, 1, img_size, img_size), dtype=np.float32)

    for i in range(n):
        # Random circle
        cx = np.random.randint(64, 192)
        cy = np.random.randint(64, 192)
        r  = np.random.randint(20, 60)

        # Image: random noise + circle brighter
        images[i, 0] = np.random.randn(img_size, img_size) * 0.1
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        circle_mask = (x_grid - cx)**2 + (y_grid - cy)**2 &lt;= r**2
        images[i, 0][circle_mask] += 1.0
        masks[i, 0][circle_mask]   = 1.0

    return torch.from_numpy(images), torch.from_numpy(masks)

x_train, y_train = generate_synthetic_data(200, img_size=128)
x_val,   y_val   = generate_synthetic_data(50,  img_size=128)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)

# Build smaller U-Net for demo
unet_small = UNet(in_channels=1, num_classes=1)
optimizer  = torch.optim.Adam(unet_small.parameters(), lr=1e-4)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_iou, epochs_no_improve, best_state = 0.0, 0, None
patience = 15   # early stopping patience

for epoch in range(30):
    unet_small.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = bce_dice_loss(yb, unet_small(xb))
        loss.backward()
        optimizer.step()

    unet_small.eval()
    with torch.no_grad():
        val_pred = unet_small(x_val)
        val_iou  = iou_score(y_val, val_pred).item()
    scheduler.step(val_iou)

    if val_iou > best_iou:
        best_iou, best_state, epochs_no_improve = val_iou, unet_small.state_dict(), 0
        torch.save(best_state, 'best_unet.pt')   # checkpoint the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:   # early stopping
            break

unet_small.load_state_dict(best_state)   # restore best weights

# Predict
sample = x_val[:1]
unet_small.eval()
with torch.no_grad():
    pred = unet_small(sample)
pred_binary = (pred > 0.5).float()
print(f"Prediction shape: {tuple(pred.shape)}")
print(f"Max pred value: {pred.max().item():.4f}")
print(f"Dice score: {dice_coefficient(y_val[:1], pred).item():.4f}")
</code></pre>

    <h3>৬. Segmentation Metrics: Pixel Accuracy, IoU ও mIoU</h3>
    <p><strong>Pixel Accuracy:</strong> সঠিকভাবে classified pixel-এর শতাংশ। সমস্যা: class imbalance-এ misleading (90% background → 90% accuracy সহজেই)।</p>
    <p><strong>IoU (Jaccard Index) per class:</strong> IoU_c = |TP_c| / (|TP_c| + |FP_c| + |FN_c|)</p>
    <p><strong>mIoU (mean IoU):</strong> সব class-এর IoU-এর mean। Standard metric।</p>
    <pre><code>import numpy as np

def compute_segmentation_metrics(pred_map, true_map, num_classes):
    """
    pred_map, true_map: (H, W) integer class maps
    Returns: pixel_acc, per_class_iou, mIoU
    """
    # Pixel accuracy
    pixel_acc = np.mean(pred_map == true_map)

    ious = []
    for cls in range(num_classes):
        pred_c = (pred_map == cls)
        true_c = (true_map == cls)

        intersection = np.logical_and(pred_c, true_c).sum()
        union        = np.logical_or(pred_c,  true_c).sum()

        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = intersection / union
        ious.append(iou)

    # mIoU: present class-গুলোর mean
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mIoU = np.mean(valid_ious)

    return pixel_acc, ious, mIoU

# Test
H, W = 256, 256
num_classes = 4
true_map = np.random.randint(0, num_classes, (H, W))
pred_map = true_map.copy()
# 20% noise
noise_mask = np.random.rand(H, W) &lt; 0.2
pred_map[noise_mask] = np.random.randint(0, num_classes, noise_mask.sum())

pixel_acc, per_class_iou, mIoU = compute_segmentation_metrics(pred_map, true_map, num_classes)
print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Per-class IoU:  {[f'{iou:.3f}' for iou in per_class_iou]}")
print(f"mIoU:           {mIoU:.4f}")
</code></pre>

    <h3>৭. Mask R-CNN: Instance Segmentation</h3>
    <p><strong>Mask R-CNN (He et al., 2017):</strong> Faster R-CNN-এর উপর একটি segmentation mask branch যোগ করা হয়েছে।</p>
    <p>প্রতিটি detected object-এর জন্য: class label + bounding box + binary mask (28×28)।</p>
    <p><strong>RoI Align:</strong> RoI Pooling-এর improved version — bilinear interpolation দিয়ে misalignment সমাধান।</p>
    <pre><code># Detectron2 দিয়ে Mask R-CNN inference
# pip install detectron2

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

setup_logger()

# Config setup
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
cfg.MODEL.DEVICE = "cpu"  # GPU থাকলে "cuda"

predictor = DefaultPredictor(cfg)

# Image-এ inference
image = cv2.imread("your_image.jpg")
outputs = predictor(image)

# Results
instances = outputs["instances"]
print(f"Detected {len(instances)} instances")
print(f"Classes: {instances.pred_classes}")
print(f"Scores:  {instances.scores}")
print(f"Masks shape: {instances.pred_masks.shape}")  # (N, H, W) boolean

# Visualize
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(instances.to("cpu"))
result_image = out.get_image()[:, :, ::-1]
cv2.imwrite("result.jpg", result_image)
</code></pre>
  `,
};
