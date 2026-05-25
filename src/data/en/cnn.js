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
# Example: Conv2D(32 filters, 3x3) on RGB input (3 channels)
#   = (3 * 3 * 3 + 1) * 32 = 896 parameters
#
# Each filter produces one feature map → output has C_out channels

import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
])
model.summary()
# Layer 1: (3*3*1+1)*32  =  320 params
# Layer 2: (3*3*32+1)*64 = 18,496 params</code></pre>

      <h3>3. Stride & Padding</h3>
      <pre><code>from tensorflow import keras

# Stride=2: skip every other position → output half the size
# Padding='same': add zeros around input → output same spatial size as input
# Padding='valid' (default): no padding → output shrinks

examples = [
    keras.layers.Conv2D(32, 3, strides=1, padding='valid'),   # shrinks by 2 each dim
    keras.layers.Conv2D(32, 3, strides=1, padding='same'),    # same spatial size
    keras.layers.Conv2D(32, 3, strides=2, padding='same'),    # halves spatial size
]

inp = keras.Input(shape=(28, 28, 1))
for layer in examples:
    out = layer(inp)
    print(f"strides={layer.strides}, padding={layer.padding} → {out.shape}")</code></pre>
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
x = keras.layers.Conv2D(64, 1, activation='relu')(inp)  # 1x1 conv
print(f"1x1 conv output: {x.shape}")</code></pre>

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

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Demonstrate max vs avg pooling on a tiny feature map
feat_map = np.array([[1, 3, 2, 4],
                     [5, 6, 1, 2],
                     [3, 1, 4, 7],
                     [2, 8, 3, 1]], dtype='float32').reshape(1,4,4,1)

max_pool = keras.layers.MaxPooling2D(2, strides=2)(feat_map)
avg_pool = keras.layers.AveragePooling2D(2, strides=2)(feat_map)
gap      = keras.layers.GlobalAveragePooling2D()(feat_map)

print("MaxPool:", max_pool.numpy().squeeze())   # [[6,4],[8,7]]
print("AvgPool:", avg_pool.numpy().squeeze())   # [[3.75,2.25],[3.5,3.75]]
print("GAP:    ", gap.numpy())                  # single value per channel</code></pre>
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

model_with_bn = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(2),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation='softmax'),
])
model_with_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])</code></pre>

      <h3>3. Dropout in CNNs</h3>
      <pre><code">from tensorflow import keras

# Regular Dropout: randomly zeroes individual activations
# SpatialDropout2D: randomly zeroes entire feature maps
# → Better for CNNs because adjacent pixels are correlated

model = keras.Sequential([
    keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(32,32,3)),
    keras.layers.SpatialDropout2D(0.25),   # drop entire feature maps
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.SpatialDropout2D(0.25),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dropout(0.5),             # regular dropout before Dense
    keras.layers.Dense(10, activation='softmax'),
])</code></pre>

      <h3>4. BN Impact on CIFAR-10</h3>
      <pre><code">(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

def build_cnn(use_bn=True):
    layers = [keras.layers.Input(shape=(32,32,3))]
    for filters in [32, 64]:
        layers += [keras.layers.Conv2D(filters, 3, padding='same')]
        if use_bn: layers.append(keras.layers.BatchNormalization())
        layers += [keras.layers.Activation('relu'), keras.layers.MaxPooling2D(2)]
    layers += [keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(10, activation='softmax')]
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

h1 = build_cnn(use_bn=False).fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_test, y_test), verbose=0)
h2 = build_cnn(use_bn=True).fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_test, y_test), verbose=0)
print(f"Without BN: {max(h1.history['val_accuracy']):.4f}")
print(f"With BN:    {max(h2.history['val_accuracy']):.4f}")</code></pre>
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
      <pre><code>import tensorflow as tf
from tensorflow import keras

# ResNet insight: learning residual F(x) = H(x) - x is easier than H(x) directly
# Skip connection: output = F(x) + x   (identity shortcut)
# If gradient vanishes through F(x), it still flows through the identity path

def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x

    # Main path
    y = keras.layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.Conv2D(filters, 3, padding='same')(y)
    y = keras.layers.BatchNormalization()(y)

    # Adjust shortcut if shape changed
    if downsample or x.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride)(x)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    y = keras.layers.Add()([y, shortcut])  # F(x) + x
    y = keras.layers.ReLU()(y)
    return y

# Build a small ResNet
inp = keras.Input(shape=(32, 32, 3))
x   = keras.layers.Conv2D(32, 3, padding='same')(inp)
x   = keras.layers.BatchNormalization()(x)
x   = keras.layers.ReLU()(x)
x   = residual_block(x, 32)
x   = residual_block(x, 64, downsample=True)
x   = residual_block(x, 64)
x   = keras.layers.GlobalAveragePooling2D()(x)
out = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inp, out)
print(f"Custom ResNet params: {model.count_params():,}")</code></pre>

      <h3>3. Inception Module — Parallel Multi-Scale Filters</h3>
      <pre><code">def inception_module(x, f1, f3, f5, fpool):
    # Branch 1: 1x1 conv
    b1 = keras.layers.Conv2D(f1, 1, activation='relu', padding='same')(x)
    # Branch 2: 1x1 → 3x3
    b2 = keras.layers.Conv2D(f3, 1, activation='relu', padding='same')(x)
    b2 = keras.layers.Conv2D(f3, 3, activation='relu', padding='same')(b2)
    # Branch 3: 1x1 → 5x5
    b3 = keras.layers.Conv2D(f5, 1, activation='relu', padding='same')(x)
    b3 = keras.layers.Conv2D(f5, 5, activation='relu', padding='same')(b3)
    # Branch 4: MaxPool → 1x1
    b4 = keras.layers.MaxPooling2D(3, strides=1, padding='same')(x)
    b4 = keras.layers.Conv2D(fpool, 1, activation='relu', padding='same')(b4)
    # Concatenate along channel axis
    return keras.layers.Concatenate()([b1, b2, b3, b4])</code></pre>

      <h3>4. MobileNet — Efficient Convolutions</h3>
      <pre><code"># Standard Conv2D:      H*W*C_in*C_out*F*F  multiplications
# Depthwise Separable:  H*W*C_in*F*F + H*W*C_in*C_out  (much less!)
# Reduction factor ≈ 1/C_out + 1/F^2  → ~8-9x fewer operations

# In Keras, use SeparableConv2D for depthwise separable:
dws_layer = keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu')

# Load pre-trained MobileNetV2 and run inference
base = keras.applications.MobileNetV2(weights='imagenet', include_top=True)
# Preprocess: scale to [-1, 1]
preprocess = keras.applications.mobilenet_v2.preprocess_input
decode     = keras.applications.mobilenet_v2.decode_predictions

import numpy as np
dummy_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype('float32')
preds = base.predict(preprocess(dummy_img))
print(decode(preds, top=3))</code></pre>

      <h3>5. Pre-trained ResNet50 Inference</h3>
      <pre><code">resnet = keras.applications.ResNet50(weights='imagenet')

# Load and preprocess an image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

img      = image.load_img('dog.jpg', target_size=(224, 224))
x        = image.img_to_array(img)
x        = np.expand_dims(x, axis=0)
x        = preprocess_input(x)

preds = resnet.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])</code></pre>
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
        order = [j for j in order if compute_iou(boxes[i], boxes[j]) < iou_threshold]
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
    description: "Fully convolutional networks, transposed convolutions for upsampling, U-Net encoder-decoder with skip connections built in Keras, Mask R-CNN, and mIoU evaluation.",
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
#   keras.layers.Conv2DTranspose(filters, kernel, strides=2, padding='same')
#     strides=2 → doubles spatial dimensions

import tensorflow as tf
from tensorflow import keras

# Encoder: downsample
inp = keras.Input(shape=(128, 128, 3))
x   = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
x   = keras.layers.MaxPooling2D(2)(x)     # 64x64
x   = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x   = keras.layers.MaxPooling2D(2)(x)     # 32x32

# Decoder: upsample
x   = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)  # 64x64
x   = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)  # 128x128
out = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)  # binary segmentation
fcn = keras.Model(inp, out)
print(f"Input: {inp.shape} → Output: {out.shape}")</code></pre>

      <h3>3. U-Net — Skip Connections for Sharp Boundaries</h3>
      <pre><code">def unet(input_shape=(128, 128, 1), num_classes=1):
    inp = keras.Input(shape=input_shape)

    # Encoder (contracting path)
    c1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    c1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D(2)(c1)   # 64x64

    c2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D(2)(c2)   # 32x32

    # Bottleneck
    bn = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    bn = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(bn)

    # Decoder (expanding path) with skip connections
    u1 = keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(bn)  # 64x64
    u1 = keras.layers.Concatenate()([u1, c2])   # skip connection from encoder
    d1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    d1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(d1)

    u2 = keras.layers.Conv2DTranspose(16, 2, strides=2, padding='same')(d1)  # 128x128
    u2 = keras.layers.Concatenate()([u2, c1])   # skip connection
    d2 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(u2)
    d2 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(d2)

    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    out = keras.layers.Conv2D(num_classes, 1, activation=activation)(d2)
    return keras.Model(inp, out)

model = unet(input_shape=(128, 128, 1), num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()</code></pre>

      <h3>4. Training U-Net with Custom Data</h3>
      <pre><code"># Synthetic segmentation data for demonstration
import numpy as np

def make_segmentation_data(n=200, img_size=128):
    X = np.zeros((n, img_size, img_size, 1), dtype='float32')
    Y = np.zeros((n, img_size, img_size, 1), dtype='float32')
    for i in range(n):
        # random circle
        cx, cy = np.random.randint(20, img_size-20, 2)
        r      = np.random.randint(10, 30)
        xx, yy = np.ogrid[:img_size, :img_size]
        mask   = ((xx-cx)**2 + (yy-cy)**2) <= r**2
        X[i, :, :, 0] = np.random.randn(img_size, img_size) * 0.1
        X[i, mask, 0] += 1.0   # bright circle
        Y[i, mask, 0]  = 1.0
    return X, Y

X, Y = make_segmentation_data(400)
X_tr, X_te, Y_tr, Y_te = X[:320], X[320:], Y[:320], Y[320:]

model = unet(input_shape=(128, 128, 1), num_classes=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_tr, Y_tr, epochs=20, batch_size=16, validation_data=(X_te, Y_te))</code></pre>

      <h3>5. Evaluation Metrics</h3>
      <pre><code"># Pixel Accuracy: correct pixels / total pixels (misleading on imbalanced data)
# IoU (Jaccard Index): intersection / union per class
# mIoU: mean IoU across all classes (primary segmentation metric)

def compute_iou(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(float)
    intersection = (y_true * y_pred_bin).sum()
    union        = y_true.sum() + y_pred_bin.sum() - intersection
    return intersection / (union + 1e-8)

y_pred = model.predict(X_te)
ious   = [compute_iou(Y_te[i,:,:,0], y_pred[i,:,:,0]) for i in range(len(X_te))]
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
      <pre><code>import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0
y_train, y_test = y_train.squeeze(), y_test.squeeze()

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i]); ax.set_title(class_names[y_train[i]]); ax.axis('off')
plt.tight_layout(); plt.show()</code></pre>

      <h3>2. Data Augmentation Pipeline</h3>
      <pre><code">augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
], name='augmentation')

# Visualize augmented images
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax in axes:
    img = augment(X_train[0:1], training=True)[0]
    ax.imshow(img); ax.axis('off')
plt.suptitle('Augmented versions of one image'); plt.show()</code></pre>

      <h3>3. Three Models</h3>
      <pre><code">def build_simple_cnn():
    return keras.Sequential([
        augment,
        keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(32,32,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax'),
    ])

def build_resnet():
    inp = keras.Input(shape=(32,32,3))
    x   = augment(inp)
    x   = keras.layers.Conv2D(32, 3, padding='same')(x)
    x   = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)
    for f in [32, 64, 128]:
        shortcut = keras.layers.Conv2D(f, 1)(x) if x.shape[-1] != f else x
        y = keras.layers.Conv2D(f, 3, padding='same')(x)
        y = keras.layers.BatchNormalization()(y); y = keras.layers.ReLU()(y)
        y = keras.layers.Conv2D(f, 3, padding='same')(y)
        y = keras.layers.BatchNormalization()(y)
        x = keras.layers.ReLU()(keras.layers.Add()([y, shortcut]))
    x   = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inp, out)

def build_transfer():
    base = keras.applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(shape=(32,32,3))
    x   = augment(inp)
    x   = keras.applications.mobilenet_v2.preprocess_input(x * 255)
    x   = base(x, training=False)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dense(128, activation='relu')(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inp, out)</code></pre>

      <h3>4. Training & Comparison</h3>
      <pre><code">callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
]

models = {'Simple CNN': build_simple_cnn(), 'ResNet': build_resnet(), 'MobileNetV2': build_transfer()}
results = {}
for name, m in models.items():
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(X_train, y_train, epochs=50, batch_size=64,
              validation_split=0.1, callbacks=callbacks, verbose=0)
    _, acc = m.evaluate(X_test, y_test, verbose=0)
    results[name] = {'val_acc': max(h.history['val_accuracy']), 'test_acc': acc, 'params': m.count_params()}

for name, r in results.items():
    print(f"{name:15s}: test={r['test_acc']:.4f}, params={r['params']:,}")</code></pre>

      <h3>5. Grad-CAM Visualization</h3>
      <pre><code"># Grad-CAM: gradient of class score w.r.t. last conv layer activations
# Positive gradients → regions important for this class

def grad_cam(model, img, class_idx, last_conv_name):
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img[np.newaxis])
        loss = preds[:, class_idx]
    grads    = tape.gradient(loss, conv_out)
    weights  = tf.reduce_mean(grads, axis=(0, 1, 2))        # global average pool
    cam      = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam      = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
    cam      = tf.image.resize(cam[..., tf.newaxis], img.shape[:2]).numpy().squeeze()
    return cam

import cv2
img       = X_test[42]
cam       = grad_cam(models['Simple CNN'], img, y_test[42], 'conv2d_2')
heatmap   = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay   = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap, 0.4, 0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(img);    axes[0].set_title('Original')
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
