export const cnn_3_famous_architectures = {
  title: "বিখ্যাত CNN আর্কিটেকচার: LeNet থেকে MobileNet — বিবর্তনের গল্প",
  description: "LeNet-5, AlexNet, VGG, ResNet, Inception ও MobileNet-এর architecture, key innovations ও parameter count। ResNet-এর skip connection কেন vanishing gradient সমাধান করে, Keras-এ pre-trained ResNet50 দিয়ে inference।",
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
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

def lenet5(num_classes=10):
    """Original LeNet-5 architecture"""
    return models.Sequential([
        layers.Input(shape=(32, 32, 1)),      # Original: 32x32 (MNIST padded)
        layers.Conv2D(6,  (5,5), activation='tanh'),          # → 28x28x6
        layers.AveragePooling2D((2,2), strides=2),            # → 14x14x6
        layers.Conv2D(16, (5,5), activation='tanh'),          # → 10x10x16
        layers.AveragePooling2D((2,2), strides=2),            # → 5x5x16
        layers.Flatten(),                                      # → 400
        layers.Dense(120, activation='tanh'),
        layers.Dense(84,  activation='tanh'),
        layers.Dense(num_classes, activation='softmax'),
    ])

def alexnet(num_classes=1000):
    """Simplified AlexNet (single GPU, no LRN)"""
    return models.Sequential([
        layers.Input(shape=(227, 227, 3)),
        layers.Conv2D(96,  (11,11), strides=4, activation='relu'), # → 55x55x96
        layers.MaxPooling2D((3,3), strides=2),                      # → 27x27x96
        layers.Conv2D(256, (5,5),  padding='same', activation='relu'), # → 27x27x256
        layers.MaxPooling2D((3,3), strides=2),                      # → 13x13x256
        layers.Conv2D(384, (3,3),  padding='same', activation='relu'),
        layers.Conv2D(384, (3,3),  padding='same', activation='relu'),
        layers.Conv2D(256, (3,3),  padding='same', activation='relu'),
        layers.MaxPooling2D((3,3), strides=2),                      # → 6x6x256
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

lenet = lenet5()
print(f"LeNet-5 params: {lenet.count_params():,}")

alex = alexnet()
print(f"AlexNet params: {alex.count_params():,}")
</code></pre>

    <h3>৩. VGG: Depth ও Simplicity</h3>
    <p><strong>VGG (Oxford, 2014):</strong> মূল idea — শুধু 3×3 conv filters ব্যবহার করো, কিন্তু network অনেক deep করো।</p>
    <p>কেন 3×3 filters? দুটি 3×3 conv = একটি 5×5 conv-এর receptive field, কিন্তু:</p>
    <ul>
      <li>Parameters: 2×(3×3×C×C) = 18C² বনাম 5×5×C×C = 25C² — ২৮% কম</li>
      <li>বেশি non-linearity (দুটি ReLU বনাম একটি)</li>
    </ul>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

def vgg_block(x, num_convs, filters):
    """VGG block: num_convs × Conv(3x3) → MaxPool"""
    for _ in range(num_convs):
        x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2), strides=2)(x)
    return x

def vgg16(num_classes=1000, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = vgg_block(inputs,  2,  64)   # Block 1: → 112x112x64
    x = vgg_block(x,       2, 128)   # Block 2: → 56x56x128
    x = vgg_block(x,       3, 256)   # Block 3: → 28x28x256
    x = vgg_block(x,       3, 512)   # Block 4: → 14x14x512
    x = vgg_block(x,       3, 512)   # Block 5: → 7x7x512

    x = layers.Flatten()(x)          # 7*7*512 = 25088
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = vgg16()
print(f"VGG-16 params: {model.count_params():,}")
# ~138 million — mostly Dense layers-এ (FC layers 102M params নেয়!)
</code></pre>

    <h3>৪. ResNet: Skip Connection ও Vanishing Gradient সমাধান</h3>
    <p>VGG-এর পর deeper network তৈরি করলে accuracy <em>কমে যেত</em> — degradation problem। ResNet (He et al., 2015) এর সমাধান করে <strong>residual/skip connections</strong> দিয়ে।</p>
    <p><strong>Skip Connection:</strong> output = F(x) + x</p>
    <p>Network শুধু residual F(x) শেখে (x থেকে কতটুকু পরিবর্তন হবে), পুরো mapping নয়। যদি optimal function = identity হয়, তাহলে F(x) = 0 শিখলেই হয় — সহজ।</p>
    <p><strong>Gradient flow:</strong> d(Loss)/dx = d(Loss)/d(F(x)+x) = gradient × (1 + dF/dx) — gradient সরাসরি পৌঁছায়।</p>
    <p><strong>ResNet-50 Bottleneck Block:</strong> 1×1 → 3×3 → 1×1 (channels: C → C/4 → C/4 → C)</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, stride=1):
    """ResNet basic residual block (for ResNet-18/34)"""
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut: dimension match করতে projection লাগতে পারে
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Skip connection: F(x) + x
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def bottleneck_block(x, filters, stride=1):
    """ResNet-50 bottleneck block: 1x1 → 3x3 → 1x1"""
    shortcut = x
    expanded = filters * 4  # Bottleneck expansion

    # 1x1 conv: reduce channels
    x = layers.Conv2D(filters, (1,1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3x3 conv
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 1x1 conv: expand channels back
    x = layers.Conv2D(expanded, (1,1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut
    if stride != 1 or shortcut.shape[-1] != expanded:
        shortcut = layers.Conv2D(expanded, (1,1), strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet18(num_classes=10, input_shape=(32, 32, 3)):
    """Small ResNet-18 for CIFAR-10"""
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 4 stages
    for filters, stride in [(64,1), (128,2), (256,2), (512,2)]:
        x = residual_block(x, filters, stride=stride)
        x = residual_block(x, filters, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

resnet18 = build_resnet18()
print(f"Custom ResNet-18 params: {resnet18.count_params():,}")
</code></pre>

    <h3>৫. Inception Module ও GoogLeNet</h3>
    <p><strong>GoogLeNet/Inception (Google, 2014):</strong> মূল idea — আলাদা আলাদা kernel size (1×1, 3×3, 5×5) parallel-এ চালাও, তারপর concatenate করো। Network নিজেই শিখবে কোন scale-এ কী feature দরকার।</p>
    <p>1×1 conv-এর ভূমিকা: channel reduction (bottleneck) — computational cost কমায়।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers

def inception_module(x, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pool):
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
    branch1 = layers.Conv2D(f_1x1, (1,1), padding='same', activation='relu')(x)

    # Branch 2: 1x1 → 3x3
    branch2 = layers.Conv2D(f_3x3_reduce, (1,1), padding='same', activation='relu')(x)
    branch2 = layers.Conv2D(f_3x3, (3,3), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 → 5x5
    branch3 = layers.Conv2D(f_5x5_reduce, (1,1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(f_5x5, (5,5), padding='same', activation='relu')(branch3)

    # Branch 4: MaxPool → 1x1
    branch4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch4 = layers.Conv2D(f_pool, (1,1), padding='same', activation='relu')(branch4)

    # Concatenate all branches along channel axis
    output = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return output

# উদাহরণ: Inception 3a module (GoogLeNet)
inputs = tf.keras.Input(shape=(28, 28, 192))
x = inception_module(inputs, 64, 96, 128, 16, 32, 32)
print(f"Inception output shape: {x.shape}")
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
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers

def depthwise_separable_block(x, filters, stride=1):
    """MobileNet-style depthwise separable conv block"""
    # Depthwise conv: প্রতি channel-এ আলাদা 3x3 filter
    x = layers.DepthwiseConv2D((3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Pointwise conv: 1x1, channel mixing
    x = layers.Conv2D(filters, (1,1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def mobilenet_small(num_classes=10):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3,3), strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    configs = [(64,1), (128,2), (128,1), (256,2), (256,1)]
    for filters, stride in configs:
        x = depthwise_separable_block(x, filters, stride=stride)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

mob = mobilenet_small()
print(f"MobileNet-small params: {mob.count_params():,}")

# Compare: standard CNN vs MobileNet-style
standard_params = sum([3*3*3*32, 3*3*32*64, 3*3*64*128, 3*3*128*256])
mobile_params   = sum([3*3*3 + 3*32, 3*3*32 + 32*64, 3*3*64 + 64*128, 3*3*128 + 128*256])
print(f"\nConv params (4 layers): {standard_params:,}")
print(f"DepthSep params (4 layers): {mobile_params:,}")
print(f"Reduction: {standard_params/mobile_params:.1f}x")
</code></pre>

    <h3>৭. Pre-trained ResNet50 দিয়ে Inference</h3>
    <p>Keras-এ ImageNet-pre-trained ResNet50 সরাসরি ব্যবহার করা যায়।</p>
    <pre><code>import tensorflow as tf
import numpy as np

# Pre-trained ResNet50 load (ImageNet weights)
resnet50 = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=True,     # Classifier head সহ
    input_shape=(224, 224, 3)
)

print(f"ResNet50 params: {resnet50.count_params():,}")

# একটি random image দিয়ে inference
dummy_image = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32)

# ResNet50 preprocessing
preprocessed = tf.keras.applications.resnet50.preprocess_input(dummy_image)
predictions   = resnet50.predict(preprocessed)

# Top-5 predictions decode করি
decoded = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)
print("\nTop-5 Predictions:")
for rank, (class_id, class_name, confidence) in enumerate(decoded[0], 1):
    print(f"  {rank}. {class_name}: {confidence:.4f}")

# Feature extractor হিসেবে ব্যবহার (transfer learning)
feature_extractor = tf.keras.Model(
    inputs=resnet50.input,
    outputs=resnet50.get_layer('avg_pool').output  # 2048-dim features
)
features = feature_extractor.predict(preprocessed)
print(f"\nFeature vector shape: {features.shape}")  # (1, 2048)
</code></pre>
  `,
};
