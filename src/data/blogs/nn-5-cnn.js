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
    <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
    <pre><code># MNIST dataset (28x28 grayscale)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess for CNN: need channel dimension
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
X_train = X_train[..., np.newaxis]  # (60000, 28, 28, 1)
X_test  = X_test[..., np.newaxis]   # (10000, 28, 28, 1)

print("CNN input shape:", X_train.shape)  # (60000, 28, 28, 1)

# CNN model
cnn_model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),     # 28x28 -> 14x14

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),     # 14x14 -> 7x7

    # Classifier head
    layers.Flatten(),                # 7*7*64 = 3136
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="cnn_mnist")

cnn_model.summary()
# Conv2D(32): 32 filters, 3x3 kernel, 320 params
# Conv2D(64): 64 filters, 18496 params
# Dense(128): 3136*128 + 128 = 401536 params
# Total: ~422K params (vs MLP 109K but much better accuracy!)
</code></pre>

    <h3>৫. Training ও MLP vs CNN তুলনা</h3>
    <pre><code">cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_history = cnn_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_acc:.4f}")  # ~0.993

# MLP comparison (same data, flattened)
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

mlp_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
mlp_model.fit(X_train_flat, y_train, epochs=10, batch_size=128,
              validation_split=0.1, verbose=0)
mlp_loss, mlp_acc = mlp_model.evaluate(X_test_flat, y_test, verbose=0)
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
    <pre><code"># CIFAR-10: 32x32 RGB, 10 classes
(X_cifar_train, y_cifar_train), (X_cifar_test, y_cifar_test) = keras.datasets.cifar10.load_data()
X_cifar_train = X_cifar_train.astype('float32') / 255.0
X_cifar_test  = X_cifar_test.astype('float32') / 255.0
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# CNN for CIFAR-10 (more complex)
cifar_cnn = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cifar_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Transfer Learning (brief intro):
# ImageNet pre-trained models শুরু থেকে train না করে fine-tune করা
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False  # Freeze pre-trained weights

transfer_model = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print("Transfer learning model: VGG16 base + custom head")
print("Pre-trained on 1.2M ImageNet images!")
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম:</p>
    <ul>
      <li>CNN-এর intuition: local patterns → hierarchical features</li>
      <li>Conv2D: filters, feature maps, padding, stride</li>
      <li>MaxPooling2D: spatial downsampling, translation invariance</li>
      <li>MNIST CNN: 99%+ accuracy (MLP ~98%)</li>
      <li>Transfer Learning: ImageNet pre-trained VGG16, ResNet, EfficientNet</li>
    </ul>
    <p>পরবর্তী ব্লগে Regularization techniques — Dropout, Batch Normalization, L1/L2 — দিয়ে overfitting কমানো শিখব।</p>
  `,
};
