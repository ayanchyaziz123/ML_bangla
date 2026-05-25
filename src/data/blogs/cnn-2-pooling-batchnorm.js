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
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

# Flatten vs GAP comparison
def model_with_flatten(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)         # 8*8*128 = 8192 neurons
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

def model_with_gap(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)  # (128,) vector
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

m_flat = model_with_flatten()
m_gap  = model_with_gap()

print(f"Flatten model params: {m_flat.count_params():,}")
print(f"GAP model params:     {m_gap.count_params():,}")
# GAP model অনেক কম params — Dense(8192→256) বাদ দেওয়া হয়েছে
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
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers

# Standard CNN block with BN
def conv_bn_relu(x, filters, kernel_size=3, stride=1):
    """Conv → BatchNorm → ReLU block"""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding='same',
        use_bias=False    # BN ব্যবহারে bias redundant, তাই False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# BN-এর learnable parameters: 4 per channel (gamma, beta, moving_mean, moving_var)
# 64 filters → 4 * 64 = 256 params (2 trainable + 2 non-trainable)

inputs = tf.keras.Input(shape=(32, 32, 3))
x = conv_bn_relu(inputs, 32)
x = conv_bn_relu(x, 64)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
</code></pre>

    <h3>৪. BN-এর সুবিধা ও Training প্রভাব</h3>
    <p>Batch Normalization কেন এত জনপ্রিয়:</p>
    <ul>
      <li><strong>Faster convergence:</strong> Higher learning rate ব্যবহার করা যায় (0.1 vs 0.001)</li>
      <li><strong>Internal Covariate Shift কমায়:</strong> প্রতিটি layer stable distribution দেখে</li>
      <li><strong>Regularization effect:</strong> Dropout-এর মতো কিছুটা noise inject করে</li>
      <li><strong>Gradient flow উন্নত করে:</strong> Deep network-এ vanishing gradient কমায়</li>
    </ul>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np

# CIFAR-10 load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test,  10)

def build_cnn(use_batchnorm=True):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    for filters in [32, 64]:
        if filters == 32:
            x = layers.Conv2D(filters, (3,3), padding='same', use_bias=not use_batchnorm)(inputs)
        else:
            x = layers.Conv2D(filters, (3,3), padding='same', use_bias=not use_batchnorm)(x)

        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Without BN — lower LR needed
model_no_bn = build_cnn(use_batchnorm=False)
model_no_bn.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy', metrics=['accuracy']
)

# With BN — higher LR ব্যবহার করা যায়
model_bn = build_cnn(use_batchnorm=True)
model_bn.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),  # 10x বেশি LR
    loss='categorical_crossentropy', metrics=['accuracy']
)

# Training (5 epochs for comparison)
history_no_bn = model_no_bn.fit(x_train, y_train, epochs=5,
    validation_data=(x_test, y_test), batch_size=128, verbose=0)
history_bn = model_bn.fit(x_train, y_train, epochs=5,
    validation_data=(x_test, y_test), batch_size=128, verbose=0)

print(f"Without BN - Final val accuracy: {history_no_bn.history['val_accuracy'][-1]:.4f}")
print(f"With BN    - Final val accuracy: {history_bn.history['val_accuracy'][-1]:.4f}")
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
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_with_dropout(spatial_dropout_rate=0.2, dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate)(x)  # Feature map-level dropout
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(spatial_dropout_rate)(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)                   # Regular dropout before Dense
    outputs = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs, outputs)

model = build_cnn_with_dropout()
model.summary()

# Note: Dropout is only active during training (training=True)
# Inference-এ automatically disabled হয়
</code></pre>

    <h3>৬. Complete CNN Block: সবকিছু একসাথে</h3>
    <p>Production-grade CNN block: Conv2D → BatchNorm → ReLU → MaxPool — এই pattern বারবার repeat করা হয়।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def cnn_block(x, filters, pool=True):
    """Standard CNN block: Conv→BN→ReLU→(MaxPool)"""
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if pool:
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.1)(x)
    return x

def build_full_cnn(num_classes=10):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = cnn_block(inputs,  32, pool=True)   # → 16x16x32
    x = cnn_block(x,       64, pool=True)   # → 8x8x64
    x = cnn_block(x,      128, pool=True)   # → 4x4x128

    x = layers.GlobalAveragePooling2D()(x)  # → (128,)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Load and train
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test,  10)

model = build_full_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

history = model.fit(
    x_train, y_train,
    epochs=50, batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=callbacks, verbose=1
)

# এই model CIFAR-10-এ ~80-85% accuracy দেওয়া উচিত
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
</code></pre>
  `,
};
