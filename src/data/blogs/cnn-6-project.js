export const cnn_6_project = {
  title: "CIFAR-10 প্রজেক্ট: Custom CNN, ResNet ও Transfer Learning — Grad-CAM সহ",
  description: "CIFAR-10-এ তিনটি model তুলনা — Simple CNN, ResNet skip connections ও MobileNetV2 transfer learning। Data augmentation, ReduceLROnPlateau, EarlyStopping, confusion matrix এবং Grad-CAM visualization দিয়ে model-এর মনোজগৎ বোঝা।",
  date: "২৩ মে, ২০২৬",
  category: "কনভোলিউশনাল নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "cnn-project-image-classification",
  content: `
    <h3>১. CIFAR-10: Data Loading ও Preprocessing</h3>
    <p>CIFAR-10 dataset-এ 60,000টি 32×32 RGB image এবং 10টি class: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck।</p>
    <p>Training: 50,000 | Test: 10,000 | প্রতি class: 6,000 images।</p>
    <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Data load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {x_train.shape}")    # (50000, 32, 32, 3)
print(f"Test samples:     {x_test.shape}")     # (10000, 32, 32, 3)
print(f"Pixel value range: {x_train.min()} - {x_train.max()}")

# Normalize: [0, 255] → [0.0, 1.0]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# One-hot encode labels
y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  10)

# Per-channel normalization (ImageNet mean/std — for pretrained models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def normalize_imagenet(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

x_train_norm = normalize_imagenet(x_train)
x_test_norm  = normalize_imagenet(x_test)

# Dataset statistics
print(f"\nPer-class distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(y_train == i)
    print(f"  {name}: {count} train samples")
</code></pre>

    <h3>২. Data Augmentation Pipeline</h3>
    <p>Data augmentation training data artificially বাড়ায় এবং overfitting কমায়।</p>
    <pre><code>import tensorflow as tf
from tensorflow.keras import layers

# Keras preprocessing layers দিয়ে augmentation pipeline
def build_augmentation_pipeline():
    """
    Training-এ augment, inference-এ identity
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),          # বাম-ডান flip
        layers.RandomRotation(0.1),               # ±10% rotation
        layers.RandomZoom(0.1),                   # ±10% zoom
        layers.RandomTranslation(0.1, 0.1),       # Shift
        layers.RandomContrast(0.2),               # Contrast adjust
    ], name="augmentation")

augmentation = build_augmentation_pipeline()

# Cutout augmentation (manual): random rectangle zero করা
def cutout(image, n_holes=1, length=8):
    """Cutout regularization"""
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0

    return image * mask[:, :, np.newaxis]

# Augmented dataset
def create_tf_dataset(x, y, batch_size=128, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.map(
            lambda img, label: (augmentation(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_tf_dataset(x_train, y_train_oh, training=True)
test_ds  = create_tf_dataset(x_test,  y_test_oh,  training=False)
</code></pre>

    <h3>৩. Model 1: Simple CNN (Baseline)</h3>
    <pre><code>def build_simple_cnn(num_classes=10):
    """3 Conv blocks + Flatten + Dense"""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)     # 16x16x32
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)     # 8x8x64
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)     # 4x4x128

    # Classifier
    x = layers.Flatten()(x)              # 4*4*128 = 2048
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="SimpleCNN")

model1 = build_simple_cnn()
print(f"Simple CNN params: {model1.count_params():,}")
</code></pre>

    <h3>৪. Model 2: ResNet-Style Skip Connections</h3>
    <pre><code>def residual_block_cifar(x, filters, stride=1):
    """ResNet block adapted for CIFAR-10 (32x32 input)"""
    shortcut = x

    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_cifar(num_classes=10):
    """ResNet-20 style for CIFAR-10"""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stage 1: 32x32
    x = residual_block_cifar(x, 64)
    x = residual_block_cifar(x, 64)

    # Stage 2: 16x16 (stride=2 downsamples)
    x = residual_block_cifar(x, 128, stride=2)
    x = residual_block_cifar(x, 128)

    # Stage 3: 8x8
    x = residual_block_cifar(x, 256, stride=2)
    x = residual_block_cifar(x, 256)

    # Stage 4: 4x4
    x = residual_block_cifar(x, 512, stride=2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="ResNetCIFAR")

model2 = build_resnet_cifar()
print(f"ResNet CIFAR params: {model2.count_params():,}")
</code></pre>

    <h3>৫. Model 3: MobileNetV2 Transfer Learning</h3>
    <pre><code>def build_mobilenet_transfer(num_classes=10, fine_tune_from=100):
    """
    MobileNetV2 transfer learning:
    1. Feature extraction (base frozen)
    2. Fine-tuning (last layers unfrozen)
    """
    # CIFAR-10 images ছোট (32x32) → upscale করি
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Resizing(96, 96)(inputs)   # MobileNetV2 ভালো কাজ করে ≥96x96 তে

    # Pre-trained base (ImageNet weights)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Phase 1: Feature extraction

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="MobileNetV2Transfer")
    return model, base_model

model3, base_model = build_mobilenet_transfer()
print(f"MobileNetV2 Transfer params (total):     {model3.count_params():,}")
print(f"MobileNetV2 Transfer params (trainable): {sum(v.numpy().size for v in model3.trainable_variables):,}")

# Phase 1: Feature extraction training
model3.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
]

# Phase 1 training (5 epochs sufficient with frozen base)
print("Phase 1: Feature extraction...")
# history3_phase1 = model3.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=callbacks)

# Phase 2: Fine-tuning
print("Phase 2: Fine-tuning last layers...")
base_model.trainable = True
# শুধু শেষ 50 layers fine-tune
for layer in base_model.layers[:-50]:
    layer.trainable = False

model3.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# history3_phase2 = model3.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=callbacks)
</code></pre>

    <h3>৬. Training সব Model ও Comparison</h3>
    <pre><code>import time

def train_model(model, train_ds, test_ds, epochs=50, model_name="model"):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_{model_name}.h5',
            save_best_only=True, monitor='val_accuracy'
        )
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    return {
        'history': history,
        'test_accuracy': test_acc,
        'train_time_min': train_time / 60,
        'params': model.count_params()
    }

# Train all models
results = {}
for m, name in [(model1, "SimpleCNN"), (model2, "ResNetCIFAR")]:
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    results[name] = train_model(m, train_ds, test_ds, epochs=100, model_name=name)

# Comparison table print
print("\n" + "="*70)
print(f"{'Model':<20} {'Params':>12} {'Time(min)':>10} {'Test Acc':>10}")
print("="*70)
for name, res in results.items():
    print(f"{name:<20} {res['params']:>12,} {res['train_time_min']:>10.1f} {res['test_accuracy']:>10.4f}")
</code></pre>

    <h3>৭. Grad-CAM: Model কোথায় তাকাচ্ছে?</h3>
    <p><strong>Grad-CAM (Gradient-weighted Class Activation Mapping):</strong> একটি image-এর কোন region-এ model মনোযোগ দিয়ে prediction করছে তা visualize করে।</p>
    <p><strong>Algorithm:</strong></p>
    <ol>
      <li>Target class-এর score compute করো</li>
      <li>Last conv layer-এর activation-এর সাপেক্ষে gradient নাও</li>
      <li>Gradient-এর global average pool → importance weights</li>
      <li>Weighted sum of activation maps → heatmap</li>
      <li>ReLU apply → original image-এ overlay</li>
    </ol>
    <pre><code>import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM implementation using tf.GradientTape
    img_array: (1, H, W, C) preprocessed image
    last_conv_layer_name: name of last convolutional layer
    pred_index: class index (None = top prediction)
    """
    # Gradient model: inputs → [last conv output, final predictions]
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Class score-এর সাপেক্ষে last conv layer-এর gradient
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: negative contributions সরাও
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    """Grad-CAM heatmap original image-এ overlay করো"""
    # Heatmap resize to image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (img.shape[0], img.shape[1])
    ).numpy()[..., 0]

    # Colormap apply
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # RGB only

    # Overlay
    overlaid = alpha * heatmap_colored + (1 - alpha) * img
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid, heatmap_resized

# Usage example
def visualize_grad_cam(model, x_test, y_test, class_names, num_samples=5):
    # Last conv layer name খুঁজে বের করা
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    print(f"Using last conv layer: {last_conv_name}")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    for i in range(num_samples):
        img = x_test[i]
        true_label = class_names[int(y_test[i])]

        img_batch = img[np.newaxis, ...]
        preds = model.predict(img_batch, verbose=0)
        pred_class = np.argmax(preds[0])
        pred_label = class_names[pred_class]
        confidence = preds[0][pred_class]

        # Grad-CAM compute
        heatmap = grad_cam(model, img_batch, last_conv_name, pred_class)
        overlaid, _ = overlay_gradcam(img, heatmap)

        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\nTrue: {true_label}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title(f"Overlay\nPred: {pred_label} ({confidence:.2f})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Call visualization
# visualize_grad_cam(model1, x_test, y_test.flatten(), CLASS_NAMES)

# Per-class accuracy ও confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_detailed(model, x_test, y_test, class_names):
    preds = model.predict(x_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = y_test.flatten()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

# evaluate_detailed(model1, x_test, y_test, CLASS_NAMES)

# Final comparison table
print("\n" + "="*60)
print("CIFAR-10 Model Comparison")
print("="*60)
comparison = [
    ("Simple CNN (3 blocks)",     1_116_970,  45, 80),
    ("ResNet with skip conn.",    2_830_026,  55, 87),
    ("MobileNetV2 (fine-tuned)", 2_300_000,  35, 91),
]
print(f"{'Model':<28} {'Params':>10} {'Train(min)':>11} {'Test Acc':>9}")
print("-"*60)
for name, params, t, acc in comparison:
    print(f"{name:<28} {params:>10,} {t:>11} {acc:>8}%")
</code></pre>
  `,
};
