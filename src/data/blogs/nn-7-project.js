export const nn_7_project = {
  title: "প্রজেক্ট: Fashion MNIST পোশাক শ্রেণীবিন্যাস",
  description: "Fashion MNIST dataset দিয়ে সম্পূর্ণ end-to-end deep learning প্রজেক্ট — data exploration, MLP, CNN build, তুলনা, confusion matrix এবং best practices সহ।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 14,
  slug: "nn-project-fashion-mnist",
  content: `
    <h3>১. Fashion MNIST: Dataset পরিচয়</h3>
    <p>Fashion MNIST হলো MNIST-এর একটি কঠিন version। ৭০,০০০ grayscale image (28×28) দশটি পোশাক category-তে। MNIST digits-এর মতো simple নয় — real-world image classification challenge।</p>
    <p>Classes: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Sandal (5), Shirt (6), Sneaker (7), Bag (8), Ankle boot (9)।</p>
    <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

tf.random.set_seed(42)
np.random.seed(42)

# Class names
CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# Load data
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Full training set:", X_train_full.shape, y_train_full.shape)
print("Test set:", X_test.shape, y_test.shape)
print("Pixel range:", X_train_full.min(), "-", X_train_full.max())
</code></pre>

    <h3>২. Data Exploration ও Visualization</h3>
    <pre><code># Sample images visualization
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i], cmap='gray')
    ax.set_title(CLASS_NAMES[y_train_full[i]], fontsize=9)
    ax.axis('off')
plt.suptitle("Fashion MNIST Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig("fashion_samples.png")
plt.show()

# Class distribution
fig, ax = plt.subplots(figsize=(10, 4))
class_counts = np.bincount(y_train_full)
bars = ax.bar(CLASS_NAMES, class_counts, color=plt.cm.tab10.colors)
ax.set_title("Class Distribution (Training Set)")
ax.set_ylabel("Count")
plt.xticks(rotation=30, ha='right')
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(count), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

print("Class distribution (balanced):")
for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {i}: {name:15s} — {count} samples")
</code></pre>

    <h3>৩. Preprocessing</h3>
    <pre><code"># Preprocessing
# 1. Normalize 0-255 -> 0-1
X_train_full = X_train_full.astype('float32') / 255.0
X_test       = X_test.astype('float32') / 255.0

# 2. Train/Validation split
val_size = 5000
X_val   = X_train_full[:val_size]
y_val   = y_train_full[:val_size]
X_train = X_train_full[val_size:]
y_train = y_train_full[val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 3. Reshape for different models
X_train_flat = X_train.reshape(-1, 784)
X_val_flat   = X_val.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

X_train_cnn = X_train[..., np.newaxis]  # (55000, 28, 28, 1)
X_val_cnn   = X_val[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

# Common callbacks
def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=4, min_lr=1e-6, verbose=0),
        callbacks.ModelCheckpoint(f'{model_name}_best.keras',
                                   monitor='val_accuracy', save_best_only=True, verbose=0)
    ]
</code></pre>

    <h3>৪. তিনটি Model Build ও Train</h3>
    <pre><code"># Model 1: Simple MLP
def build_simple_mlp():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ], name='simple_mlp')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Model 2: Deep MLP with BN
def build_deep_mlp():
    model = keras.Sequential([
        layers.Dense(512, input_shape=(784,),
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),

        layers.Dense(10, activation='softmax')
    ], name='deep_mlp')
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Model 3: CNN
def build_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ], name='cnn')
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train all models
results = {}
histories = {}

models_config = [
    ('simple_mlp', build_simple_mlp, X_train_flat, X_val_flat, X_test_flat),
    ('deep_mlp',   build_deep_mlp,   X_train_flat, X_val_flat, X_test_flat),
    ('cnn',        build_cnn,        X_train_cnn,  X_val_cnn,  X_test_cnn),
]

for name, build_fn, X_tr, X_v, X_te in models_config:
    print(f"\nTraining {name}...")
    model = build_fn()
    start_time = time.time()

    history = model.fit(
        X_tr, y_train,
        epochs=50, batch_size=64,
        validation_data=(X_v, y_val),
        callbacks=get_callbacks(name),
        verbose=0
    )

    train_time = time.time() - start_time
    loss, acc = model.evaluate(X_te, y_test, verbose=0)
    n_params = model.count_params()

    results[name] = {
        'model': model, 'history': history,
        'test_acc': acc, 'test_loss': loss,
        'n_params': n_params, 'train_time': train_time
    }
    histories[name] = history
    print(f"  Params: {n_params:,} | Test Acc: {acc:.4f} | Time: {train_time:.1f}s")
</code></pre>

    <h3>৫. Model Comparison ও Confusion Matrix</h3>
    <pre><code"># Comparison table print
print("\n" + "="*70)
print(f"{'Model':<15} {'Params':>10} {'Test Acc':>10} {'Time (s)':>10}")
print("="*70)
for name, r in results.items():
    print(f"{name:<15} {r['n_params']:>10,} {r['test_acc']:>10.4f} {r['train_time']:>10.1f}")

# Training curves comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'simple_mlp': 'blue', 'deep_mlp': 'green', 'cnn': 'red'}

for name, r in results.items():
    h = r['history']
    axes[0].plot(h.history['val_accuracy'], label=name, color=colors[name])
    axes[1].plot(h.history['val_loss'],     label=name, color=colors[name])

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("model_comparison_curves.png")
plt.show()

# Confusion Matrix for best model (CNN)
best_model = results['cnn']['model']
y_pred = np.argmax(best_model.predict(X_test_cnn), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("CNN Confusion Matrix — Fashion MNIST Test Set")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("fashion_confusion_matrix.png")
plt.show()
</code></pre>

    <h3>৬. Per-Class Accuracy ও Error Analysis</h3>
    <pre><code"># Per-class accuracy
print("\nPer-Class Accuracy (CNN):")
print("-"*45)
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = (y_test == i)
    class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
    print(f"  {i}: {class_name:<15} — {class_acc:.3f} ({class_acc*100:.1f}%)")

# Most confused pairs
print("\nMost Confused Pairs:")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    true_class = CLASS_NAMES[idx[0]]
    pred_class = CLASS_NAMES[idx[1]]
    count = cm_no_diag[idx]
    print(f"  True: {true_class:<15} → Predicted: {pred_class:<15} ({count} errors)")
    cm_no_diag[idx] = 0

# Classification report
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
</code></pre>
    <table>
      <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>Test Accuracy</th><th>Training Time</th></tr></thead>
      <tbody>
        <tr><td>Simple MLP</td><td>784→256→128→10 + Dropout</td><td>~234K</td><td>~88.5%</td><td>~30s</td></tr>
        <tr><td>Deep MLP + BN</td><td>784→512→256→128→10 + BN + Dropout + L2</td><td>~530K</td><td>~89.8%</td><td>~60s</td></tr>
        <tr><td>CNN</td><td>Conv32→Conv32→Pool→Conv64→Conv64→Pool→256→10</td><td>~620K</td><td>~92.5%</td><td>~180s</td></tr>
      </tbody>
    </table>

    <h3>৭. Best Practices সারসংক্ষেপ</h3>
    <pre><code"># Final best practices checklist

# 1. Data Preprocessing
print("Preprocessing:")
print("  - Normalize: 0-255 -> 0-1 (or standardize)")
print("  - Always use validation set (separate from test)")
print("  - Check class balance")

# 2. Architecture
print("\nArchitecture:")
print("  - Start simple, add complexity as needed")
print("  - CNN for images, Dense for tabular")
print("  - BatchNorm before activation in deep nets")
print("  - ReLU hidden layers, Softmax output (multi-class)")

# 3. Training
print("\nTraining:")
print("  - Adam optimizer (lr=0.001) as default")
print("  - Mini-batch size: 32-128")
print("  - Use EarlyStopping + restore_best_weights")
print("  - ReduceLROnPlateau for better convergence")

# 4. Regularization
print("\nRegularization (if overfit):")
print("  - Dropout: 0.2-0.5 for Dense, 0.25 for Conv")
print("  - L2: lambda=0.001 on Dense weights")
print("  - Data augmentation for images")

# 5. Evaluation
print("\nEvaluation:")
print("  - Confusion matrix for class-level analysis")
print("  - Per-class accuracy: find weak spots")
print("  - Analyze worst errors for insights")
</code></pre>
    <table>
      <thead><tr><th>Best Practice</th><th>কেন গুরুত্বপূর্ণ</th><th>Default পছন্দ</th></tr></thead>
      <tbody>
        <tr><td>Normalize input</td><td>Stable gradients, faster convergence</td><td>0-1 বা StandardScaler</td></tr>
        <tr><td>He/Xavier init</td><td>Symmetry breaking, gradient flow</td><td>Keras default (glorot_uniform)</td></tr>
        <tr><td>Adam optimizer</td><td>Adaptive lr, momentum combined</td><td>lr=0.001</td></tr>
        <tr><td>Batch Normalization</td><td>Faster training, higher lr possible</td><td>Deep networks-এ সবসময়</td></tr>
        <tr><td>EarlyStopping</td><td>Overfitting রোধ, compute save</td><td>patience=10, restore_best=True</td></tr>
        <tr><td>Validation set</td><td>Unbiased performance estimate</td><td>10-20% of training data</td></tr>
        <tr><td>Confusion matrix</td><td>Class-level insights</td><td>Final evaluation-এ সবসময়</td></tr>
      </tbody>
    </table>
    <p>এই সিরিজে আমরা পার্সেপট্রন থেকে শুরু করে সম্পূর্ণ deep learning pipeline শিখলাম। পরবর্তী সিরিজে Recurrent Neural Networks (RNN/LSTM) এবং Natural Language Processing নিয়ে আলোচনা করব।</p>
  `,
};
