export const nn_4_keras = {
  title: "Keras ও TensorFlow দিয়ে নিউরাল নেটওয়ার্ক",
  description: "TensorFlow/Keras-এর Sequential API দিয়ে neural network তৈরি, MNIST handwritten digit classification, training callbacks এবং model save/load সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-keras-tensorflow",
  content: `
    <h3>১. Keras কেন? NumPy থেকে Keras-এ যাওয়া</h3>
    <p>আগের ব্লগে আমরা NumPy দিয়ে neural network implement করেছিলাম — ৩০০+ লাইন কোড। Keras দিয়ে একই কাজ ১০-১৫ লাইনে করা যায়।</p>
    <p>Keras-এর সুবিধা:</p>
    <ul>
      <li>GPU/TPU support: automatic</li>
      <li>Automatic differentiation: backprop নিজেই করে</li>
      <li>Built-in optimizers, losses, metrics</li>
      <li>Callbacks: EarlyStopping, ModelCheckpoint</li>
      <li>TensorBoard integration</li>
    </ul>
    <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow version check
print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)
</code></pre>

    <h3>২. MNIST Dataset: Handwritten Digits</h3>
    <p>MNIST হলো ML-এর "Hello World" — ৭০,০০০ handwritten digit image (0-9)। প্রতিটি image 28×28 grayscale pixel।</p>
    <pre><code># MNIST load
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training:", X_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print("Test:", X_test.shape, y_test.shape)          # (10000, 28, 28) (10000,)
print("Pixel range:", X_train.min(), X_train.max()) # 0, 255

# Preprocessing
# 1. Normalize: 0-255 -> 0-1
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# 2. Flatten: 28x28 -> 784 (for Dense/MLP)
X_train_flat = X_train.reshape(-1, 784)  # (60000, 784)
X_test_flat  = X_test.reshape(-1, 784)   # (10000, 784)

# 3. Labels: integer (0-9) — sparse_categorical_crossentropy ব্যবহার করব
print("Label sample:", y_train[:10])  # [5 0 4 1 9 2 1 3 1 4]

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.suptitle("MNIST Sample Images")
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()
</code></pre>

    <h3>৩. Sequential Model তৈরি</h3>
    <p>Keras Sequential model-এ layer গুলো একটির পর একটি stack হয়।</p>
    <pre><code># Model 1: Simple MLP
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
], name="simple_mlp")

# Model summary
model.summary()
# Output:
# Layer (type)         Output Shape         Param #
# dense (Dense)        (None, 128)          100480
# dense_1 (Dense)      (None, 64)           8256
# dense_2 (Dense)      (None, 10)           650
# Total params: 109,386

# Compile: optimizer, loss, metrics নির্ধারণ
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # integer labels
    metrics=['accuracy']
)
# Note: one-hot labels হলে 'categorical_crossentropy' ব্যবহার করতে হবে
</code></pre>

    <h3>৪. Training: fit(), evaluate(), predict()</h3>
    <pre><code"># Callbacks define
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_mnist_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Training
history = model.fit(
    X_train_flat, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,  # 10% of train as validation
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")  # ~0.98

# Prediction
y_pred_proba = model.predict(X_test_flat[:5])   # shape: (5, 10)
y_pred = np.argmax(y_pred_proba, axis=1)
print("Predicted:", y_pred)
print("Actual:   ", y_test[:5])
</code></pre>

    <h3>৫. Training Curves ও Analysis</h3>
    <pre><code>def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

plot_training_history(history)

# Confusion matrix (top errors)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred_all = np.argmax(model.predict(X_test_flat), axis=1)
cm = confusion_matrix(y_test, y_pred_all)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix — MNIST Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_test, y_pred_all))
</code></pre>

    <h3>৬. Model Save ও Load</h3>
    <pre><code># Save করার তিনটি উপায়

# 1. SavedModel format (recommended)
model.save('mnist_model_savedmodel')

# 2. Keras format (.keras)
model.save('mnist_model.keras')

# 3. Weights only
model.save_weights('mnist_weights.weights.h5')

# Load করা
loaded_model = keras.models.load_model('mnist_model.keras')
test_loss, test_acc = loaded_model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Loaded model accuracy: {test_acc:.4f}")

# Model architecture দেখা
loaded_model.summary()

# Inference: single image
single_image = X_test_flat[0:1]  # shape (1, 784)
prediction = loaded_model.predict(single_image)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class]
print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
</code></pre>
    <table>
      <thead><tr><th>API</th><th>ব্যবহার</th><th>Return</th></tr></thead>
      <tbody>
        <tr><td>model.fit()</td><td>Training</td><td>History object (loss, accuracy per epoch)</td></tr>
        <tr><td>model.evaluate()</td><td>Test set performance</td><td>[loss, metric1, metric2, ...]</td></tr>
        <tr><td>model.predict()</td><td>Probability/output</td><td>NumPy array (raw predictions)</td></tr>
        <tr><td>model.summary()</td><td>Architecture overview</td><td>Print to stdout</td></tr>
        <tr><td>model.save()</td><td>Persistence</td><td>Saves to disk</td></tr>
        <tr><td>keras.models.load_model()</td><td>Load saved model</td><td>Keras model object</td></tr>
      </tbody>
    </table>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা Keras দিয়ে শিখলাম:</p>
    <ul>
      <li>Sequential API: layer stack করার সহজ পদ্ধতি</li>
      <li>compile(): optimizer, loss, metrics নির্ধারণ</li>
      <li>fit(): epochs, batch_size, validation_split, callbacks</li>
      <li>MNIST: normalize → flatten → Dense layers → softmax → 98%+ accuracy</li>
      <li>EarlyStopping: overfitting এড়াতে</li>
      <li>Model save/load: production deployment-এর জন্য</li>
    </ul>
    <p>পরবর্তী ব্লগে CNN (Convolutional Neural Network) দিয়ে image classification করব — MLP-এর চেয়ে অনেক বেশি accurate।</p>
  `,
};
