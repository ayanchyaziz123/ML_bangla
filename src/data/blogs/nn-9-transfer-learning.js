export const nn_9_transfer_learning = {
  title: "Transfer Learning: Pre-trained Model দিয়ে দ্রুত Image Classification",
  description: "ImageNet pre-trained VGG16, MobileNetV2 দিয়ে feature extraction ও fine-tuning — কম data-তে high accuracy, data augmentation এবং Keras দিয়ে সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-transfer-learning",
  content: `
    <h3>১. Transfer Learning কী ও কেন</h3>
    <p>ImageNet-এ train করা model (1.2M images, 1000 classes) ইতিমধ্যে edges, textures, shapes চেনে। আমরা সেই <strong>শেখা features পুনরায় ব্যবহার</strong> করি নতুন task-এ — শূন্য থেকে শেখার দরকার নেই।</p>
    <pre><code># Transfer Learning-এর দুটো approach:
#
# 1. Feature Extraction (Frozen Base):
#    - Pre-trained model-এর weights freeze করো
#    - শুধু নতুন top layers train করো
#    - কম data থাকলে এটাই ভালো
#
# 2. Fine-Tuning (Partial Unfreeze):
#    - Base model-এর শেষ কিছু layer unfreeze করো
#    - পুরো model-কে নতুন data দিয়ে low learning rate-এ re-train করো
#    - বেশি data থাকলে ও higher accuracy দরকার হলে ব্যবহার করো</code></pre>
    <table>
      <thead><tr><th>Model</th><th>ImageNet Accuracy</th><th>Parameters</th><th>Speed</th><th>ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>VGG16</td><td>92.7%</td><td>138M</td><td>ধীর</td><td>Research, feature extraction</td></tr>
        <tr><td>ResNet50</td><td>93.0%</td><td>25.6M</td><td>মাঝারি</td><td>General purpose</td></tr>
        <tr><td>MobileNetV2</td><td>91.0%</td><td>3.4M</td><td>দ্রুত</td><td>Mobile, edge device</td></tr>
        <tr><td>EfficientNetB0</td><td>93.3%</td><td>5.3M</td><td>দ্রুত</td><td>State-of-the-art efficiency</td></tr>
      </tbody>
    </table>

    <h3>২. Feature Extraction — Frozen Base Model</h3>
    <pre><code">import tensorflow as tf
from tensorflow import keras
import numpy as np

# Cats vs Dogs dataset (binary classification example)
# Keras-এর built-in flowers dataset ব্যবহার করছি
IMG_SIZE   = (160, 160)
BATCH_SIZE = 32

# Data loading — নিজের data ব্যবহার করলে ImageDataGenerator ব্যবহার করো
(train_ds, val_ds, test_ds), info = tf.keras.utils.image_dataset_from_directory(
    "path/to/your/dataset",
    validation_split=0.2,
    subset="both",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
), None  # placeholder — নিচে MobileNetV2-এ CIFAR-10 ব্যবহার করেছি

# CIFAR-10 দিয়ে demonstrate করা (10-class, 32x32 images)
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# MobileNetV2: include_top=False → classifier head বাদ
base_model = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,      # ImageNet classifier head বাদ
    weights='imagenet',     # pre-trained weights
)
base_model.trainable = False  # weights freeze

# নতুন classifier head যোগ
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),  # feature map → vector
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax'),  # 10 CIFAR classes
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()
print(f"Trainable params: {model.trainable_variables.__len__()}")</code></pre>

    <h3>৩. Fine-Tuning — Base Model Unfreeze</h3>
    <pre><code">history1 = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1,
)

print(f"Feature extraction val accuracy: {max(history1.history['val_accuracy']):.4f}")

# Fine-tuning: base model-এর শেষ 30 layer unfreeze করো
base_model.trainable = True
fine_tune_from = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

print(f"Trainable layers: {sum(1 for l in base_model.layers if l.trainable)} / {len(base_model.layers)}")

# Fine-tuning-এ learning rate অনেক কম রাখতে হবে
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 10x smaller than default
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history2 = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1,
)
print(f"Fine-tuning val accuracy: {max(history2.history['val_accuracy']):.4f}")</code></pre>

    <h3>৪. Data Augmentation</h3>
    <pre><code">from tensorflow import keras

# Training data-তে artificial variation তৈরি করো → overfitting কমে
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
])

# Augmentation model-এ যোগ করা
augmented_model = keras.Sequential([
    data_augmentation,          # augmentation শুধু training-এ apply হয়
    keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet',
    ),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax'),
])
augmented_model.layers[1].trainable = False  # base frozen

augmented_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history_aug = augmented_model.fit(
    X_train, y_train,
    epochs=5, batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1,
)</code></pre>

    <h3>৫. Scratch vs Transfer Learning তুলনা</h3>
    <pre><code">import matplotlib.pyplot as plt

# Scratch model (comparison baseline)
scratch_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
scratch_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
history_scratch = scratch_model.fit(
    X_train, y_train, epochs=10, batch_size=64,
    validation_data=(X_test, y_test), verbose=0,
)

# Loss curve তুলনা
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history_scratch.history['val_accuracy'], label='Scratch CNN')
plt.plot(history1.history['val_accuracy'], label='Transfer (frozen)')
plt.xlabel('Epoch'); plt.ylabel('Val Accuracy')
plt.title('Transfer vs Scratch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history_scratch.history['val_loss'], label='Scratch CNN')
plt.plot(history1.history['val_loss'], label='Transfer (frozen)')
plt.xlabel('Epoch'); plt.ylabel('Val Loss')
plt.legend()
plt.tight_layout(); plt.show()

results = {
    'Scratch CNN (10 epochs)':          max(history_scratch.history['val_accuracy']),
    'Transfer — Feature Extraction':    max(history1.history['val_accuracy']),
    'Transfer — Fine-tuned':            max(history2.history['val_accuracy']),
}
for name, acc in results.items():
    print(f"{name:40s}: {acc:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Feature Extraction</td><td>Base model freeze, শুধু top layers train — কম data-তে দ্রুত ফলাফল</td></tr>
        <tr><td>Fine-Tuning</td><td>Base model partially unfreeze, খুব কম learning rate — accuracy আরও বাড়ে</td></tr>
        <tr><td>include_top=False</td><td>ImageNet-এর 1000-class head বাদ দিয়ে নিজের head যোগ করো</td></tr>
        <tr><td>GlobalAveragePooling2D</td><td>Feature map → 1D vector — Flatten-এর চেয়ে overfitting কম</td></tr>
        <tr><td>Data Augmentation</td><td>কম data-তে অপরিহার্য — rotation, flip, zoom দিয়ে variety বাড়ায়</td></tr>
        <tr><td>MobileNetV2</td><td>Production ও mobile-এ সেরা choice — ছোট, দ্রুত, accurate</td></tr>
      </tbody>
    </table>
  `,
};
