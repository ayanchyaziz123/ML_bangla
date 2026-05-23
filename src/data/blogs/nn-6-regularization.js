export const nn_6_regularization = {
  title: "Regularization: Dropout, Batch Norm ও Overfitting রোধ",
  description: "নিউরাল নেটওয়ার্কে overfitting সনাক্ত ও রোধ করতে Dropout, Batch Normalization, L1/L2 regularization, EarlyStopping এবং Learning Rate Scheduling বিস্তারিত।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 11,
  slug: "nn-regularization-dropout",
  content: `
    <h3>১. Overfitting কী ও কীভাবে সনাক্ত করবেন</h3>
    <p>Overfitting হলো যখন model training data খুব ভালো শেখে কিন্তু new data-তে কাজ করে না। Training accuracy বেশি, validation accuracy কম — এটাই overfitting-এর চিহ্ন।</p>
    <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Synthetic dataset: overfitting demonstrate করার জন্য
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X = X.astype('float32')

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Baseline model (overfit করবে)
def build_overfit_model():
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

baseline = build_overfit_model()
baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = baseline.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

train_acc = history_overfit.history['accuracy'][-1]
val_acc   = history_overfit.history['val_accuracy'][-1]
print(f"Baseline — Train: {train_acc:.3f}, Val: {val_acc:.3f}")
# Train: 0.99, Val: 0.72 → OVERFITTING!
</code></pre>

    <h3>২. Dropout: Random Neuron বন্ধ করা</h3>
    <p>Dropout training-এর সময় randomly কিছু neurons-এর output শূন্য করে দেয়। এটি neurons-কে forced করে independently শিখতে, একে অপরের উপর নির্ভর না করতে।</p>
    <p>Rate=0.3 মানে প্রতিটি batch-এ 30% neurons বন্ধ থাকবে। Inference-এর সময় সব neurons active থাকে কিন্তু output (1-rate) দিয়ে scale হয়।</p>
    <pre><code>def build_dropout_model(dropout_rate=0.3):
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,)),
        layers.Dropout(dropout_rate),

        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(1, activation='sigmoid')
    ])

dropout_model = build_dropout_model(dropout_rate=0.3)
dropout_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

history_dropout = dropout_model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

train_acc = history_dropout.history['accuracy'][-1]
val_acc   = history_dropout.history['val_accuracy'][-1]
print(f"Dropout(0.3) — Train: {train_acc:.3f}, Val: {val_acc:.3f}")
# Train: 0.88, Val: 0.85 → Gap অনেক কম!

# Dropout rate guidelines:
# Input layer: 0.1 - 0.2
# Hidden layers: 0.3 - 0.5
# Too high (>0.7): underfitting হবে
</code></pre>

    <h3>৩. Batch Normalization</h3>
    <p>Batch Normalization প্রতিটি layer-এর input normalize করে (mean≈0, std≈1)। এটি training accelerate করে, higher learning rate সম্ভব করে এবং slight regularization effect আছে।</p>
    <pre><code>def build_batchnorm_model():
    return keras.Sequential([
        layers.Dense(512, input_shape=(20,)),
        layers.BatchNormalization(),   # normalize layer output
        layers.Activation('relu'),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(1, activation='sigmoid')
    ])
# Note: BN -> Activation order (BN আগে, Activation পরে) কিছুটা ভালো কাজ করে

bn_model = build_batchnorm_model()
bn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='binary_crossentropy', metrics=['accuracy'])

history_bn = bn_model.fit(X_train, y_train, epochs=50, batch_size=32,
                           validation_data=(X_val, y_val), verbose=0)
print("BN model converges faster with higher lr!")
</code></pre>

    <h3>৪. L1 ও L2 Regularization</h3>
    <p>L1/L2 regularization weight penalty loss function-এ যোগ করে বড় weights discourage করে।</p>
    <p>L2 (Ridge): <strong>Total Loss = CE + λ·Σw²</strong> — weights ছোট রাখে</p>
    <p>L1 (Lasso): <strong>Total Loss = CE + λ·Σ|w|</strong> — sparse weights (অনেক zero)</p>
    <pre><code>def build_l2_model(l2_lambda=0.001):
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,),
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(1, activation='sigmoid')
    ])

# L1, L2, L1L2 combined
# regularizers.l1(0.001)
# regularizers.l2(0.001)
# regularizers.l1_l2(l1=0.001, l2=0.001)

l2_model = build_l2_model(l2_lambda=0.001)
l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_l2 = l2_model.fit(X_train, y_train, epochs=100, batch_size=32,
                            validation_data=(X_val, y_val), verbose=0)
train_acc = history_l2.history['accuracy'][-1]
val_acc   = history_l2.history['val_accuracy'][-1]
print(f"L2 Reg — Train: {train_acc:.3f}, Val: {val_acc:.3f}")
</code></pre>

    <h3>৫. EarlyStopping ও Learning Rate Scheduling</h3>
    <pre><code># EarlyStopping: val_loss উন্নতি না হলে training বন্ধ
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,  # সেরা epoch-এর weights ফিরিয়ে আনে
    min_delta=0.001,            # এর কম improvement হলে count করা হয় না
    verbose=1
)

# ReduceLROnPlateau: plateau-তে lr কমায়
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # lr = lr * factor
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Best model save
checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Combined: all best practices
best_model = keras.Sequential([
    layers.Dense(256, input_shape=(20,),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(256,
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')
])

best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy', metrics=['accuracy'])

history_best = best_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)
</code></pre>
    <table>
      <thead><tr><th>Technique</th><th>প্রভাব</th><th>কখন ব্যবহার</th><th>Parameter</th></tr></thead>
      <tbody>
        <tr><td>Dropout</td><td>Random neurons মুছে co-adaptation রোধ</td><td>Fully connected layers</td><td>rate: 0.2-0.5</td></tr>
        <tr><td>Batch Norm</td><td>Layer inputs normalize, training speed up</td><td>Deep networks, Conv layers</td><td>momentum, epsilon</td></tr>
        <tr><td>L2 Reg</td><td>Weight magnitude penalty</td><td>Weight regularization</td><td>lambda: 0.001-0.01</td></tr>
        <tr><td>L1 Reg</td><td>Sparse weights</td><td>Feature selection</td><td>lambda: 0.001</td></tr>
        <tr><td>EarlyStopping</td><td>Overfitting শুরু হলে থামায়</td><td>সবসময়</td><td>patience: 5-20</td></tr>
        <tr><td>ReduceLROnPlateau</td><td>Plateau-তে lr কমায়</td><td>Learning stagnation</td><td>factor: 0.5, patience: 5</td></tr>
      </tbody>
    </table>

    <h3>৬. Comparison Plot</h3>
    <pre><code">fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Val accuracy comparison
axes[0].plot(history_overfit.history['val_accuracy'], label='Baseline (Overfit)', color='red')
axes[0].plot(history_dropout.history['val_accuracy'], label='Dropout(0.3)', color='blue')
axes[0].plot(history_l2.history['val_accuracy'], label='L2 Reg', color='green')
axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Train-Val gap (overfitting indicator)
gap_baseline = np.array(history_overfit.history['accuracy']) - np.array(history_overfit.history['val_accuracy'])
gap_dropout  = np.array(history_dropout.history['accuracy']) - np.array(history_dropout.history['val_accuracy'])

axes[1].plot(gap_baseline, label='Baseline Gap', color='red')
axes[1].plot(gap_dropout, label='Dropout Gap', color='blue')
axes[1].set_title("Train-Val Accuracy Gap (Overfitting Indicator)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Gap")
axes[1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("regularization_comparison.png")
plt.show()
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম কীভাবে overfitting রোধ করতে হয়:</p>
    <ul>
      <li>Dropout: random neurons বন্ধ করে co-adaptation রোধ</li>
      <li>Batch Normalization: normalize + accelerate training</li>
      <li>L1/L2: weight penalty, weight-কে ছোট রাখে</li>
      <li>EarlyStopping: best epoch-এ থামে, restore_best_weights=True</li>
      <li>ReduceLROnPlateau: plateau-তে learning rate কমায়</li>
    </ul>
    <p>পরবর্তী ও শেষ ব্লগে Fashion MNIST দিয়ে একটি সম্পূর্ণ end-to-end project করব।</p>
  `,
};
