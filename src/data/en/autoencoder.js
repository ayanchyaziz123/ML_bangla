export const autoencoderEn = [
  {
    slug: 'autoencoder-1-basics',
    title: 'Autoencoders: Learning Through Compression',
    description: 'Encoder-bottleneck-decoder architecture, reconstruction loss, and latent space visualization',
    category: 'Autoencoders',
    content: `
<h3>What is an Autoencoder?</h3>
<p>An autoencoder is a neural network that learns to compress data into a low-dimensional <strong>latent space</strong>, then reconstruct it. The bottleneck forces the network to learn the most important features.</p>

<pre><code class="language-python">import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Build autoencoder
encoding_dim = 32  # Bottleneck

inputs = keras.Input(shape=(784,))
encoded = keras.layers.Dense(128, activation='relu')(inputs)
encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
decoded = keras.layers.Dense(128, activation='relu')(encoded)
decoded = keras.layers.Dense(784, activation='sigmoid')(decoded)

autoencoder = keras.Model(inputs, decoded)
encoder = keras.Model(inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                validation_data=(X_test, X_test), verbose=0)

# Visualize reconstructions
X_pred = autoencoder.predict(X_test[:10])
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[1, i].imshow(X_pred[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off'); axes[1, i].axis('off')
plt.suptitle('Original (top) vs Reconstructed (bottom)')
plt.show()
</code></pre>
`
  },
  {
    slug: 'autoencoder-2-denoising',
    title: 'Denoising Autoencoders',
    description: 'Training autoencoders on corrupted inputs to learn robust representations',
    category: 'Autoencoders',
    content: `
<h3>Denoising Principle</h3>
<p>A Denoising Autoencoder (DAE) receives corrupted input and must reconstruct the clean version. This forces the network to learn more robust features — it can't simply memorize.</p>

<pre><code class="language-python">import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

def add_noise(X, noise_factor=0.3):
    noisy = X + noise_factor * np.random.randn(*X.shape)
    return np.clip(noisy, 0., 1.)

X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)

# Build DAE
autoencoder = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(784, activation='sigmoid'),
    keras.layers.Reshape((28, 28))
])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_noisy, X_train, epochs=30, batch_size=256,
                validation_data=(X_test_noisy, X_test), verbose=0)

# Visualize denoising
X_pred = autoencoder.predict(X_test_noisy[:8])
fig, axes = plt.subplots(3, 8, figsize=(15, 5))
for i in range(8):
    axes[0, i].imshow(X_test[i], cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(X_test_noisy[i], cmap='gray'); axes[1, i].axis('off')
    axes[2, i].imshow(X_pred[i], cmap='gray'); axes[2, i].axis('off')
plt.show()
</code></pre>
`
  },
  {
    slug: 'autoencoder-3-sparse',
    title: 'Sparse Autoencoders: Feature Learning',
    description: 'L1 regularization and KL divergence sparsity constraints for better representations',
    category: 'Autoencoders',
    content: `
<h3>Why Sparse?</h3>
<p>Sparse autoencoders encourage most hidden units to be inactive (near zero) for any given input. This forces the network to use distinct features for different inputs — analogous to how neurons in the brain work.</p>

<pre><code class="language-python">from tensorflow import keras
import tensorflow as tf

class SparseRegularizer(keras.regularizers.Regularizer):
    def __init__(self, rho=0.05, beta=3.0):
        self.rho = rho  # Target sparsity
        self.beta = beta  # Penalty weight

    def __call__(self, x):
        rho_hat = tf.reduce_mean(tf.math.sigmoid(x), axis=0)
        kl_div = (self.rho * tf.math.log(self.rho / (rho_hat + 1e-8)) +
                  (1 - self.rho) * tf.math.log((1 - self.rho) / (1 - rho_hat + 1e-8)))
        return self.beta * tf.reduce_sum(kl_div)

(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0

inputs = keras.Input(shape=(784,))
encoded = keras.layers.Dense(64, activation='sigmoid',
                              activity_regularizer=SparseRegularizer(rho=0.05))(inputs)
decoded = keras.layers.Dense(784, activation='sigmoid')(encoded)

sparse_ae = keras.Model(inputs, decoded)
sparse_ae.compile(optimizer='adam', loss='mse')
sparse_ae.fit(X_train, X_train, epochs=30, batch_size=256, verbose=0)

# Measure sparsity
import numpy as np
encoder_model = keras.Model(inputs, encoded)
activations = encoder_model.predict(X_train[:1000])
sparsity = (activations < 0.1).mean()
print(f"Fraction of near-zero activations: {sparsity:.3f}")
</code></pre>
`
  },
  {
    slug: 'autoencoder-4-vae',
    title: 'Variational Autoencoders (VAE)',
    description: 'Probabilistic encoder, reparameterization trick, ELBO loss, and new sample generation',
    category: 'Autoencoders',
    content: `
<h3>VAE vs Standard Autoencoder</h3>
<p>Standard autoencoders learn a discrete latent code — the space between codes may produce garbage. VAEs learn a <strong>continuous probability distribution</strong> over latent space, enabling smooth interpolation and generation.</p>

<pre><code class="language-python">import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 2

# Encoder
encoder_inputs = keras.Input(shape=(784,))
h = keras.layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = keras.layers.Dense(latent_dim)(h)
z_log_var = keras.layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
h_dec = keras.layers.Dense(256, activation='relu')(latent_inputs)
outputs = keras.layers.Dense(784, activation='sigmoid')(h_dec)
decoder = keras.Model(latent_inputs, outputs)

# VAE with custom loss
class VAE(keras.Model):
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction)) * 784
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
</code></pre>
`
  },
  {
    slug: 'autoencoder-5-project',
    title: 'Autoencoder Project: Anomaly Detection',
    description: 'Using reconstruction error to detect anomalies in credit card fraud and sensor data',
    category: 'Autoencoders',
    content: `
<h3>Anomaly Detection with Autoencoders</h3>
<p>Train the autoencoder only on normal data. When an anomaly is presented, the autoencoder fails to reconstruct it accurately, yielding a high reconstruction error — our anomaly signal.</p>

<pre><code class="language-python">import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# Simulate sensor data
np.random.seed(42)
n_normal, n_anomaly = 5000, 250
X_normal = np.random.multivariate_normal([0, 0, 0], np.eye(3), n_normal)
X_anomaly = np.random.multivariate_normal([5, 5, 5], np.eye(3), n_anomaly)
X = np.vstack([X_normal, X_anomaly])
y = np.array([0]*n_normal + [1]*n_anomaly)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train on normal data only
X_train = X_scaled[:n_normal]

# Autoencoder
ae = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3)
])
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_train, X_train, epochs=100, batch_size=64, validation_split=0.1, verbose=0)

# Reconstruction error as anomaly score
X_pred = ae.predict(X_scaled)
recon_error = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

auc = roc_auc_score(y, recon_error)
print(f"ROC-AUC: {auc:.4f}")

threshold = np.percentile(recon_error[:n_normal], 95)
y_pred = (recon_error > threshold).astype(int)
print(classification_report(y, y_pred, target_names=['Normal', 'Anomaly']))
</code></pre>
`
  },
];
