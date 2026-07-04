export const autoencoderEn = [
  {
    slug: 'autoencoder-1-basics',
    title: 'Autoencoders: Learning Through Compression',
    description: 'Encoder-bottleneck-decoder architecture, reconstruction loss, and latent space visualization',
    category: 'Autoencoders',
    content: `
<h3>What is an Autoencoder?</h3>
<p>An autoencoder is a neural network that learns to compress data into a low-dimensional <strong>latent space</strong>, then reconstruct it. The bottleneck forces the network to learn the most important features.</p>

<pre><code class="language-python">import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_data.data.float() / 255.0
X_test = test_data.data.float() / 255.0
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Build autoencoder
encoding_dim = 32  # Bottleneck

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super().__init__()
        self.enc_fc1 = nn.Linear(784, 128)
        self.enc_fc2 = nn.Linear(128, encoding_dim)
        self.dec_fc1 = nn.Linear(encoding_dim, 128)
        self.dec_fc2 = nn.Linear(128, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.enc_fc1(x))
        return self.relu(self.enc_fc2(x))

    def decode(self, z):
        z = self.relu(self.dec_fc1(z))
        return self.sigmoid(self.dec_fc2(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

autoencoder = Autoencoder(encoding_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=256, shuffle=True)

for epoch in range(50):
    autoencoder.train()
    for xb, _ in train_loader:
        optimizer.zero_grad()
        reconstructed = autoencoder(xb)
        loss = criterion(reconstructed, xb)
        loss.backward()
        optimizer.step()

# Visualize reconstructions
autoencoder.eval()
with torch.no_grad():
    X_pred = autoencoder(X_test[:10]).numpy()
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(X_test[i].numpy().reshape(28, 28), cmap='gray')
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

<pre><code class="language-python">import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_data.data.float() / 255.0
X_test = test_data.data.float() / 255.0

def add_noise(X, noise_factor=0.3):
    noisy = X + noise_factor * torch.randn_like(X)
    return torch.clamp(noisy, 0., 1.)

X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)

# Build DAE
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x.view(-1, 28, 28)

autoencoder = DenoisingAutoencoder()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train_noisy, X_train), batch_size=256, shuffle=True)

for epoch in range(30):
    autoencoder.train()
    for noisy_xb, clean_xb in train_loader:
        optimizer.zero_grad()
        reconstructed = autoencoder(noisy_xb)
        loss = criterion(reconstructed, clean_xb)
        loss.backward()
        optimizer.step()

# Visualize denoising
autoencoder.eval()
with torch.no_grad():
    X_pred = autoencoder(X_test_noisy[:8]).numpy()
fig, axes = plt.subplots(3, 8, figsize=(15, 5))
for i in range(8):
    axes[0, i].imshow(X_test[i].numpy(), cmap='gray'); axes[0, i].axis('off')
    axes[1, i].imshow(X_test_noisy[i].numpy(), cmap='gray'); axes[1, i].axis('off')
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

<pre><code class="language-python">import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def kl_divergence_sparsity(rho, rho_hat):
    # Target sparsity rho, actual mean activation rho_hat
    rho_hat = torch.clamp(rho_hat, 1e-8, 1 - 1e-8)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

class SparseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_enc = nn.Linear(784, 64)
        self.fc_dec = nn.Linear(64, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = self.sigmoid(self.fc_enc(x))
        decoded = self.sigmoid(self.fc_dec(encoded))
        return decoded, encoded

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
X_train = train_data.data.reshape(-1, 784).float() / 255.0

sparse_ae = SparseAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(sparse_ae.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=256, shuffle=True)

rho, beta = 0.05, 3.0  # target sparsity, penalty weight

for epoch in range(30):
    sparse_ae.train()
    for xb, _ in train_loader:
        optimizer.zero_grad()
        reconstructed, encoded = sparse_ae(xb)
        recon_loss = criterion(reconstructed, xb)
        rho_hat = torch.mean(encoded, dim=0)
        sparsity_loss = beta * torch.sum(kl_divergence_sparsity(rho, rho_hat))
        loss = recon_loss + sparsity_loss
        loss.backward()
        optimizer.step()

# Measure sparsity
sparse_ae.eval()
with torch.no_grad():
    _, activations = sparse_ae(X_train[:1000])
sparsity = (activations &lt; 0.1).float().mean()
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

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 2

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)

    def reparameterize(self, mu, logvar):
        # z = mu + exp(0.5 * logvar) * epsilon
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * epsilon

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

encoder = Encoder(latent_dim)

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.out = nn.Linear(256, 784)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.out(h))

decoder = Decoder(latent_dim)

# VAE with explicit training step
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

vae = VAE(encoder, decoder)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def train_step(data):
    optimizer.zero_grad()
    reconstruction, z_mean, z_log_var = vae(data)
    recon_loss = F.binary_cross_entropy(reconstruction, data, reduction='mean') * 784
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    total_loss = recon_loss + kl_loss
    total_loss.backward()
    optimizer.step()
    return {"loss": total_loss.item(), "recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()}
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
X_train = torch.tensor(X_scaled[:n_normal], dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=64, shuffle=True)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_fc1 = nn.Linear(3, 8)
        self.enc_fc2 = nn.Linear(8, 2)
        self.dec_fc1 = nn.Linear(2, 8)
        self.dec_fc2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc_fc1(x))
        x = self.relu(self.enc_fc2(x))
        x = self.relu(self.dec_fc1(x))
        return self.dec_fc2(x)

ae = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)

for epoch in range(100):
    ae.train()
    for xb, _ in train_loader:
        optimizer.zero_grad()
        reconstructed = ae(xb)
        loss = criterion(reconstructed, xb)
        loss.backward()
        optimizer.step()

# Reconstruction error as anomaly score
ae.eval()
with torch.no_grad():
    X_pred = ae(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
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
