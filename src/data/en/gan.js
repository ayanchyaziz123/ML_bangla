export const ganEn = [
  {
    slug: 'gan-1-basics',
    title: 'GANs: Generator vs Discriminator Game',
    description: 'GAN intuition, minimax objective, mode collapse, vanishing gradients, and basic PyTorch GAN',
    category: 'GANs',
    content: `
<h3>The GAN Concept</h3>
<p>A GAN consists of two neural networks competing: the <strong>Generator</strong> creates fake data from random noise, and the <strong>Discriminator</strong> tries to distinguish real from fake. Training proceeds until the generator fools the discriminator.</p>
<p><strong>Minimax objective:</strong> min_G max_D [ E[log D(x)] + E[log(1-D(G(z)))] ]</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# Training loop (simplified)
G = Generator(); D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

dataset = datasets.MNIST('.', download=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(50):
    for real, _ in loader:
        real = real.view(-1, 784); batch = real.size(0)
        real_lbl = torch.ones(batch, 1); fake_lbl = torch.zeros(batch, 1)
        z = torch.randn(batch, 100); fake = G(z).detach()
        loss_D = criterion(D(real), real_lbl) + criterion(D(fake), fake_lbl)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        z = torch.randn(batch, 100)
        loss_G = criterion(D(G(z)), real_lbl)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: G={loss_G.item():.4f}, D={loss_D.item():.4f}")
</code></pre>
`
  },
  {
    slug: 'gan-2-dcgan',
    title: 'DCGAN: Convolutional GANs',
    description: 'DCGAN architecture rules, transposed convolution, batch normalization, and stable training',
    category: 'GANs',
    content: `
<h3>DCGAN Architecture Rules</h3>
<ul>
<li>Replace pooling with strided convolutions (discriminator) and transposed convolutions (generator)</li>
<li>Use Batch Normalization in both networks (except output/input layers)</li>
<li>Use ReLU in generator, LeakyReLU in discriminator</li>
<li>No fully connected layers</li>
</ul>

<pre><code class="language-python">import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, z): return self.main(z)

class DCDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
    def forward(self, x): return self.main(x).view(-1)

G = DCGenerator(); D = DCDiscriminator()
G.apply(weights_init); D.apply(weights_init)
print(G); print(D)
</code></pre>
`
  },
  {
    slug: 'gan-3-conditional',
    title: 'Conditional GAN: Label-Controlled Generation',
    description: 'Conditioning on class labels, label embedding, and Pix2Pix image-to-image translation',
    category: 'GANs',
    content: `
<h3>Conditional GAN</h3>
<p>cGAN adds class label information to both generator and discriminator, enabling controlled generation — "generate digit 7" instead of a random digit.</p>

<pre><code class="language-python">import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, n_classes=10, img_dim=784, embed_dim=50):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim), nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_embed(labels)
        x = torch.cat([z, label_embed], dim=1)
        return self.net(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_dim=784, n_classes=10, embed_dim=50):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(img_dim + embed_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_embed = self.label_embed(labels)
        x = torch.cat([x, label_embed], dim=1)
        return self.net(x)

# Generation of specific digits
G = ConditionalGenerator()
# Generate three 7s
z = torch.randn(3, 100)
labels = torch.LongTensor([7, 7, 7])
with torch.no_grad():
    fake = G(z, labels)
print(f"Generated shape: {fake.shape}")  # (3, 784)
</code></pre>
`
  },
  {
    slug: 'gan-4-wgan',
    title: 'WGAN: Wasserstein Distance for Stable Training',
    description: 'Wasserstein-1 distance, weight clipping, WGAN-GP gradient penalty, and training stability',
    category: 'GANs',
    content: `
<h3>Problems with Original GAN</h3>
<ul>
<li><strong>Mode collapse:</strong> Generator produces only a few types of outputs</li>
<li><strong>Vanishing gradients:</strong> When discriminator is too good</li>
<li><strong>Training instability:</strong> Loss doesn't correlate with quality</li>
</ul>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim

# WGAN: Critic instead of Discriminator (no sigmoid!)
class Critic(nn.Module):
    def __init__(self, img_dim=784, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)  # No sigmoid — outputs real number
        )
    def forward(self, x): return self.net(x)

def gradient_penalty(critic, real, fake):
    """WGAN-GP: penalize gradients that deviate from norm 1"""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    score = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=score, inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True
    )[0]
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# WGAN-GP Loss:
# L_critic = E[C(fake)] - E[C(real)] + lambda * GP
# L_generator = -E[C(fake)]
lambda_gp = 10
# Critic is updated n_critic=5 times per generator update
</code></pre>
`
  },
  {
    slug: 'gan-5-project',
    title: 'GAN Project: Image Generation and Applications',
    description: 'End-to-end image generation, FID score evaluation, and data augmentation with GANs',
    category: 'GANs',
    content: `
<h3>GAN Applications Overview</h3>
<table>
<tr><th>Application</th><th>Model</th><th>Use Case</th></tr>
<tr><td>Face generation</td><td>StyleGAN2</td><td>Gaming avatars, privacy</td></tr>
<tr><td>Image-to-image</td><td>Pix2Pix</td><td>Sketch → Photo, day → night</td></tr>
<tr><td>Super resolution</td><td>SRGAN</td><td>Upscale old photos</td></tr>
<tr><td>Data augmentation</td><td>DCGAN</td><td>Boost small medical datasets</td></tr>
<tr><td>Deepfake detection</td><td>Discriminator</td><td>Media verification</td></tr>
</table>

<pre><code class="language-python">import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, utils
from scipy import linalg
from torchvision.models import inception_v3

def calculate_fid(real_features, fake_features):
    """Frechet Inception Distance — lower is better"""
    mu_r = real_features.mean(0)
    mu_f = fake_features.mean(0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean = linalg.sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)

def generate_samples(generator, n_samples=1000, noise_dim=100, batch_size=64):
    """Generate samples from trained generator"""
    generator.eval()
    all_samples = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - i)
            z = torch.randn(n, noise_dim, 1, 1)
            samples = generator(z)
            all_samples.append(samples)
    return torch.cat(all_samples)

# Data augmentation workflow
def augment_with_gan(real_dataset, generator, n_synthetic=500):
    synthetic = generate_samples(generator, n_synthetic)
    print(f"Original: {len(real_dataset)}, Synthetic: {n_synthetic}, Total: {len(real_dataset)+n_synthetic}")
    return synthetic

# FID benchmarks:
# DCGAN:    ~60-80   (acceptable)
# WGAN-GP:  ~40-60   (good)
# StyleGAN2: ~3-5    (state-of-the-art)
print("FID guide: lower = more realistic images")
</code></pre>
`
  },
];
