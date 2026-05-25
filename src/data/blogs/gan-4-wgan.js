export const gan_4_wgan = {
  slug: 'gan-4-wgan',
  title: 'WGAN: Wasserstein দূরত্ব দিয়ে উন্নত ট্রেনিং',
  description: 'মূল GAN-এর সমস্যা সমাধানে Wasserstein দূরত্ব ও গ্রেডিয়েন্ট পেনাল্টি ব্যবহার করুন',
  date: 'মে ২০২৫',
  category: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক',
  readTime: 13,
  content: `
<h3>মূল GAN-এর সমস্যা</h3>
<p>মূল GAN ট্রেনিংয়ে দুটি বড় সমস্যা:</p>
<ul>
<li><strong>Mode Collapse:</strong> জেনারেটর শুধু কয়েকটি ধরনের ছবি বানায়, বৈচিত্র্য হারিয়ে যায়</li>
<li><strong>Vanishing Gradient:</strong> ডিসক্রিমিনেটর খুব ভালো হলে জেনারেটর কোনো গ্রেডিয়েন্ট পায় না</li>
</ul>
<p>এই সমস্যার মূল কারণ: JS Divergence (Jensen-Shannon) অকার্যকর হয় যখন দুটি বিতরণ ওভারল্যাপ করে না।</p>

<h3>Wasserstein দূরত্ব</h3>
<p><strong>Wasserstein-1 দূরত্ব</strong> (Earth Mover's Distance) দুটি বিতরণ মেলাতে "মাটি সরানোর" খরচ পরিমাপ করে।</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# WGAN সমীকরণ:
# min_G max_D (D(x) - D(G(z)))
# D কে "Critic" বলা হয় — 0/1 আউটপুট নয়, সংখ্যা দেয়

# Wasserstein দূরত্ব: W(p, q) = sup_{||f||_L <= 1} E[f(x)] - E[f(G(z))]
# Lipschitz শর্ত: |f(x1) - f(x2)| <= |x1 - x2|
</code></pre>

<h3>WGAN আর্কিটেকচার</h3>

<pre><code class="language-python">class Critic(nn.Module):
    """ডিসক্রিমিনেটরের বদলে Critic — sigmoid নেই"""
    def __init__(self, img_dim=784, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)  # বাস্তব সংখ্যা, সিগময়েড নেই
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_dim=784, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, img_dim),
            nn.Tanh()  # আউটপুট [-1, 1]
        )

    def forward(self, z):
        return self.net(z)
</code></pre>

<h3>WGAN ট্রেনিং — Weight Clipping</h3>

<pre><code class="language-python">def train_wgan(n_epochs=50, clip_value=0.01, n_critic=5, lr=5e-5):
    # ডেটা
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST('.', download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    noise_dim = 100
    img_dim = 784

    G = Generator(noise_dim, img_dim)
    C = Critic(img_dim)

    # WGAN: Adam বদলে RMSprop, কম লার্নিং রেট
    opt_G = optim.RMSprop(G.parameters(), lr=lr)
    opt_C = optim.RMSprop(C.parameters(), lr=lr)

    g_losses, c_losses = [], []

    for epoch in range(n_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            batch_size = real.size(0)
            real = real.view(batch_size, -1)

            # Critic কে n_critic বার আপডেট
            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_dim)
                fake = G(noise).detach()

                # Wasserstein loss
                loss_C = -(C(real).mean() - C(fake).mean())

                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

                # Weight Clipping — Lipschitz শর্ত মানতে
                for p in C.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Generator আপডেট
            noise = torch.randn(batch_size, noise_dim)
            fake = G(noise)
            loss_G = -C(fake).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        g_losses.append(loss_G.item())
        c_losses.append(-loss_C.item())  # Wasserstein দূরত্ব

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: W-Distance={-loss_C.item():.4f}")

    return G, g_losses, c_losses
</code></pre>

<h3>WGAN-GP: গ্রেডিয়েন্ট পেনাল্টি (উন্নত)</h3>

<pre><code class="language-python">def gradient_penalty(critic, real, fake, device='cpu'):
    """Weight clipping-এর বদলে গ্রেডিয়েন্ট পেনাল্টি"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    critic_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

# WGAN-GP লস:
# L_C = -E[C(x)] + E[C(G(z))] + lambda * GP
# WGAN-GP আরও স্থিতিশীল এবং weight clipping লাগে না
lambda_gp = 10
</code></pre>

<h4>WGAN vs মূল GAN তুলনা</h4>
<table>
<tr><th>দিক</th><th>মূল GAN</th><th>WGAN</th><th>WGAN-GP</th></tr>
<tr><td>লস</td><td>Binary Cross-Entropy</td><td>Wasserstein</td><td>Wasserstein + GP</td></tr>
<tr><td>Mode Collapse</td><td>ঘন ঘন</td><td>কম</td><td>বিরল</td></tr>
<tr><td>Gradient Vanishing</td><td>হ্যাঁ</td><td>না</td><td>না</td></tr>
<tr><td>ট্রেনিং স্থিতিশীলতা</td><td>কম</td><td>বেশি</td><td>সর্বোচ্চ</td></tr>
<tr><td>Hyperparameter সংবেদনশীলতা</td><td>উচ্চ</td><td>মাঝারি</td><td>কম</td></tr>
</table>

<p>WGAN-GP বর্তমানে সবচেয়ে বহুল ব্যবহৃত GAN ট্রেনিং পদ্ধতি। StyleGAN, BigGAN সহ অনেক উন্নত মডেল এটি ব্যবহার করে।</p>
`
};
