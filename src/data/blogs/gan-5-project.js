export const gan_5_project = {
  slug: 'gan-5-project',
  title: 'GAN প্রজেক্ট: ছবি জেনারেশন ও অ্যাপ্লিকেশন',
  description: 'DCGAN দিয়ে মুখের ছবি তৈরি, FID স্কোর মূল্যায়ন এবং ডেটা অগমেন্টেশনে GAN ব্যবহার',
  date: 'মে ২০২৫',
  category: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক',
  readTime: 15,
  content: `
<h3>প্রজেক্ট সংক্ষিপ্ত বিবরণ</h3>
<p>এই প্রজেক্টে তিনটি অ্যাপ্লিকেশন দেখব:</p>
<ol>
<li>CelebA ডেটাসেটে DCGAN দিয়ে মানুষের মুখ তৈরি</li>
<li>FID স্কোর দিয়ে মান মূল্যায়ন</li>
<li>চিকিৎসা ছবিতে ডেটা অগমেন্টেশন</li>
</ol>

<h3>DCGAN দিয়ে মুখ জেনারেশন</h3>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class DCGenerator(nn.Module):
    def __init__(self, noise_dim=100, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # noise_dim x 1 x 1 → 512 x 4 x 4
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4 → 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8 → 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16 → 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32 → channels x 64 x 64
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)
</code></pre>

<h3>ট্রেনিং পাইপলাইন</h3>

<pre><code class="language-python">def train_dcgan(dataloader, n_epochs=100, noise_dim=100, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = DCGenerator(noise_dim).to(device)
    D = DCDiscriminator().to(device)

    # ওজন প্রাথমিককরণ (DCGAN পেপার থেকে)
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
    g_losses, d_losses = [], []

    for epoch in range(n_epochs):
        for real, _ in dataloader:
            real = real.to(device)
            batch_size = real.size(0)

            # ডিসক্রিমিনেটর
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake = G(noise).detach()

            loss_D = criterion(D(real), real_label) + criterion(D(fake), fake_label)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # জেনারেটর
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake = G(noise)
            loss_G = criterion(D(fake), real_label)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: G={loss_G.item():.4f}, D={loss_D.item():.4f}")
            # নমুনা ছবি সেভ
            with torch.no_grad():
                fake = G(fixed_noise)
            utils.save_image(fake, f'generated_epoch_{epoch+1}.png',
                           normalize=True, nrow=8)

    return G, D
</code></pre>

<h3>FID স্কোর — মান মূল্যায়ন</h3>

<pre><code class="language-python">from scipy import linalg
from torchvision.models import inception_v3

def calculate_fid(real_images, fake_images, batch_size=50):
    """
    FID = ||mu_r - mu_f||^2 + Tr(sigma_r + sigma_f - 2*sqrt(sigma_r*sigma_f))
    কম FID = আরও বাস্তবসম্মত ছবি
    """
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval()

    def get_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                feat = inception(batch)
            features.append(feat.numpy())
        return np.concatenate(features)

    real_feat = get_features(real_images)
    fake_feat = get_features(fake_images)

    mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
    mu_f, sigma_f = fake_feat.mean(0), np.cov(fake_feat, rowvar=False)

    diff = mu_r - mu_f
    covmean = linalg.sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return fid

# FID রেফারেন্স মান:
# DCGAN CelebA: ~60-80  (ভালো)
# WGAN-GP:      ~40-60  (আরও ভালো)
# StyleGAN2:    ~3-5    (অসাধারণ)
print("FID স্কোর: কম হলে ভালো (মানবিক মুখ ≈ 0)")
</code></pre>

<h3>ডেটা অগমেন্টেশন অ্যাপ্লিকেশন</h3>

<pre><code class="language-python"># চিকিৎসা ছবিতে কম ডেটার সমস্যা সমাধান
# GAN দিয়ে কৃত্রিম MRI/X-ray তৈরি করে প্রশিক্ষণ ডেটা বাড়ানো

# সাধারণ কর্মপ্রবাহ:
# ১. আসল ডেটায় DCGAN/WGAN-GP ট্রেন করুন
# ২. সিনথেটিক ছবি তৈরি করুন
# ৩. আসল + সিনথেটিক ডেটায় ক্লাসিফায়ার ট্রেন করুন
# ৪. FID দিয়ে সিনথেটিক মান যাচাই করুন

def augment_dataset_with_gan(generator, real_dataset, n_synthetic=1000):
    """GAN দিয়ে ডেটাসেট বড় করা"""
    noise_dim = 100
    synthetic_images = []

    generator.eval()
    with torch.no_grad():
        for _ in range(0, n_synthetic, 64):
            noise = torch.randn(64, noise_dim, 1, 1)
            fake = generator(noise)
            synthetic_images.append(fake)

    synthetic = torch.cat(synthetic_images)[:n_synthetic]
    print(f"তৈরি: {n_synthetic}টি সিনথেটিক ছবি")
    print(f"মূল ডেটা: {len(real_dataset)}, মোট: {len(real_dataset) + n_synthetic}")
    return synthetic
</code></pre>

<h4>GAN অ্যাপ্লিকেশন সারাংশ</h4>
<table>
<tr><th>অ্যাপ্লিকেশন</th><th>মডেল</th><th>ব্যবহার</th></tr>
<tr><td>মুখ জেনারেশন</td><td>StyleGAN2</td><td>গেম চরিত্র, অ্যাভাটার</td></tr>
<tr><td>ছবি-থেকে-ছবি</td><td>Pix2Pix</td><td>ডিজাইন, চিকিৎসা</td></tr>
<tr><td>সুপার রেজোলিউশন</td><td>SRGAN</td><td>পুরনো ছবি উন্নত করা</td></tr>
<tr><td>ডেটা অগমেন্টেশন</td><td>DCGAN</td><td>কম ডেটায় মডেল উন্নত করা</td></tr>
<tr><td>Deepfake শনাক্তকরণ</td><td>Discriminator</td><td>মিডিয়া যাচাই</td></tr>
</table>
`
};
