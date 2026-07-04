export const autoencoder_4_vae = {
  slug: 'autoencoder-4-vae',
  title: 'VAE: ভ্যারিয়েশনাল অটোএনকোডার',
  description: 'প্রোবাবিলিস্টিক এনকোডার, reparameterization trick, ELBO লস এবং PyTorch দিয়ে নতুন ছবি জেনারেট করার সম্পূর্ণ গাইড।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 15,
  content: `
    <h3>১. কেন VAE দরকার?</h3>
    <p>
      সাধারণ অটোএনকোডার ডেটা কম্প্রেস এবং পুনরুদ্ধার করতে পারে, কিন্তু এটি দিয়ে <em>নতুন ডেটা জেনারেট করা</em> কঠিন। কারণ লেটেন্ট স্পেসটি অসংগঠিত — বিভিন্ন অঞ্চলের মধ্যে শূন্য এলাকা থাকে। সেই শূন্য এলাকা থেকে স্যাম্পল করলে অর্থহীন আউটপুট আসে।
    </p>
    <p>
      <strong>ভ্যারিয়েশনাল অটোএনকোডার (Variational Autoencoder — VAE)</strong> এই সমস্যার সমাধান করে। এটি ২০১৩ সালে <strong>Kingma এবং Welling</strong> প্রস্তাব করেন। VAE লেটেন্ট স্পেসকে একটি সুগঠিত, সংযুক্ত প্রোবাবিলিটি বিতরণ হিসেবে মডেল করে।
    </p>
    <p>
      VAE একটি <strong>জেনারেটিভ মডেল</strong> — এটি ডেটার অন্তর্নিহিত বিতরণ শেখে এবং সেখান থেকে নতুন ডেটা স্যাম্পল করতে পারে।
    </p>

    <h3>২. VAE-এর মূল ধারণা: প্রোবাবিলিস্টিক এনকোডার</h3>
    <p>
      সাধারণ AE: <strong>z = f(x)</strong> — একটি নির্দিষ্ট বিন্দুতে ম্যাপ করে।
    </p>
    <p>
      VAE: <strong>z ~ q(z|x) = N(μ(x), σ²(x))</strong> — একটি গাউসিয়ান বিতরণে ম্যাপ করে।
    </p>
    <p>
      মানে, VAE এনকোডার ইনপুট x দিয়ে দুটি জিনিস আউটপুট করে:
      <br/> — <strong>μ (mu)</strong>: লেটেন্ট বিতরণের গড়
      <br/> — <strong>log(σ²) (log variance)</strong>: বিতরণের বিস্তার (variance-এর log)
    </p>
    <p>
      তারপর এই বিতরণ থেকে একটি z স্যাম্পল করা হয়: <strong>z ~ N(μ, σ²)</strong>
    </p>
    <p>
      এটি লেটেন্ট স্পেসকে একটি ক্রমাগত (continuous) এবং মসৃণ (smooth) অঞ্চলে পরিণত করে।
    </p>

    <h3>৩. Reparameterization Trick</h3>
    <p>
      একটি সমস্যা: z ~ N(μ, σ²) থেকে স্যাম্পলিং একটি random operation, এবং এর মাধ্যমে backpropagation করা যায় না (গ্রেডিয়েন্ট প্রবাহিত হতে পারে না)।
    </p>
    <p>
      সমাধান হলো <strong>Reparameterization Trick</strong>:
    </p>
    <p>
      পরিবর্তে লিখি: <strong>z = μ + σ · ε</strong>, যেখানে <strong>ε ~ N(0, 1)</strong>
    </p>
    <p>
      এখন randomness ε-এ আলাদা করা হয়েছে। μ এবং σ-এর মাধ্যমে গ্রেডিয়েন্ট প্রবাহিত হতে পারে কারণ এরা deterministic operations।
    </p>
    <p>
      এটি VAE-এর সবচেয়ে চমৎকার গণিতীয় কৌশল — স্যাম্পলিংকে backward pass-এর বাইরে নিয়ে যাওয়া।
    </p>

    <h3>৪. ELBO লস: Reconstruction + KL Divergence</h3>
    <p>
      VAE-এর লস ফাংশন হলো <strong>ELBO (Evidence Lower BOund)</strong> এর নেগেটিভ:
    </p>
    <p>
      <strong>L_VAE = L_reconstruction + L_KL</strong>
    </p>
    <p>
      <strong>L_reconstruction</strong> = -E[log p(x|z)] = BCE বা MSE রিকনস্ট্রাকশন লস
    </p>
    <p>
      <strong>L_KL</strong> = KL(q(z|x) || p(z)) = KL divergence প্রাইওর N(0,1) থেকে এনকোডারের বিতরণের দূরত্ব
    </p>
    <p>
      গাউসিয়ান বিতরণের জন্য KL-এর বিশ্লেষণী সমাধান:
      <br/><strong>KL = -½ · Σ(1 + log(σ²) - μ² - σ²)</strong>
    </p>
    <p>
      <strong>KL পেনাল্টির কাজ:</strong> এটি এনকোডার বিতরণকে প্রাইওর N(0,1)-এর কাছাকাছি রাখে। এতে লেটেন্ট স্পেস নিয়মিত এবং সংযুক্ত থাকে — যেকোনো জায়গা থেকে স্যাম্পল করলে অর্থপূর্ণ আউটপুট আসে।
    </p>

    <h3>৫. PyTorch দিয়ে VAE ইমপ্লিমেন্টেশন</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# ডেটা প্রস্তুত
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
x_train = train_dataset.data.reshape(-1, 784).float() / 255.0
x_test  = test_dataset.data.reshape(-1, 784).float()  / 255.0
y_test  = test_dataset.targets.numpy()

latent_dim = 2  # ভিজুয়ালাইজেশনের জন্য ২D লেটেন্ট স্পেস

def reparameterize(mu, log_var):
    """μ এবং log_var থেকে z স্যাম্পল করা (Reparameterization Trick)"""
    # N(0,1) থেকে epsilon স্যাম্পল
    epsilon = torch.randn_like(mu)
    # Reparameterization: z = mu + exp(0.5 * log_var) * epsilon
    return mu + torch.exp(0.5 * log_var) * epsilon</code></pre>

    <pre><code># এনকোডার তৈরি
class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        # দুটি আউটপুট: mu এবং log_var
        self.z_mean    = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        # Reparameterization trick দিয়ে z স্যাম্পল
        z = reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

encoder = Encoder(latent_dim)
print(encoder)</code></pre>

    <pre><code># ডিকোডার তৈরি
class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.out = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.out(z))

decoder = Decoder(latent_dim)
print(decoder)</code></pre>

    <pre><code># সম্পূর্ণ VAE মডেল
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

def train_step(vae, data, optimizer):
    optimizer.zero_grad()
    reconstruction, z_mean, z_log_var = vae(data)

    # রিকনস্ট্রাকশন লস (BCE)
    recon_loss = torch.mean(
        torch.sum(
            F.binary_cross_entropy(reconstruction, data, reduction='none'),
            dim=-1
        )
    )

    # KL Divergence লস
    kl_loss = -0.5 * torch.mean(
        torch.sum(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp(),
            dim=1
        )
    )

    total_loss = recon_loss + kl_loss

    total_loss.backward()
    optimizer.step()

    return {
        'loss': total_loss.item(),
        'reconstruction_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
    }

vae = VAE(encoder, decoder)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=128, shuffle=True)

for epoch in range(50):
    vae.train()
    epoch_loss = epoch_recon = epoch_kl = 0.0
    for xb, _ in train_loader:
        metrics = train_step(vae, xb, optimizer)
        epoch_loss  += metrics['loss'] * xb.size(0)
        epoch_recon += metrics['reconstruction_loss'] * xb.size(0)
        epoch_kl    += metrics['kl_loss'] * xb.size(0)
    n = len(train_loader.dataset)
    print(f"Epoch {epoch+1}: loss={epoch_loss/n:.4f}, "
          f"reconstruction_loss={epoch_recon/n:.4f}, kl_loss={epoch_kl/n:.4f}")</code></pre>

    <h3>৬. লেটেন্ট স্পেস ভিজুয়ালাইজেশন</h3>
    <pre><code># ২D লেটেন্ট স্পেস ভিজুয়ালাইজ করা
def plot_latent_space(encoder, data, labels, n=10000):
    encoder.eval()
    with torch.no_grad():
        z_means, _, _ = encoder(data[:n])
    z_means = z_means.numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_means[:, 0], z_means[:, 1],
        c=labels[:n], cmap='tab10', alpha=0.6, s=8
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('z[0]', fontsize=13)
    plt.ylabel('z[1]', fontsize=13)
    plt.title('VAE Latent Space — MNIST', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig('vae_latent_space.png', dpi=150)
    plt.show()

plot_latent_space(encoder, x_test, y_test)</code></pre>

    <pre><code># লেটেন্ট স্পেস থেকে নতুন ছবি জেনারেট করা
def plot_latent_grid(decoder, n=15, figsize=15):
    """লেটেন্ট স্পেসের একটি গ্রিড থেকে ছবি জেনারেট"""
    scale = 3.0
    figure = np.zeros((28 * n, 28 * n))

    # গ্রিড তৈরি: z[0] এবং z[1] সমানভাবে ভাগ
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    decoder.eval()
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32)
            with torch.no_grad():
                x_decoded = decoder(z_sample).numpy()
            digit = x_decoded[0].reshape(28, 28)
            figure[i * 28:(i + 1) * 28,
                   j * 28:(j + 1) * 28] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='gray')
    plt.title('Generated Digits from Latent Space Grid', fontsize=16)
    plt.axis('off')
    plt.savefig('vae_generated_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_latent_grid(decoder, n=15)</code></pre>

    <h3>৭. লেটেন্ট স্পেসে ইন্টারপোলেশন</h3>
    <pre><code># দুটি ডিজিটের মধ্যে মসৃণ ট্রানজিশন
def interpolate_digits(encoder, decoder, x_test, y_test,
                       digit_a=3, digit_b=8, steps=10):
    # দুটি ভিন্ন ডিজিটের উদাহরণ বের করা
    idx_a = np.where(y_test == digit_a)[0][0]
    idx_b = np.where(y_test == digit_b)[0][0]

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # লেটেন্ট ভেক্টর এনকোড করা
        z_a, _, _ = encoder(x_test[idx_a:idx_a+1])
        z_b, _, _ = encoder(x_test[idx_b:idx_b+1])

        # ইন্টারপোলেশন
        alphas = np.linspace(0, 1, steps)
        z_interp = torch.cat([alpha * z_b + (1 - alpha) * z_a for alpha in alphas], dim=0)

        images = decoder(z_interp).numpy()

    plt.figure(figsize=(20, 3))
    for i in range(steps):
        ax = plt.subplot(1, steps, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(f'{digit_a}', fontsize=12)
        elif i == steps - 1:
            plt.title(f'{digit_b}', fontsize=12)

    plt.suptitle(f'Interpolation: {digit_a} → {digit_b}', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('vae_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()

interpolate_digits(encoder, decoder, x_test, y_test, digit_a=3, digit_b=8)</code></pre>

    <h3>৮. নতুন ছবি জেনারেট করা</h3>
    <pre><code># র্যান্ডম স্যাম্পলিং দিয়ে নতুন ছবি তৈরি
def generate_new_images(decoder, n_samples=20, latent_dim=2):
    # প্রাইওর N(0,1) থেকে স্যাম্পল
    z_random = torch.randn(n_samples, latent_dim)
    decoder.eval()
    with torch.no_grad():
        generated = decoder(z_random).numpy()

    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        ax = plt.subplot(2, n_samples // 2, i + 1)
        plt.imshow(generated[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle('VAE Generated Digits (Sampled from N(0,1))', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_generated.png', dpi=150)
    plt.show()

generate_new_images(decoder, n_samples=20, latent_dim=2)</code></pre>

    <h3>৯. উচ্চতর লেটেন্ট ডিমেনশন</h3>
    <pre><code># ২D লেটেন্ট দিয়ে ভিজুয়ালাইজ সহজ, কিন্তু মান কম
# বেশি latent_dim ব্যবহার করলে মান ভালো হয়

latent_dim_high = 16

encoder_h = Encoder(latent_dim_high)
decoder_h = Decoder(latent_dim_high)
vae_h = VAE(encoder_h, decoder_h)
optimizer_h = torch.optim.Adam(vae_h.parameters(), lr=1e-3)

for epoch in range(50):
    vae_h.train()
    for xb, _ in train_loader:
        train_step(vae_h, xb, optimizer_h)

# তুলনা
vae.eval()
vae_h.eval()
with torch.no_grad():
    recon_2d, _, _  = vae(x_test)
    mse_2d = torch.mean((x_test - recon_2d) ** 2).item()

    recon_16d, _, _ = vae_h(x_test)
    mse_16d = torch.mean((x_test - recon_16d) ** 2).item()

print(f"VAE (latent_dim=2):  MSE = {mse_2d:.4f}")
print(f"VAE (latent_dim=16): MSE = {mse_16d:.4f}")</code></pre>

    <h3>১০. VAE বনাম Standard AE</h3>
    <p>
      <strong>লেটেন্ট স্পেসের কাঠামো:</strong> Standard AE-এ লেটেন্ট স্পেস অনিয়মিত। VAE-এ এটি N(0,1)-এর কাছাকাছি, তাই যেকোনো জায়গা থেকে স্যাম্পল করলে ভালো আউটপুট আসে।
    </p>
    <p>
      <strong>জেনারেশন:</strong> Standard AE দিয়ে নতুন ডেটা তৈরি করা কঠিন। VAE স্বাভাবিকভাবেই জেনারেটিভ।
    </p>
    <p>
      <strong>ইন্টারপোলেশন:</strong> VAE-এ দুটি বিন্দুর মধ্যে মসৃণ ট্রানজিশন সম্ভব।
    </p>
    <p>
      <strong>ট্রেডঅফ:</strong> VAE-এর রিকনস্ট্রাকশন কিছুটা অস্পষ্ট (blurry) হতে পারে কারণ এটি বিতরণের গড় মডেল করে। GAN আরও তীক্ষ্ণ ছবি তৈরি করতে পারে।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      VAE একটি শক্তিশালী জেনারেটিভ মডেল যা অটোএনকোডার এবং বায়েজিয়ান ইনফারেন্সকে একত্রিত করে। প্রোবাবিলিস্টিক এনকোডার (μ, σ), reparameterization trick এবং ELBO লস (reconstruction + KL) — এই তিনটি উপাদান মিলে VAE কাজ করে। এটি নতুন ডেটা জেনারেট করতে, ইন্টারপোলেশন করতে এবং ডেটার অন্তর্নিহিত বিতরণ বুঝতে সাহায্য করে।
    </p>
  `
};
