export const autoencoder_3_sparse = {
  slug: 'autoencoder-3-sparse',
  title: 'স্পার্স অটোএনকোডার: কম অ্যাক্টিভেশন, বেশি শেখা',
  description: 'L1 রেগুলারাইজেশন এবং KL divergence দিয়ে স্পার্স অটোএনকোডার তৈরি, ফিচার লার্নিং এবং সাধারণ AE-এর সাথে তুলনা।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 12,
  content: `
    <h3>১. স্পার্সিটি কী এবং কেন দরকার?</h3>
    <p>
      মানুষের মস্তিষ্কে যেকোনো একটি উদ্দীপনার প্রতি মাত্র কিছু নিউরন সক্রিয় হয়, বাকিগুলো নিষ্ক্রিয় থাকে। এই ধারণাটি নিউরোসায়েন্স থেকে নেওয়া এবং মেশিন লার্নিংয়ে প্রয়োগ করা হয়েছে। এটাই হলো <strong>স্পার্সিটি (Sparsity)</strong>।
    </p>
    <p>
      একটি স্তরে যদি বেশিরভাগ নিউরন ০ অথবা ০ এর কাছাকাছি মান পায়, তাহলে সেই স্তরটি <em>স্পার্স</em>। <strong>স্পার্স অটোএনকোডার (Sparse Autoencoder — SAE)</strong> লেটেন্ট স্তরে স্পার্সিটি নিশ্চিত করতে একটি পেনাল্টি যোগ করে।
    </p>
    <p>
      কেন স্পার্সিটি দরকার? কারণ:
      <br/><strong>১)</strong> এটি নেটওয়ার্ককে ওভারকমপ্লিট (overcomplete) লেটেন্ট স্পেস ব্যবহার করতে দেয় — মানে লেটেন্ট ডিমেনশন ইনপুটের চেয়ে বড় হতে পারে।
      <br/><strong>২)</strong> এটি নেটওয়ার্ককে আরও ডিসেন্ট্যাংগেলড (disentangled) ফিচার শিখতে সাহায্য করে।
      <br/><strong>৩)</strong> প্রতিটি ফিচার নিউরন ডেটার একটি নির্দিষ্ট দিক উপস্থাপন করে।
    </p>

    <h3>২. L1 রেগুলারাইজেশন দিয়ে স্পার্সিটি</h3>
    <p>
      সবচেয়ে সরল পদ্ধতি হলো লেটেন্ট অ্যাক্টিভেশনে <strong>L1 রেগুলারাইজেশন</strong> যোগ করা।
    </p>
    <p>
      মোট লস: <strong>L_total = L_reconstruction + λ · ||z||₁</strong>
    </p>
    <p>
      এখানে:
      <br/> — L_reconstruction হলো রিকনস্ট্রাকশন লস (BCE বা MSE)
      <br/> — ||z||₁ = Σ|zᵢ| হলো লেটেন্ট ভেক্টরের L1 নর্ম
      <br/> — λ (lambda) হলো স্পার্সিটির শক্তি নিয়ন্ত্রণকারী হাইপারপ্যারামিটার
    </p>
    <p>
      L1 পেনাল্টি ছোট অ্যাক্টিভেশনগুলোকে একদম ০ এ নামিয়ে আনে, ফলে প্রকৃত স্পার্সিটি তৈরি হয়। (L2 পেনাল্টি ছোট করে কিন্তু ০ করে না।)
    </p>

    <h3>৩. KL Divergence দিয়ে স্পার্সিটি পেনাল্টি</h3>
    <p>
      আরও পরিশীলিত পদ্ধতি হলো <strong>KL Divergence</strong> ব্যবহার করে একটি নির্দিষ্ট গড় অ্যাক্টিভেশন (ρ) নিশ্চিত করা।
    </p>
    <p>
      ধরুন আমরা চাই প্রতিটি নিউরনের গড় অ্যাক্টিভেশন ρ = ০.০৫ (মাত্র ৫%)। প্রতিটি নিউরনের প্রকৃত গড় অ্যাক্টিভেশন হলো ρ̂ⱼ।
    </p>
    <p>
      KL স্পার্সিটি পেনাল্টি: <strong>KL(ρ || ρ̂ⱼ) = ρ·log(ρ/ρ̂ⱼ) + (1-ρ)·log((1-ρ)/(1-ρ̂ⱼ))</strong>
    </p>
    <p>
      মোট লস: <strong>L_total = L_reconstruction + β · Σⱼ KL(ρ || ρ̂ⱼ)</strong>
    </p>
    <p>
      যদি ρ̂ⱼ = ρ হয়, তাহলে KL = ০। যদি ρ̂ⱼ ρ থেকে বিচ্যুত হয়, পেনাল্টি বাড়ে। এটি প্রতিটি নিউরনকে লক্ষ্য অ্যাক্টিভেশনের দিকে ঠেলে দেয়।
    </p>

    <h3>৪. PyTorch দিয়ে L1 স্পার্স অটোএনকোডার</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ডেটা প্রস্তুত
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
x_train = train_dataset.data.reshape(-1, 784).float() / 255.0
x_test  = test_dataset.data.reshape(-1, 784).float()  / 255.0

# L1 রেগুলারাইজেশন সহ স্পার্স অটোএনকোডার
# encoding_dim > input_dim হতে পারে (overcomplete)
encoding_dim = 1024  # ইনপুট ৭৮৪ এর চেয়ে বড়!
l1_lambda = 1e-5  # λ = 0.00001

class SparseAutoencoderL1(nn.Module):
    def __init__(self, encoding_dim=1024):
        super().__init__()
        self.fc_enc = nn.Linear(784, encoding_dim)
        self.fc_dec = nn.Linear(encoding_dim, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # activity_regularizer-এর সমতুল্য পেনাল্টি ট্রেনিং লুপে যোগ করা হবে
        encoded = self.relu(self.fc_enc(x))
        decoded = self.sigmoid(self.fc_dec(encoded))
        return decoded, encoded

sparse_ae_l1 = SparseAutoencoderL1(encoding_dim)
criterion = nn.BCELoss()
optimizer_l1 = torch.optim.Adam(sparse_ae_l1.parameters(), lr=0.001)
print(sparse_ae_l1)</code></pre>

    <pre><code># ট্রেনিং
train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test, x_test),  batch_size=256)

history_l1 = {'loss': [], 'val_loss': []}

for epoch in range(40):
    sparse_ae_l1.train()
    train_loss = 0.0
    for xb, _ in train_loader:
        optimizer_l1.zero_grad()
        reconstructed, encoded = sparse_ae_l1(xb)
        recon_loss = criterion(reconstructed, xb)
        # activity_regularizer-এর সমতুল্য: অ্যাক্টিভেশনে সরাসরি L1 পেনাল্টি
        l1_penalty = l1_lambda * torch.sum(torch.abs(encoded))
        loss = recon_loss + l1_penalty
        loss.backward()
        optimizer_l1.step()
        train_loss += recon_loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    sparse_ae_l1.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, _ in test_loader:
            reconstructed, _ = sparse_ae_l1(xb)
            val_loss += criterion(reconstructed, xb).item() * xb.size(0)
    val_loss /= len(test_loader.dataset)

    history_l1['loss'].append(train_loss)
    history_l1['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")</code></pre>

    <h3>৫. KL Divergence স্পার্সিটি কাস্টম লেয়ার</h3>
    <pre><code>import torch

class KLSparsityRegularizer:
    """KL Divergence ভিত্তিক স্পার্সিটি রেগুলারাইজার"""

    def __init__(self, rho=0.05, beta=3.0):
        self.rho  = rho    # টার্গেট স্পার্সিটি (গড় অ্যাক্টিভেশন)
        self.beta = beta   # পেনাল্টির শক্তি

    def __call__(self, activations):
        # ব্যাচে গড় অ্যাক্টিভেশন
        rho_hat = torch.mean(activations, dim=0)
        rho_hat = torch.clamp(rho_hat, 1e-7, 1.0 - 1e-7)

        rho = self.rho
        # KL(rho || rho_hat)
        kl = rho * torch.log(rho / rho_hat) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return self.beta * torch.sum(kl)</code></pre>

    <pre><code># KL স্পার্স অটোএনকোডার তৈরি
class SparseAutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_enc = nn.Linear(784, 512)
        self.fc_dec = nn.Linear(512, 784)
        self.sigmoid = nn.Sigmoid()   # sigmoid যাতে অ্যাক্টিভেশন ০-১ এর মধ্যে থাকে

    def forward(self, x):
        encoded = self.sigmoid(self.fc_enc(x))
        decoded = self.sigmoid(self.fc_dec(encoded))
        return decoded, encoded

sparse_ae_kl = SparseAutoencoderKL()
criterion_kl = nn.BCELoss()
optimizer_kl = torch.optim.Adam(sparse_ae_kl.parameters(), lr=0.001)
kl_regularizer = KLSparsityRegularizer(rho=0.05, beta=3.0)

history_kl = {'loss': [], 'val_loss': []}

for epoch in range(40):
    sparse_ae_kl.train()
    train_loss = 0.0
    for xb, _ in train_loader:
        optimizer_kl.zero_grad()
        reconstructed, encoded = sparse_ae_kl(xb)
        recon_loss = criterion_kl(reconstructed, xb)
        sparsity_loss = kl_regularizer(encoded)
        loss = recon_loss + sparsity_loss
        loss.backward()
        optimizer_kl.step()
        train_loss += recon_loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    sparse_ae_kl.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, _ in test_loader:
            reconstructed, _ = sparse_ae_kl(xb)
            val_loss += criterion_kl(reconstructed, xb).item() * xb.size(0)
    val_loss /= len(test_loader.dataset)

    history_kl['loss'].append(train_loss)
    history_kl['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")</code></pre>

    <h3>৬. স্পার্সিটি পরিমাপ করা</h3>
    <pre><code>def measure_sparsity(model, data, threshold=0.1):
    """নেটওয়ার্কের স্পার্সিটি পরিমাপ"""
    model.eval()
    with torch.no_grad():
        _, activations = model(data)
    activations = activations.numpy()

    # threshold এর নিচের মান = নিষ্ক্রিয় নিউরন
    inactive = np.mean(activations &lt; threshold)
    mean_activation = np.mean(activations)

    return {
        'sparsity': inactive,
        'mean_activation': mean_activation,
        'active_neurons_per_sample': np.mean(np.sum(activations > threshold, axis=1))
    }

# সাধারণ AE তৈরি (তুলনার জন্য)
class StandardAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_enc = nn.Linear(784, 512)
        self.fc_dec = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = self.relu(self.fc_enc(x))
        decoded = self.sigmoid(self.fc_dec(encoded))
        return decoded, encoded

standard_ae = StandardAutoencoder()
criterion_std = nn.BCELoss()
optimizer_std = torch.optim.Adam(standard_ae.parameters(), lr=0.001)

for epoch in range(30):
    standard_ae.train()
    for xb, _ in train_loader:
        optimizer_std.zero_grad()
        reconstructed, _ = standard_ae(xb)
        loss = criterion_std(reconstructed, xb)
        loss.backward()
        optimizer_std.step()

# তুলনা
stats_standard = measure_sparsity(standard_ae, x_test)
stats_l1      = measure_sparsity(sparse_ae_l1, x_test)
stats_kl      = measure_sparsity(sparse_ae_kl, x_test)

print("Standard AE:")
print(f"  Sparsity:           {stats_standard['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_standard['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_standard['active_neurons_per_sample']:.1f}")

print("\\nL1 Sparse AE:")
print(f"  Sparsity:           {stats_l1['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_l1['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_l1['active_neurons_per_sample']:.1f}")

print("\\nKL Sparse AE:")
print(f"  Sparsity:           {stats_kl['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_kl['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_kl['active_neurons_per_sample']:.1f}")</code></pre>

    <h3>৭. লার্নড ফিচার ভিজুয়ালাইজেশন</h3>
    <pre><code>def visualize_filters(model, model_name, n_filters=64):
    """এনকোডার ওয়েট ভিজুয়ালাইজ করা"""
    # প্রথম এনকোডার লেয়ারের ওয়েট নিন (PyTorch Linear weight আকার: out_features x in_features)
    weights = model.fc_enc.weight.detach().numpy().T
    # ট্রান্সপোজের পর weights আকার: (784, n_neurons)

    n = min(n_filters, weights.shape[1])
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i &lt; n:
            filter_img = weights[:, i].reshape(28, 28)
            ax.imshow(filter_img, cmap='RdBu_r',
                      vmin=-abs(filter_img).max(),
                      vmax=abs(filter_img).max())
        ax.axis('off')

    plt.suptitle(f'Learned Features — {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'features_{model_name}.png', dpi=150)
    plt.show()

visualize_filters(sparse_ae_l1, 'sparse_ae_l1')</code></pre>

    <h3>৮. Overcomplete স্পার্স অটোএনকোডার</h3>
    <p>
      স্পার্স অটোএনকোডারের সবচেয়ে আকর্ষণীয় বৈশিষ্ট্য: লেটেন্ট ডিমেনশন ইনপুটের চেয়ে <em>বড়</em> হতে পারে। এটি <strong>overcomplete representation</strong> তৈরি করে।
    </p>
    <p>
      যদি encoding_dim = 2000 এবং input_dim = 784 হয়, তাহলে স্পার্সিটি ছাড়া নেটওয়ার্ক trivially ইনপুট কপি করবে। কিন্তু স্পার্সিটি পেনাল্টি নিশ্চিত করে যে যেকোনো একটি স্যাম্পলের জন্য মাত্র কিছু নিউরন সক্রিয় থাকবে।
    </p>
    <p>
      এর সুবিধা: প্রতিটি ফিচার নিউরন ডেটার একটি ছোট, নির্দিষ্ট দিক (আংশিক বৈশিষ্ট্য) শেখে। MNIST-এ একটি নিউরন হয়তো "উপরে বাঁকানো রেখা" শেখে, আরেকটি "গোলাকার আকৃতি" শেখে।
    </p>

    <h3>৯. স্পার্স AE বনাম অন্যান্য পদ্ধতি</h3>
    <p>
      <strong>স্পার্স AE বনাম আন্ডারকমপ্লিট AE:</strong> আন্ডারকমপ্লিট AE কম্প্রেশনের মাধ্যমে স্পার্সিটি নিশ্চিত করে। স্পার্স AE স্পষ্ট পেনাল্টি দিয়ে বড় লেটেন্ট স্পেসেও স্পার্সিটি বজায় রাখে।
    </p>
    <p>
      <strong>স্পার্স AE বনাম PCA:</strong> PCA orthogonal basis vectors শেখে। স্পার্স AE আরও localized এবং interpretable ফিচার শিখতে পারে।
    </p>
    <p>
      <strong>Dictionary Learning-এর সাথে সম্পর্ক:</strong> Overcomplete স্পার্স AE মূলত neural dictionary learning করছে — এটি এমন একটি dictionary (weight matrix) শেখে যেখানে যেকোনো ইনপুট কিছু সংখ্যক basis vector-এর linear combination হিসেবে প্রকাশ পায়।
    </p>

    <h3>১০. Lambda পছন্দ করা</h3>
    <pre><code>lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
sparsity_results = {}

for lam in lambdas:
    model = SparseAutoencoderL1(256)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()

    for epoch in range(15):
        model.train()
        for xb, _ in train_loader:
            opt.zero_grad()
            reconstructed, encoded = model(xb)
            recon_loss = crit(reconstructed, xb)
            l1_penalty = lam * torch.sum(torch.abs(encoded))
            loss = recon_loss + l1_penalty
            loss.backward()
            opt.step()

    # রিকনস্ট্রাকশন লস ও স্পার্সিটি
    model.eval()
    with torch.no_grad():
        reconstructed, acts = model(x_test)
        val_loss = crit(reconstructed, x_test).item()
    sparsity = (acts &lt; 0.1).float().mean().item()
    sparsity_results[lam] = {'loss': val_loss, 'sparsity': sparsity}
    print(f"lambda={lam:.0e}: val_loss={val_loss:.4f}, sparsity={sparsity:.2%}")

# ট্রেডঅফ প্লট
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
lams = list(sparsity_results.keys())

ax1.semilogx(lams, [sparsity_results[l]['loss'] for l in lams], 'bo-', linewidth=2)
ax1.set_xlabel('Lambda (L1 coefficient)', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Reconstruction Quality vs Lambda', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.semilogx(lams, [sparsity_results[l]['sparsity'] for l in lams], 'rs-', linewidth=2)
ax2.set_xlabel('Lambda (L1 coefficient)', fontsize=12)
ax2.set_ylabel('Sparsity (%)', fontsize=12)
ax2.set_title('Sparsity vs Lambda', fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('lambda_tradeoff.png', dpi=150)
plt.show()</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <p>
      স্পার্স অটোএনকোডার লেটেন্ট স্তরে L1 রেগুলারাইজেশন বা KL Divergence পেনাল্টি যোগ করে নিউরনের অ্যাক্টিভেশন কম রাখে। এটি নেটওয়ার্ককে আরও অর্থপূর্ণ, localized এবং ইন্টারপ্রিটেবল ফিচার শিখতে সাহায্য করে। Overcomplete লেটেন্ট স্পেস ব্যবহার করলেও স্পার্সিটি trivial solution রোধ করে।
    </p>
  `
};
