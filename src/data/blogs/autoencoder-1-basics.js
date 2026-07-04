export const autoencoder_1_basics = {
  slug: 'autoencoder-1-basics',
  title: 'অটোএনকোডার: কম্প্রেশন দিয়ে শেখা',
  description: 'অটোএনকোডারের আর্কিটেকচার, রিকনস্ট্রাকশন লস, লেটেন্ট স্পেস এবং PyTorch দিয়ে MNIST ডেটাসেটে সম্পূর্ণ ইমপ্লিমেন্টেশন।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 12,
  content: `
    <h3>১. অটোএনকোডার কী?</h3>
    <p>
      <strong>অটোএনকোডার (Autoencoder)</strong> হলো একটি বিশেষ ধরনের নিউরাল নেটওয়ার্ক যা ডেটাকে প্রথমে একটি ছোট উপস্থাপনায় (compact representation) কম্প্রেস করে, তারপর সেই কম্প্রেসড উপস্থাপনা থেকে মূল ডেটা পুনরুদ্ধার করার চেষ্টা করে। এটি একটি unsupervised learning পদ্ধতি — কোনো লেবেলের প্রয়োজন নেই।
    </p>
    <p>
      কল্পনা করুন একটি ছবিকে আপনি JPEG ফরম্যাটে সংরক্ষণ করছেন। JPEG কম্প্রেশন ছবির সবচেয়ে গুরুত্বপূর্ণ তথ্য রেখে বাকিটা ফেলে দেয়, ফলে ফাইলের আকার ছোট হয়। অটোএনকোডার ঠিক একইভাবে কাজ করে — কিন্তু এটি <em>শেখে</em> কোন তথ্য গুরুত্বপূর্ণ।
    </p>
    <p>
      অটোএনকোডারের মূল বৈশিষ্ট্য হলো ইনপুট এবং আউটপুট একই রকম। নেটওয়ার্কটি নিজেই নিজের টিচার — এটি ইনপুটকে টার্গেট হিসেবে ব্যবহার করে শেখে।
    </p>

    <h3>২. অটোএনকোডারের আর্কিটেকচার</h3>
    <p>একটি অটোএনকোডার তিনটি মূল অংশে বিভক্ত:</p>

    <h4>২.১ এনকোডার (Encoder)</h4>
    <p>
      এনকোডার ইনপুট ডেটাকে একটি ছোট, ঘন (dense) উপস্থাপনায় রূপান্তরিত করে। যদি ইনপুট একটি ২৮×২৮ পিক্সেলের ছবি হয় (৭৮৪ ডিমেনশন), তাহলে এনকোডার সেটাকে হয়তো মাত্র ৩২ সংখ্যায় প্রকাশ করে।
    </p>
    <p>
      গাণিতিকভাবে: <strong>z = f(x) = σ(Wx + b)</strong>, যেখানে x হলো ইনপুট, W হলো ওয়েট ম্যাট্রিক্স, b হলো বায়াস এবং σ হলো অ্যাক্টিভেশন ফাংশন।
    </p>

    <h4>২.২ বটলনেক / লেটেন্ট স্পেস (Bottleneck / Latent Space)</h4>
    <p>
      এটি অটোএনকোডারের সবচেয়ে সরু স্তর। এই স্তরটিকে <strong>লেটেন্ট ভেক্টর</strong> বা <strong>কোড</strong> বলা হয়। এখানে ডেটার সবচেয়ে গুরুত্বপূর্ণ বৈশিষ্ট্যগুলো সংকুচিত আকারে সংরক্ষিত থাকে। বটলনেক যত ছোট, তত বেশি কম্প্রেশন — তত বেশি কিছু শিখতে বাধ্য হয় নেটওয়ার্ক।
    </p>

    <h4>২.৩ ডিকোডার (Decoder)</h4>
    <p>
      ডিকোডার লেটেন্ট ভেক্টর থেকে মূল ডেটার মতো আউটপুট তৈরি করার চেষ্টা করে। এটি এনকোডারের বিপরীত কাজ করে — ছোট থেকে বড়তে যায়।
    </p>
    <p>
      গাণিতিকভাবে: <strong>x̂ = g(z) = σ(W'z + b')</strong>, যেখানে x̂ হলো পুনর্গঠিত আউটপুট।
    </p>

    <h3>৩. রিকনস্ট্রাকশন লস (Reconstruction Loss)</h3>
    <p>
      অটোএনকোডার ট্রেন করতে আমরা মূল ইনপুট (x) এবং রিকনস্ট্রাক্টেড আউটপুট (x̂) এর মধ্যে পার্থক্য কমাতে চাই। এই পার্থক্যকে বলা হয় <strong>রিকনস্ট্রাকশন লস</strong>।
    </p>
    <p>
      বাইনারি ডেটার জন্য (যেমন ০ থেকে ১ এর মধ্যে নর্মালাইজড পিক্সেল) <strong>Binary Cross-Entropy</strong> ব্যবহার হয়:
    </p>
    <p>
      <strong>L = -Σ [x·log(x̂) + (1-x)·log(1-x̂)]</strong>
    </p>
    <p>
      কন্টিনিউয়াস ডেটার জন্য <strong>Mean Squared Error (MSE)</strong> ব্যবহার হয়:
    </p>
    <p>
      <strong>L = (1/n) · Σ(x - x̂)²</strong>
    </p>
    <p>
      লক্ষ্য করুন, লেবেলের প্রয়োজন নেই — ইনপুটই লেবেল। এটাই অটোএনকোডারের সৌন্দর্য।
    </p>

    <h3>৪. আন্ডারকমপ্লিট অটোএনকোডার (Undercomplete Autoencoder)</h3>
    <p>
      যখন লেটেন্ট স্পেসের ডিমেনশন ইনপুটের চেয়ে ছোট হয়, তখন তাকে <strong>আন্ডারকমপ্লিট অটোএনকোডার</strong> বলে। এটাই সবচেয়ে সাধারণ এবং দরকারি ধরন।
    </p>
    <p>
      কেন? কারণ যদি লেটেন্ট স্পেস ইনপুটের মতো বড় হয়, তাহলে নেটওয়ার্ক শুধু ইনপুট কপি করতে শিখবে — কোনো অর্থপূর্ণ বৈশিষ্ট্য শিখবে না। সংকীর্ণ বটলনেক নেটওয়ার্ককে বাধ্য করে সত্যিকারের গুরুত্বপূর্ণ তথ্য বের করতে।
    </p>
    <p>
      <strong>উদাহরণ:</strong> MNIST ডিজিটের ছবি ৭৮৪ পিক্সেলের, কিন্তু আমরা লেটেন্ট স্পেস মাত্র ৩২ ডিমেনশনের করলাম। এর মানে নেটওয়ার্ককে ৭৮৪টি পিক্সেলের তথ্য মাত্র ৩২টি সংখ্যায় সংকুচিত করতে হবে।
    </p>

    <h3>৫. লেটেন্ট স্পেস বিশ্লেষণ</h3>
    <p>
      লেটেন্ট স্পেস অত্যন্ত তথ্যবহুল। একবার ট্রেনিং শেষ হলে:
    </p>
    <p>
      <strong>ক) ফিচার এক্সট্রাকশন:</strong> এনকোডারকে ফিচার এক্সট্র্যাক্টর হিসেবে ব্যবহার করা যায়। লেটেন্ট ভেক্টরগুলো ডেটার ঘন উপস্থাপনা, যা ক্লাসিফায়ারে পাঠানো যায়।
    </p>
    <p>
      <strong>খ) ভিজুয়ালাইজেশন:</strong> লেটেন্ট স্পেসকে ২D-তে প্রজেক্ট করলে (t-SNE বা PCA দিয়ে) একই ধরনের ডেটা পয়েন্ট একসাথে গুচ্ছবদ্ধ হয়।
    </p>
    <p>
      <strong>গ) ইন্টারপোলেশন:</strong> দুটি লেটেন্ট ভেক্টরের মাঝামাঝি বিন্দু ডিকোড করলে দুটি ডেটার মিশ্রণ পাওয়া যায়।
    </p>

    <h3>৬. PyTorch দিয়ে MNIST অটোএনকোডার</h3>
    <p>
      এবার সম্পূর্ণ কোড লিখে দেখি। আমরা MNIST হাতের অঙ্ক ডেটাসেটে একটি সাধারণ ডেন্স অটোএনকোডার তৈরি করব।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# ডেটা লোড এবং প্রিপ্রসেস
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ০ থেকে ১ এর মধ্যে নর্মালাইজ করা
x_train = train_dataset.data.float() / 255.0
x_test  = test_dataset.data.float()  / 255.0

# ২৮x২৮ ছবিকে ৭৮৪ ডিমেনশনের ভেক্টরে রূপান্তর
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

print(f"Training shape: {x_train.shape}")
print(f"Test shape:     {x_test.shape}")</code></pre>

    <pre><code>import torch.nn as nn

# এনকোডার তৈরি
encoding_dim = 32  # লেটেন্ট স্পেসের আকার

class Encoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super().__init__()
        self.enc_1 = nn.Linear(784, 128)
        self.enc_2 = nn.Linear(128, 64)
        self.latent = nn.Linear(64, encoding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc_1(x))
        x = self.relu(self.enc_2(x))
        return self.relu(self.latent(x))

encoder = Encoder(encoding_dim)
print(encoder)</code></pre>

    <pre><code># ডিকোডার তৈরি
class Decoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super().__init__()
        self.dec_1 = nn.Linear(encoding_dim, 64)
        self.dec_2 = nn.Linear(64, 128)
        # Sigmoid কারণ পিক্সেল মান ০ থেকে ১ এর মধ্যে
        self.output_layer = nn.Linear(128, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.dec_1(z))
        z = self.relu(self.dec_2(z))
        return self.sigmoid(self.output_layer(z))

decoder = Decoder(encoding_dim)</code></pre>

    <pre><code># সম্পূর্ণ অটোএনকোডার
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

autoencoder = Autoencoder(encoder, decoder)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
print(autoencoder)</code></pre>

    <pre><code># ট্রেনিং
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test, x_test),  batch_size=256)

history = {'loss': [], 'val_loss': []}

for epoch in range(30):
    autoencoder.train()
    train_loss = 0.0
    for xb, _ in train_loader:              # ইনপুট এবং টার্গেট একই
        optimizer.zero_grad()
        reconstructed = autoencoder(xb)
        loss = criterion(reconstructed, xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, _ in test_loader:
            reconstructed = autoencoder(xb)
            val_loss += criterion(reconstructed, xb).item() * xb.size(0)
    val_loss /= len(test_loader.dataset)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")</code></pre>

    <pre><code># রিকনস্ট্রাকশন দেখা
autoencoder.eval()
with torch.no_grad():
    x_test_encoded = encoder(x_test)
    x_test_decoded = decoder(x_test_encoded).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # মূল ছবি
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].numpy().reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # রিকনস্ট্রাক্টেড ছবি
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_decoded[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.savefig('reconstruction.png', dpi=150)
plt.show()
print(f"Latent space shape: {x_test_encoded.shape}")</code></pre>

    <pre><code># লেটেন্ট স্পেস ভিজুয়ালাইজেশন
from sklearn.manifold import TSNE

y_test_labels = test_dataset.targets.numpy()

# শুধু ৩০০০ স্যাম্পল নেওয়া (t-SNE ধীর)
sample_idx = np.random.choice(len(x_test), 3000, replace=False)
x_sample   = x_test[sample_idx]
y_sample   = y_test_labels[sample_idx]

with torch.no_grad():
    latent_sample = encoder(x_sample).numpy()

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
latent_2d = tsne.fit_transform(latent_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=y_sample, cmap='tab10', alpha=0.7, s=10)
plt.colorbar(scatter, label='Digit Class')
plt.title('Latent Space (t-SNE) — MNIST Digits', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('latent_space.png', dpi=150)
plt.show()</code></pre>

    <h3>৭. লস কার্ভ বিশ্লেষণ</h3>
    <pre><code># ট্রেনিং লস কার্ভ
plt.figure(figsize=(10, 5))
plt.plot(history['loss'],     label='Training Loss',   linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=13)
plt.title('Autoencoder Training Loss', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()</code></pre>

    <h3>৮. অটোএনকোডারের প্রয়োগক্ষেত্র</h3>
    <p>
      <strong>ডাইমেনশনালিটি রিডাকশন:</strong> PCA-এর মতো কাজ করে, তবে নন-লিনিয়ার ট্রান্সফরমেশন শিখতে পারে। জটিল ডেটার জন্য PCA-এর চেয়ে অনেক শক্তিশালী।
    </p>
    <p>
      <strong>অ্যানোমালি ডিটেকশন:</strong> স্বাভাবিক ডেটায় ট্রেন করা অটোএনকোডার অস্বাভাবিক ডেটা ভালো রিকনস্ট্রাক্ট করতে পারে না, ফলে উচ্চ রিকনস্ট্রাকশন এরর অ্যানোমালি নির্দেশ করে।
    </p>
    <p>
      <strong>ডিনয়েজিং:</strong> নয়েজযুক্ত ডেটা থেকে পরিষ্কার ডেটা তৈরি করা যায়।
    </p>
    <p>
      <strong>ডেটা জেনারেশন:</strong> ভ্যারিয়েশনাল অটোএনকোডার (VAE) নতুন ডেটা তৈরি করতে পারে।
    </p>
    <p>
      <strong>ফিচার লার্নিং:</strong> লেটেন্ট ভেক্টর downstream কাজে ব্যবহার করা যায়।
    </p>

    <h3>৯. সাধারণ সমস্যা এবং সমাধান</h3>
    <p>
      <strong>Underfitting:</strong> মডেল যদি ভালোভাবে রিকনস্ট্রাক্ট না করতে পারে, তাহলে এনকোডার/ডিকোডারের লেয়ার বা নোড সংখ্যা বাড়ান, বা epoch বাড়ান।
    </p>
    <p>
      <strong>Trivial Solution:</strong> বটলনেক খুব বড় হলে নেটওয়ার্ক শুধু কপি করতে শেখে। সমাধান: লেটেন্ট ডিমেনশন কমান বা রেগুলারাইজেশন যোগ করুন।
    </p>
    <p>
      <strong>Blurry Reconstruction:</strong> MSE বা BCE ব্যবহারে রিকনস্ট্রাকশন অস্পষ্ট হতে পারে। Perceptual loss বা adversarial training (AAE) ব্যবহার করুন।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      অটোএনকোডার হলো একটি শক্তিশালী unsupervised learning টুল। এনকোডার ডেটাকে কম্প্রেস করে লেটেন্ট স্পেসে পাঠায়, আর ডিকোডার সেখান থেকে পুনরুদ্ধার করে। রিকনস্ট্রাকশন লস নেটওয়ার্ককে শেখায়। এই মূল ধারণাটি বুঝলে ডিনয়েজিং, স্পার্স এবং ভ্যারিয়েশনাল অটোএনকোডার বোঝা অনেক সহজ হবে।
    </p>
  `
};
