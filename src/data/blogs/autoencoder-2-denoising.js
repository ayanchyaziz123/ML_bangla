export const autoencoder_2_denoising = {
  slug: 'autoencoder-2-denoising',
  title: 'ডিনয়েজিং অটোএনকোডার: নয়েজ থেকে মুক্তি',
  description: 'ডিনয়েজিং অটোএনকোডার কীভাবে নয়েজযুক্ত ইনপুট থেকে পরিষ্কার ডেটা শেখে, Gaussian noise, DAE ট্রেনিং কৌশল এবং MNIST-এ সম্পূর্ণ PyTorch ইমপ্লিমেন্টেশন।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 13,
  content: `
    <h3>১. ডিনয়েজিং অটোএনকোডার কী?</h3>
    <p>
      সাধারণ অটোএনকোডার শেখে কীভাবে ডেটাকে কম্প্রেস করে পুনরুদ্ধার করতে হয়। কিন্তু <strong>ডিনয়েজিং অটোএনকোডার (Denoising Autoencoder — DAE)</strong> একটি কঠিন কাজ করে: এটি <em>নয়েজযুক্ত ইনপুট</em> নেয় এবং <em>পরিষ্কার আউটপুট</em> তৈরি করতে শেখে।
    </p>
    <p>
      ধারণাটি সহজ কিন্তু শক্তিশালী। যদি আমরা একটি ছবিতে কৃত্রিম নয়েজ যোগ করে নেটওয়ার্ককে বলি "এই নয়েজযুক্ত ছবি থেকে আসল ছবি বের করো", তাহলে নেটওয়ার্ককে শুধু পিক্সেল কপি করলে চলবে না — সত্যিকারের উপযোগী বৈশিষ্ট্য (features) শিখতে হবে।
    </p>
    <p>
      DAE-এর প্রস্তাব করেছিলেন <strong>Pascal Vincent এবং দল (২০০৮)</strong>। তাদের মূল অন্তর্দৃষ্টি ছিল: নয়েজ যোগ করে ট্রেনিং নেটওয়ার্ককে আরও শক্তিশালী এবং সাধারণীকরণযোগ্য (generalizable) ফিচার শিখতে সাহায্য করে।
    </p>

    <h3>২. DAE-এর ট্রেনিং উদ্দেশ্য</h3>
    <p>
      সাধারণ অটোএনকোডারের লস: <strong>L(x, g(f(x)))</strong> — মূল ইনপুট এবং রিকনস্ট্রাকশনের পার্থক্য।
    </p>
    <p>
      DAE-এর লস: <strong>L(x, g(f(x̃)))</strong> — এখানে x̃ হলো নয়েজযুক্ত ইনপুট, কিন্তু টার্গেট হলো মূল পরিষ্কার x।
    </p>
    <p>
      এই পরিবর্তনটি ছোট কিন্তু গভীর প্রভাব ফেলে। নেটওয়ার্ককে শিখতে হয় কীভাবে দুর্নীতিগ্রস্ত (corrupted) ডেটা থেকে আসল তথ্য পুনরুদ্ধার করতে হয়।
    </p>

    <h3>৩. নয়েজের ধরন</h3>
    <p>
      <strong>Gaussian Noise:</strong> প্রতিটি পিক্সেলে একটি সাধারণ বিতরণ থেকে র্যান্ডম মান যোগ করা হয়। সবচেয়ে সাধারণ নয়েজ।
      <br/><code>x̃ = x + ε, যেখানে ε ~ N(0, σ²)</code>
    </p>
    <p>
      <strong>Salt-and-Pepper Noise:</strong> কিছু পিক্সেল এলোমেলোভাবে ০ (কালো) বা ১ (সাদা) করা হয়।
    </p>
    <p>
      <strong>Masking Noise (Dropout Noise):</strong> কিছু পিক্সেলকে এলোমেলোভাবে ০ করে দেওয়া হয়। এটি BERT-এর masked language model-এর ধারণার অনুরূপ।
    </p>

    <h3>৪. DAE কেন ভালো ফিচার শেখে?</h3>
    <p>
      সাধারণ অটোএনকোডার যদি লেটেন্ট স্পেস যথেষ্ট বড় হয়, তাহলে ইনপুট হুবহু কপি করে একটি trivial solution খুঁজে পেতে পারে। DAE এই সমস্যা সমাধান করে।
    </p>
    <p>
      <strong>কারণ:</strong> নয়েজযুক্ত ইনপুট থেকে আসল ডেটা পুনরুদ্ধার করতে হলে নেটওয়ার্ককে ডেটার সত্যিকারের কাঠামো (underlying structure) বুঝতে হবে। সে শিখবে কোন পিক্সেলগুলো পরস্পর সম্পর্কিত, ডেটায় কোন প্যাটার্ন আছে।
    </p>
    <p>
      <strong>রোবাস্টনেস:</strong> নয়েজে ট্রেনিং করলে নেটওয়ার্ক বাস্তব জগতের অপূর্ণ ডেটার সাথে মোকাবিলা করতে পারে।
    </p>

    <h3>৫. PyTorch দিয়ে DAE ইমপ্লিমেন্টেশন</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# ডেটা লোড
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data.float() / 255.0
x_test  = test_dataset.data.float()  / 255.0

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# Gaussian নয়েজ যোগ করা
noise_factor = 0.5

x_train_noisy = x_train + noise_factor * torch.randn_like(x_train)
x_test_noisy  = x_test  + noise_factor * torch.randn_like(x_test)

# পিক্সেল মান ০ থেকে ১ এর মধ্যে রাখা
x_train_noisy = torch.clamp(x_train_noisy, 0.0, 1.0)
x_test_noisy  = torch.clamp(x_test_noisy,  0.0, 1.0)

print(f"Clean train shape: {x_train.shape}")
print(f"Noisy train shape: {x_train_noisy.shape}")</code></pre>

    <pre><code>import torch.nn as nn

# DAE মডেল তৈরি
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=128):
        super().__init__()
        # এনকোডার
        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_fc3 = nn.Linear(256, encoding_dim)
        # ডিকোডার
        self.dec_fc1 = nn.Linear(encoding_dim, 256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc3 = nn.Linear(512, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc_fc1(x))
        x = self.relu(self.enc_fc2(x))
        encoded = self.relu(self.enc_fc3(x))
        x = self.relu(self.dec_fc1(encoded))
        x = self.relu(self.dec_fc2(x))
        return self.sigmoid(self.dec_fc3(x))

dae = DenoisingAutoencoder(encoding_dim=128)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(dae.parameters(), lr=0.001)
print(dae)</code></pre>

    <pre><code># ট্রেনিং: নয়েজযুক্ত ইনপুট -> পরিষ্কার আউটপুট
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(x_train_noisy, x_train), batch_size=256, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test_noisy, x_test),  batch_size=256)

history = {'loss': [], 'val_loss': []}

for epoch in range(40):
    dae.train()
    train_loss = 0.0
    for noisy_xb, clean_xb in train_loader:   # নয়েজযুক্ত ইনপুট, পরিষ্কার টার্গেট
        optimizer.zero_grad()
        reconstructed = dae(noisy_xb)
        loss = criterion(reconstructed, clean_xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy_xb.size(0)
    train_loss /= len(train_loader.dataset)

    dae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_xb, clean_xb in test_loader:
            reconstructed = dae(noisy_xb)
            val_loss += criterion(reconstructed, clean_xb).item() * noisy_xb.size(0)
    val_loss /= len(test_loader.dataset)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")</code></pre>

    <pre><code># ডিনয়েজিং রেজাল্ট দেখা
dae.eval()
with torch.no_grad():
    x_test_denoised = dae(x_test_noisy).numpy()

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # মূল পরিষ্কার ছবি
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].numpy().reshape(28, 28), cmap='gray')
    plt.title('Clean', fontsize=8)
    plt.axis('off')

    # নয়েজযুক্ত ছবি
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].numpy().reshape(28, 28), cmap='gray')
    plt.title('Noisy', fontsize=8)
    plt.axis('off')

    # ডিনয়েজড ছবি
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_test_denoised[i].reshape(28, 28), cmap='gray')
    plt.title('Denoised', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('denoising_results.png', dpi=150)
plt.show()</code></pre>

    <h3>৬. বিভিন্ন নয়েজ লেভেলের তুলনা</h3>
    <pre><code># বিভিন্ন noise_factor পরীক্ষা
noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
results = {}

dae.eval()
for nf in noise_levels:
    x_noisy = torch.clamp(x_test + nf * torch.randn_like(x_test), 0, 1)
    with torch.no_grad():
        x_denoised = dae(x_noisy).numpy()

    # MSE হিসাব করা (মূল ছবি vs ডিনয়েজড ছবি)
    mse = np.mean((x_test.numpy() - x_denoised) ** 2)
    results[nf] = mse
    print(f"Noise Factor {nf:.1f} -> MSE: {mse:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), 'bo-', linewidth=2)
plt.xlabel('Noise Factor (σ)', fontsize=13)
plt.ylabel('Reconstruction MSE', fontsize=13)
plt.title('DAE Performance vs Noise Level', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('noise_vs_mse.png', dpi=150)
plt.show()</code></pre>

    <h3>৭. Convolutional DAE (CDAE)</h3>
    <p>
      ছবির জন্য Convolutional স্তর ব্যবহার করলে আরও ভালো ফলাফল পাওয়া যায়, কারণ CNN স্থানিক (spatial) তথ্য সংরক্ষণ করে।
    </p>
    <pre><code>class ConvolutionalDAE(nn.Module):
    def __init__(self):
        super().__init__()
        # এনকোডার
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # ডিকোডার
        self.dec_deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # 7x7 -> 14x14
        self.dec_deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)  # 14x14 -> 28x28
        self.dec_conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.pool1(x)
        x = self.relu(self.enc_conv2(x))
        encoded = self.pool2(x)

        x = self.relu(self.dec_deconv1(encoded))
        x = self.relu(self.dec_deconv2(x))
        return self.sigmoid(self.dec_conv3(x))

# ডেটা রিশেপ (CNN-এর জন্য ৪D: batch, channel, height, width)
x_train_4d = x_train.reshape(-1, 1, 28, 28)
x_test_4d  = x_test.reshape(-1, 1, 28, 28)
x_train_noisy_4d = x_train_noisy.reshape(-1, 1, 28, 28)
x_test_noisy_4d  = x_test_noisy.reshape(-1, 1, 28, 28)

cdae = ConvolutionalDAE()
criterion_conv = nn.BCELoss()
optimizer_conv = torch.optim.Adam(cdae.parameters(), lr=0.001)

conv_train_loader = DataLoader(TensorDataset(x_train_noisy_4d, x_train_4d), batch_size=128, shuffle=True)
conv_test_loader  = DataLoader(TensorDataset(x_test_noisy_4d, x_test_4d),  batch_size=128)

history_conv = {'loss': [], 'val_loss': []}

for epoch in range(30):
    cdae.train()
    train_loss = 0.0
    for noisy_xb, clean_xb in conv_train_loader:
        optimizer_conv.zero_grad()
        reconstructed = cdae(noisy_xb)
        loss = criterion_conv(reconstructed, clean_xb)
        loss.backward()
        optimizer_conv.step()
        train_loss += loss.item() * noisy_xb.size(0)
    train_loss /= len(conv_train_loader.dataset)

    cdae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_xb, clean_xb in conv_test_loader:
            reconstructed = cdae(noisy_xb)
            val_loss += criterion_conv(reconstructed, clean_xb).item() * noisy_xb.size(0)
    val_loss /= len(conv_test_loader.dataset)

    history_conv['loss'].append(train_loss)
    history_conv['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}")</code></pre>

    <h3>৮. DAE এবং সাধারণ AE-এর তুলনা</h3>
    <pre><code># সাধারণ AE তৈরি (একই আর্কিটেকচার কিন্তু পরিষ্কার ইনপুটে ট্রেন)
class StandardAutoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=128):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_fc3 = nn.Linear(256, encoding_dim)
        self.dec_fc1 = nn.Linear(encoding_dim, 256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_fc3 = nn.Linear(512, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc_fc1(x))
        x = self.relu(self.enc_fc2(x))
        x = self.relu(self.enc_fc3(x))
        x = self.relu(self.dec_fc1(x))
        x = self.relu(self.dec_fc2(x))
        return self.sigmoid(self.dec_fc3(x))

sae = StandardAutoencoder()
criterion_sae = nn.BCELoss()
optimizer_sae = torch.optim.Adam(sae.parameters(), lr=0.001)
sae_train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=256, shuffle=True)

for epoch in range(30):
    sae.train()
    for xb, _ in sae_train_loader:
        optimizer_sae.zero_grad()
        reconstructed = sae(xb)
        loss = criterion_sae(reconstructed, xb)
        loss.backward()
        optimizer_sae.step()

# নয়েজযুক্ত টেস্ট ডেটায় উভয় মডেলের MSE
sae.eval()
dae.eval()
with torch.no_grad():
    sae_denoised = sae(x_test_noisy).numpy()
    dae_denoised = dae(x_test_noisy).numpy()

mse_sae = np.mean((x_test.numpy() - sae_denoised) ** 2)
mse_dae = np.mean((x_test.numpy() - dae_denoised) ** 2)

print(f"Standard AE MSE on noisy data: {mse_sae:.4f}")
print(f"Denoising AE MSE on noisy data: {mse_dae:.4f}")
print(f"DAE improvement: {(mse_sae - mse_dae) / mse_sae * 100:.1f}%")</code></pre>

    <h3>৯. বাস্তব জীবনে DAE-এর প্রয়োগ</h3>
    <p>
      <strong>মেডিকেল ইমেজিং:</strong> MRI বা CT স্ক্যানে নয়েজ কমাতে DAE ব্যবহার হয়। কম রেডিয়েশনে তোলা ছবিকে উচ্চমানের ছবিতে রূপান্তরিত করা যায়।
    </p>
    <p>
      <strong>অডিও ডিনয়েজিং:</strong> ব্যাকগ্রাউন্ড নয়েজ থেকে পরিষ্কার কণ্ঠস্বর আলাদা করা। স্পিচ রিকগনিশন সিস্টেমে ব্যবহৃত হয়।
    </p>
    <p>
      <strong>ওল্ড ফটো রেস্টোরেশন:</strong> পুরানো, ক্ষতিগ্রস্ত ছবিকে পুনরুদ্ধার করতে DAE প্রয়োগ করা হচ্ছে।
    </p>
    <p>
      <strong>টেক্সট ক্লিনিং:</strong> স্ক্যান করা ডকুমেন্টে OCR-এর আগে DAE দিয়ে নয়েজ কমানো হয়।
    </p>
    <p>
      <strong>প্রি-ট্রেনিং:</strong> DAE দিয়ে প্রি-ট্রেন করা এনকোডার পরবর্তী কাজে (classification, etc.) fine-tune করা যায় এবং এটি ভালো generalization দেয়।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      ডিনয়েজিং অটোএনকোডার সাধারণ অটোএনকোডারের একটি শক্তিশালী সংস্করণ। নয়েজযুক্ত ইনপুট থেকে পরিষ্কার আউটপুট শেখার মাধ্যমে এটি আরও অর্থপূর্ণ ফিচার শেখে এবং বাস্তব জগতের অপূর্ণ ডেটায় ভালো পারফর্ম করে। Gaussian noise, masking noise বা অন্য যেকোনো corruption strategy ব্যবহার করে DAE ট্রেন করা যায়।
    </p>
  `
};
