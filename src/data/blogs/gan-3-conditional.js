export const gan_3_conditional = {
  slug: 'gan-3-conditional',
  title: 'Conditional GAN: লেবেল-নিয়ন্ত্রিত জেনারেশন',
  description: 'Conditional GAN-এ ক্লাস লেবেল দিয়ে নির্দিষ্ট ছবি জেনারেট করা, লেবেল এমবেডিং এবং Pix2Pix দিয়ে image-to-image ট্রান্সলেশন।',
  date: 'মে ২০২৫',
  category: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক',
  readTime: 14,
  content: `
    <h3>১. Conditional GAN কী?</h3>
    <p>
      সাধারণ GAN যেকোনো ছবি তৈরি করে — আপনি বলতে পারবেন না কোন ধরনের ছবি চাই। কিন্তু <strong>Conditional GAN (cGAN)</strong> দিয়ে আপনি নির্দিষ্ট করে বলতে পারবেন "৭ নম্বর ডিজিটের ছবি তৈরি করো" বা "বিড়ালের ছবি তৈরি করো"।
    </p>
    <p>
      Mirza ও Osindero ২০১৪ সালে cGAN প্রস্তাব করেন। ধারণাটি সহজ: জেনারেটর ও ডিসক্রিমিনেটর — উভয়কেই লেবেল তথ্য (condition) দেওয়া হয়।
    </p>

    <h3>২. cGAN-এর গাণিতিক কাঠামো</h3>
    <p>
      cGAN-এর অবজেক্টিভ ফাংশন:
    </p>
    <p>
      <strong>min_G max_D V(D, G) = E[log D(x|y)] + E[log(1 - D(G(z|y)|y))]</strong>
    </p>
    <p>
      এখানে <strong>y</strong> হলো condition — ক্লাস লেবেল, টেক্সট, বা অন্য যেকোনো তথ্য। জেনারেটর G(z|y) লেবেল y দিয়ে কন্ডিশনড হয়ে ছবি তৈরি করে। ডিসক্রিমিনেটর D(x|y) ছবি এবং লেবেল উভয় দেখে বলে আসল না নকল।
    </p>

    <h3>৩. লেবেল এমবেডিং (Label Embedding)</h3>
    <p>
      লেবেল (integer) সরাসরি নেটওয়ার্কে দেওয়া যায় না। লেবেলকে ভেক্টরে রূপান্তর করতে <strong>Embedding লেয়ার</strong> ব্যবহার করা হয়।
    </p>
    <pre><code>import torch
import torch.nn as nn

# উদাহরণ: ১০টি ক্লাস (MNIST 0-9)
embedding = nn.Embedding(num_embeddings=10, embedding_dim=50)

# ক্লাস লেবেল 3 এবং 7
labels = torch.tensor([3, 7])
label_embeds = embedding(labels)
print(f"Label embedding shape: {label_embeds.shape}")  # [2, 50]

# এই embedding ভেক্টর জেনারেটর/ডিসক্রিমিনেটরের ইনপুটের সাথে যুক্ত হয়</code></pre>

    <h4>৩.১ জেনারেটরে লেবেল কন্ডিশনিং</h4>
    <p>
      জেনারেটরে লেবেল এমবেডিং-কে নয়েজ ভেক্টর z-এর সাথে concatenate করা হয়:
    </p>
    <pre><code>class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, embed_dim, img_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.model = nn.Sequential(
            # latent_dim + embed_dim → hidden layers
            nn.Linear(latent_dim + embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embeds = self.label_embed(labels)         # (B, embed_dim)
        x = torch.cat([z, label_embeds], dim=1)         # (B, latent_dim + embed_dim)
        return self.model(x)</code></pre>

    <h4>৩.২ ডিসক্রিমিনেটরে লেবেল কন্ডিশনিং</h4>
    <pre><code>class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes, embed_dim, img_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(img_dim + embed_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_embeds = self.label_embed(labels)     # (B, embed_dim)
        x = torch.cat([x, label_embeds], dim=1)     # (B, img_dim + embed_dim)
        return self.model(x)</code></pre>

    <h3>৪. সম্পূর্ণ cGAN MNIST ট্রেনিং</h3>
    <pre><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

LATENT_DIM   = 100
EMBED_DIM    = 50
NUM_CLASSES  = 10
IMAGE_DIM    = 784
BATCH_SIZE   = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS   = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# মডেল তৈরি
G = ConditionalGenerator(LATENT_DIM, NUM_CLASSES, EMBED_DIM, IMAGE_DIM).to(device)
D = ConditionalDiscriminator(NUM_CLASSES, EMBED_DIM, IMAGE_DIM).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion   = nn.BCELoss()

# ডেটা
transform  = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])
dataset    = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)</code></pre>

    <pre><code># cGAN ট্রেনিং লুপ
for epoch in range(NUM_EPOCHS):
    for real_imgs, real_labels in dataloader:
        real_imgs   = real_imgs.view(-1, IMAGE_DIM).to(device)
        real_labels = real_labels.to(device)
        batch_size  = real_imgs.size(0)

        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # ---- ডিসক্রিমিনেটর ----
        D.zero_grad()

        # আসল ছবি + সঠিক লেবেল
        out_real  = D(real_imgs, real_labels)
        loss_real = criterion(out_real, real_targets)

        # নকল ছবি + র‍্যান্ডম লেবেল
        z           = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
        fake_imgs   = G(z, fake_labels).detach()
        out_fake    = D(fake_imgs, fake_labels)
        loss_fake   = criterion(out_fake, fake_targets)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_D.step()

        # ---- জেনারেটর ----
        G.zero_grad()
        z           = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
        fake_imgs   = G(z, fake_labels)
        out         = D(fake_imgs, fake_labels)
        g_loss      = criterion(out, real_targets)
        g_loss.backward()
        optimizer_G.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  "
              f"D: {d_loss.item():.4f}  G: {g_loss.item():.4f}")</code></pre>

    <pre><code># নির্দিষ্ট ডিজিট জেনারেট করা
G.eval()
with torch.no_grad():
    # প্রতিটি ডিজিট (0-9) ৮টি করে জেনারেট করা
    all_images = []
    for digit in range(10):
        z      = torch.randn(8, LATENT_DIM).to(device)
        labels = torch.full((8,), digit, dtype=torch.long).to(device)
        imgs   = G(z, labels).cpu().numpy().reshape(-1, 28, 28)
        imgs   = (imgs + 1) / 2
        all_images.append(imgs)

# ভিজুয়ালাইজ
fig, axes = plt.subplots(10, 8, figsize=(12, 16))
for digit, imgs in enumerate(all_images):
    for col, img in enumerate(imgs):
        axes[digit, col].imshow(img, cmap='gray')
        axes[digit, col].axis('off')
        if col == 0:
            axes[digit, col].set_ylabel(f'Digit {digit}', fontsize=10)

plt.suptitle('Conditional GAN: Each Row = Specific Digit', fontsize=14)
plt.tight_layout()
plt.savefig('cgan_digits.png', dpi=150)
plt.show()</code></pre>

    <h3>৫. Pix2Pix: Image-to-Image Translation</h3>
    <p>
      <strong>Pix2Pix</strong> (Isola et al., 2017) হলো একটি বিশেষ cGAN যা একটি ছবিকে অন্য ধরনের ছবিতে রূপান্তর করে। উদাহরণ:
    </p>
    <p>
      স্কেচ → ফটো, দিনের ছবি → রাতের ছবি, স্যাটেলাইট ছবি → মানচিত্র, সাদাকালো → রঙিন।
    </p>

    <h4>৫.১ Pix2Pix জেনারেটর: U-Net</h4>
    <p>
      Pix2Pix জেনারেটরে <strong>U-Net</strong> আর্কিটেকচার ব্যবহার হয়। U-Net এ এনকোডার ও ডিকোডারের মধ্যে skip connections থাকে — এতে fine-grained detail সংরক্ষিত হয়।
    </p>
    <pre><code>class UNetBlock(nn.Module):
    """U-Net এর একটি ব্লক (এনকোডার বা ডিকোডার)"""
    def __init__(self, in_channels, out_channels, down=True,
                 use_bn=True, dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size=4, stride=2,
                                              padding=1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if down:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)</code></pre>

    <h4>৫.২ Pix2Pix ডিসক্রিমিনেটর: PatchGAN</h4>
    <p>
      Pix2Pix-এ <strong>PatchGAN</strong> ডিসক্রিমিনেটর ব্যবহার হয়। এটি পুরো ছবির বদলে 70×70 পিক্সেলের প্যাচ ধরে বলে real/fake। প্যাচ-ভিত্তিক মূল্যায়নে local texture detail ভালো ধরা পড়ে।
    </p>
    <pre><code>class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6):  # input_img + target_img
        super().__init__()
        self.model = nn.Sequential(
            # ইনপুট: source ও target ছবি concatenate (6 channels)
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1)  # আউটপুট: patch-level scores
        )

    def forward(self, source, target):
        # source এবং target ছবি একসাথে দেওয়া হয়
        x = torch.cat([source, target], dim=1)
        return self.model(x)</code></pre>

    <h4>৫.৩ Pix2Pix লস</h4>
    <p>
      Pix2Pix লস দুটি অংশ নিয়ে গঠিত:
    </p>
    <p>
      <strong>Adversarial Loss:</strong> সাধারণ GAN লসের মতো — জেনারেটেড ছবিকে বাস্তবসম্মত দেখাতে হবে।
    </p>
    <p>
      <strong>L1 Loss:</strong> জেনারেটেড ছবি এবং টার্গেট ছবির মধ্যে pixel-wise পার্থক্য কমাতে হবে।
    </p>
    <pre><code># Pix2Pix মোট লস
lambda_L1 = 100  # L1 লসের ওজন

# জেনারেটর লস
g_adv_loss = criterion(D(source, generated), real_labels)
g_l1_loss  = l1_loss(generated, target) * lambda_L1
g_total    = g_adv_loss + g_l1_loss

# ডিসক্রিমিনেটর লস
d_real_loss = criterion(D(source, target), real_labels)
d_fake_loss = criterion(D(source, generated.detach()), fake_labels)
d_loss      = (d_real_loss + d_fake_loss) * 0.5</code></pre>

    <h3>৬. cGAN এর প্রয়োগক্ষেত্র</h3>
    <p>
      <strong>ডেটা অ্যাগমেন্টেশন:</strong> নির্দিষ্ট ক্লাসের জন্য বাড়তি ট্রেনিং ডেটা তৈরি করা। ইমব্যালেন্সড ক্লাসে খুব উপকারী।
    </p>
    <p>
      <strong>স্টাইল ট্রান্সফার:</strong> বিভিন্ন শিল্পীর স্টাইলে ছবি আঁকা।
    </p>
    <p>
      <strong>মেডিকেল ইমেজিং:</strong> X-ray থেকে MRI, বা বিভিন্ন modality-তে রূপান্তর।
    </p>
    <p>
      <strong>ফ্যাশন ডিজাইন:</strong> কাপড়ের প্যাটার্ন থেকে পরা ছবি তৈরি।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      Conditional GAN লেবেল বা অন্য condition দিয়ে নিয়ন্ত্রিত জেনারেশন সম্ভব করে। লেবেল এমবেডিং জেনারেটর ও ডিসক্রিমিনেটর উভয়ে দেওয়া হয়। Pix2Pix image-to-image translation-এ U-Net ও PatchGAN ব্যবহার করে শিল্পোত্তীর্ণ ফলাফল দেয়। cGAN-এর এই নমনীয়তা একে বাস্তব প্রয়োগে অত্যন্ত কার্যকর করে তোলে।
    </p>
  `
};
