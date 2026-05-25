export const gan_1_basics = {
  slug: 'gan-1-basics',
  title: 'GAN: জেনারেটর ও ডিসক্রিমিনেটরের খেলা',
  description: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্কের মূল ধারণা, মিনিম্যাক্স অবজেক্টিভ, ন্যাশ ইকুইলিব্রিয়াম এবং PyTorch দিয়ে MNIST-এ সম্পূর্ণ GAN ইমপ্লিমেন্টেশন।',
  date: 'মে ২০২৫',
  category: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক',
  readTime: 14,
  content: `
    <h3>১. GAN কী এবং কেন এত গুরুত্বপূর্ণ?</h3>
    <p>
      <strong>জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক (GAN)</strong> হলো মেশিন লার্নিংয়ের একটি যুগান্তকারী আইডিয়া, যা Ian Goodfellow ২০১৪ সালে প্রস্তাব করেন। GAN দুটি নিউরাল নেটওয়ার্ককে একে অপরের বিরুদ্ধে প্রতিযোগিতায় রেখে শেখায় — একজন শিল্পী যে ছবি তৈরি করে, আর একজন সমালোচক যে বলে ছবিটি আসল না নকল।
    </p>
    <p>
      GAN-এর জাদু হলো এটি সম্পূর্ণ নতুন ডেটা তৈরি করতে পারে — মানুষের মুখের ছবি, শিল্পকর্ম, সঙ্গীত, এমনকি ভিডিও পর্যন্ত। এই কারণেই GAN কে "মেশিন লার্নিংয়ের সবচেয়ে আকর্ষণীয় আইডিয়া" বলা হয়।
    </p>

    <h3>২. দুই প্রতিযোগী: জেনারেটর ও ডিসক্রিমিনেটর</h3>
    <h4>২.১ জেনারেটর (Generator)</h4>
    <p>
      জেনারেটর একটি র‍্যান্ডম নয়েজ ভেক্টর (z) নিয়ে সেটাকে একটি বাস্তবসম্মত ছবি বা ডেটায় রূপান্তরিত করে। এটি একজন জালিয়াত শিল্পীর মতো — যে কখনো আসল ছবি না দেখেও জাল ছবি তৈরি করতে চায়।
    </p>
    <p>
      গণিতের ভাষায়: <strong>G: z → x̃</strong>, যেখানে z ~ N(0, I) হলো র‍্যান্ডম নয়েজ এবং x̃ হলো জেনারেটেড স্যাম্পল।
    </p>

    <h4>২.২ ডিসক্রিমিনেটর (Discriminator)</h4>
    <p>
      ডিসক্রিমিনেটর একটি বাইনারি ক্লাসিফায়ার — এটি বলে দেয় ইনপুট ডেটাটি আসল (real distribution থেকে) না জেনারেটরের তৈরি (fake)। এটি একজন বিশেষজ্ঞ সমালোচকের মতো।
    </p>
    <p>
      গণিতের ভাষায়: <strong>D: x → [0, 1]</strong>, যেখানে আউটপুট ১ মানে "আসল" এবং ০ মানে "নকল"।
    </p>

    <h3>৩. মিনিম্যাক্স অবজেক্টিভ (Minimax Objective)</h3>
    <p>
      GAN-এর ট্রেনিং একটি মিনিম্যাক্স গেম হিসেবে সংজ্ঞায়িত করা হয়:
    </p>
    <p>
      <strong>min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]</strong>
    </p>
    <p>
      এখানে:
    </p>
    <p>
      <strong>ডিসক্রিমিনেটর maximize করতে চায়:</strong> আসল ডেটায় D(x) ≈ 1 এবং নকল ডেটায় D(G(z)) ≈ 0 রাখতে চায়, ফলে log D(x) + log(1 - D(G(z))) বাড়াতে চায়।
    </p>
    <p>
      <strong>জেনারেটর minimize করতে চায়:</strong> D(G(z)) ≈ 1 করতে চায় (ডিসক্রিমিনেটরকে বোকা বানাতে), ফলে log(1 - D(G(z))) কমাতে চায়।
    </p>

    <h3>৪. ন্যাশ ইকুইলিব্রিয়াম (Nash Equilibrium)</h3>
    <p>
      তাত্ত্বিকভাবে, যখন জেনারেটর এতটাই ভালো হয়ে যায় যে তার তৈরি ডেটা আসল ডেটার মতো হুবহু একই ডিস্ট্রিবিউশন অনুসরণ করে, তখন ডিসক্রিমিনেটর আর পার্থক্য করতে পারে না। এই অবস্থায় D(x) = 0.5 সব ক্ষেত্রে — এটাই <strong>ন্যাশ ইকুইলিব্রিয়াম</strong>।
    </p>
    <p>
      এই পয়েন্টে p_g (জেনারেটরের ডিস্ট্রিবিউশন) = p_data (আসল ডেটার ডিস্ট্রিবিউশন) হয়। এটি GAN-এর আদর্শ লক্ষ্য।
    </p>

    <h3>৫. ট্রেনিং অ্যালগরিদম</h3>
    <pre><code># GAN ট্রেনিং লুপের সিউডোকোড
# প্রতিটি ইটারেশনে:

# ধাপ ১: ডিসক্রিমিনেটর আপডেট (k বার)
for _ in range(k):  # সাধারণত k=1
    # আসল ডেটা সাম্পল করা
    x_real = sample_real_data(batch_size)

    # নকল ডেটা তৈরি
    z = sample_noise(batch_size, latent_dim)
    x_fake = Generator(z)  # detach gradient

    # ডিসক্রিমিনেটর লস
    D_loss = -[mean(log(D(x_real))) + mean(log(1 - D(x_fake)))]

    # ডিসক্রিমিনেটর আপডেট
    D_optimizer.step(D_loss)

# ধাপ ২: জেনারেটর আপডেট
z = sample_noise(batch_size, latent_dim)
x_fake = Generator(z)

# জেনারেটর লস (ডিসক্রিমিনেটরকে বোকা বানানো)
G_loss = -mean(log(D(x_fake)))  # non-saturating version

# জেনারেটর আপডেট
G_optimizer.step(G_loss)</code></pre>

    <h3>৬. ট্রেনিং সমস্যা</h3>
    <h4>৬.১ মোড কোল্যাপ্স (Mode Collapse)</h4>
    <p>
      মোড কোল্যাপ্স GAN-এর সবচেয়ে পরিচিত সমস্যা। এতে জেনারেটর শুধুমাত্র কয়েকটি ধরনের আউটপুট তৈরি করতে শেখে, বৈচিত্র্য হারিয়ে ফেলে। উদাহরণস্বরূপ, MNIST ডিজিট জেনারেট করতে গিয়ে শুধু "৭" তৈরি করতে থাকে।
    </p>
    <p>
      সমাধান: Minibatch discrimination, Unrolled GAN, বা WGAN ব্যবহার করা।
    </p>

    <h4>৬.২ ভ্যানিশিং গ্রেডিয়েন্ট (Vanishing Gradients)</h4>
    <p>
      যখন ডিসক্রিমিনেটর অনেক শক্তিশালী হয়, D(G(z)) ≈ 0 হয়ে যায়। এই অবস্থায় log(1 - D(G(z))) ≈ 0 এবং গ্রেডিয়েন্ট প্রায় শূন্য হয়ে যায়, জেনারেটর আর কিছু শিখতে পারে না।
    </p>
    <p>
      সমাধান: Non-saturating loss (-log D(G(z))-এর পরিবর্তে log(1-D(G(z)))) ব্যবহার করা।
    </p>

    <h4>৬.৩ ট্রেনিং ইনস্ট্যাবিলিটি</h4>
    <p>
      জেনারেটর এবং ডিসক্রিমিনেটর একই সাথে ট্রেন হয়, ফলে লস oscillate করে। সঠিক ব্যালেন্স বজায় রাখা কঠিন।
    </p>

    <h3>৭. PyTorch দিয়ে Simple GAN — MNIST</h3>
    <pre><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# হাইপারপ্যারামিটার
LATENT_DIM   = 100
HIDDEN_DIM   = 256
IMAGE_DIM    = 784  # 28x28
BATCH_SIZE   = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS   = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")</code></pre>

    <pre><code># জেনারেটর নেটওয়ার্ক
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # আউটপুট -1 থেকে 1 এর মধ্যে
        )

    def forward(self, z):
        return self.model(z)

# ডিসক্রিমিনেটর নেটওয়ার্ক
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # আউটপুট 0 থেকে 1 এর মধ্যে
        )

    def forward(self, x):
        return self.model(x)</code></pre>

    <pre><code># ডেটা লোড করা
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # -1 থেকে 1 এ নর্মালাইজ
])

dataset    = datasets.MNIST(root='./data', train=True,
                             download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# মডেল ইনিশিয়ালাইজ
G = Generator(LATENT_DIM, HIDDEN_DIM, IMAGE_DIM).to(device)
D = Discriminator(IMAGE_DIM, HIDDEN_DIM).to(device)

# অপটিমাইজার ও লস ফাংশন
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion   = nn.BCELoss()

print(f"Generator parameters:     {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")</code></pre>

    <pre><code># ট্রেনিং লুপ
G_losses, D_losses = [], []

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.view(-1, IMAGE_DIM).to(device)
        batch_size = real_imgs.size(0)

        # লেবেল তৈরি
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ====== ডিসক্রিমিনেটর ট্রেনিং ======
        optimizer_D.zero_grad()

        # আসল ছবিতে ডিসক্রিমিনেটরের লস
        d_real_out  = D(real_imgs)
        d_real_loss = criterion(d_real_out, real_labels)

        # নকল ছবি তৈরি
        z           = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs   = G(z).detach()  # জেনারেটর গ্রেডিয়েন্ট আটকানো

        # নকল ছবিতে ডিসক্রিমিনেটরের লস
        d_fake_out  = D(fake_imgs)
        d_fake_loss = criterion(d_fake_out, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # ====== জেনারেটর ট্রেনিং ======
        optimizer_G.zero_grad()

        z         = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = G(z)
        g_out     = D(fake_imgs)

        # জেনারেটর চায় ডিসক্রিমিনেটর fake কে real ভাবুক
        g_loss    = criterion(g_out, real_labels)
        g_loss.backward()
        optimizer_G.step()

    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"D_loss: {d_loss.item():.4f}  "
              f"G_loss: {g_loss.item():.4f}")</code></pre>

    <pre><code># জেনারেটেড ছবি দেখা
G.eval()
with torch.no_grad():
    z            = torch.randn(64, LATENT_DIM).to(device)
    generated    = G(z).cpu().numpy()
    generated    = generated.reshape(-1, 28, 28)
    generated    = (generated + 1) / 2  # -1~1 থেকে 0~1 এ রূপান্তর

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i], cmap='gray')
    ax.axis('off')
plt.suptitle('GAN Generated MNIST Digits', fontsize=16)
plt.tight_layout()
plt.savefig('gan_generated.png', dpi=150)
plt.show()

# লস কার্ভ
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss', linewidth=2)
plt.plot(D_losses, label='Discriminator Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('gan_loss.png', dpi=150)
plt.show()</code></pre>

    <h3>৮. ভালো ট্রেনিংয়ের টিপস</h3>
    <p>
      <strong>লার্নিং রেট:</strong> জেনারেটর এবং ডিসক্রিমিনেটরের জন্য আলাদা লার্নিং রেট ব্যবহার করা যায়। সাধারণত 0.0002 এবং Adam optimizer-এ betas=(0.5, 0.999) ভালো কাজ করে।
    </p>
    <p>
      <strong>ব্যালেন্স রাখা:</strong> ডিসক্রিমিনেটর যদি অনেক বেশি শক্তিশালী হয়ে যায়, তাহলে জেনারেটর শিখতে পারে না। প্রতিটি ব্যাচে একবার G আপডেট, একবার D আপডেট সাধারণত ভালো।
    </p>
    <p>
      <strong>লেবেল স্মুদিং (Label Smoothing):</strong> real লেবেলকে 1-এর পরিবর্তে 0.9 ব্যবহার করলে ট্রেনিং আরো স্থিতিশীল হয়।
    </p>
    <p>
      <strong>নয়েজ ভেক্টর:</strong> Gaussian noise (z ~ N(0,1)) সাধারণত Uniform noise-এর চেয়ে ভালো ফল দেয়।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      GAN হলো দুটি নেটওয়ার্কের মধ্যে একটি সৃজনশীল প্রতিযোগিতা। জেনারেটর নতুন ডেটা তৈরি করে, ডিসক্রিমিনেটর সেটা যাচাই করে। মিনিম্যাক্স অবজেক্টিভ উভয়কে একসাথে উন্নত করে। মোড কোল্যাপ্স ও ভ্যানিশিং গ্রেডিয়েন্ট প্রধান চ্যালেঞ্জ, যা DCGAN, WGAN ইত্যাদি পরবর্তী আর্কিটেকচারে সমাধান করা হয়েছে।
    </p>
  `
};
