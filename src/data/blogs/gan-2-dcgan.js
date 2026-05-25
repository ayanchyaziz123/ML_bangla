export const gan_2_dcgan = {
  slug: 'gan-2-dcgan',
  title: 'DCGAN: কনভোলিউশনাল GAN',
  description: 'ডিপ কনভোলিউশনাল GAN-এর আর্কিটেকচার নিয়ম, ট্রান্সপোজড কনভোলিউশন, ব্যাচ নর্মালাইজেশন এবং PyTorch দিয়ে স্থিতিশীল ছবি জেনারেশন।',
  date: 'মে ২০২৫',
  category: 'জেনারেটিভ অ্যাডভার্সারিয়াল নেটওয়ার্ক',
  readTime: 15,
  content: `
    <h3>১. DCGAN কেন দরকার?</h3>
    <p>
      মূল GAN-এ শুধু ফুলি-কানেক্টেড লেয়ার ব্যবহার করা হয়। ছবির মতো স্থানীয় কাঠামোগত তথ্য (spatial structure) ধরার জন্য এটি যথেষ্ট নয়। ২০১৫ সালে Radford et al. <strong>DCGAN (Deep Convolutional GAN)</strong> প্রস্তাব করেন, যেখানে কনভোলিউশনাল লেয়ার ব্যবহার করে অনেক বেশি মানসম্পন্ন ও স্পষ্ট ছবি তৈরি করা সম্ভব হয়।
    </p>
    <p>
      DCGAN প্রথমবার দেখিয়েছিল যে GAN দিয়ে বেডরুম, মুখের ছবি ইত্যাদি বাস্তবসম্মতভাবে তৈরি করা যায়। এটি পরবর্তী সমস্ত কনভোলিউশনাল GAN-এর ভিত্তি।
    </p>

    <h3>২. DCGAN আর্কিটেকচারের মূল নিয়ম</h3>
    <p>
      DCGAN পেপারে কিছু সুনির্দিষ্ট আর্কিটেকচারাল নিয়ম প্রস্তাব করা হয়েছে যা ট্রেনিং স্থিতিশীল করে:
    </p>

    <h4>নিয়ম ১: পুলিং লেয়ার নয়, স্ট্রাইডেড কনভোলিউশন</h4>
    <p>
      Max pooling-এর পরিবর্তে স্ট্রাইডেড কনভোলিউশন (stride=2) ব্যবহার করুন। এতে নেটওয়ার্ক নিজেই ডাউনস্যাম্পলিং শিখে নেয়। ডিসক্রিমিনেটরে strided conv, জেনারেটরে fractionally-strided conv (transposed conv)।
    </p>

    <h4>নিয়ম ২: ব্যাচ নর্মালাইজেশন</h4>
    <p>
      জেনারেটর ও ডিসক্রিমিনেটরে ব্যাচ নর্মালাইজেশন ব্যবহার করুন। তবে জেনারেটরের আউটপুট লেয়ারে এবং ডিসক্রিমিনেটরের ইনপুট লেয়ারে BatchNorm ব্যবহার করবেন না।
    </p>

    <h4>নিয়ম ৩: Leaky ReLU (ডিসক্রিমিনেটরে)</h4>
    <p>
      ডিসক্রিমিনেটরে সব লেয়ারে Leaky ReLU (alpha=0.2) ব্যবহার করুন। এটি নেগেটিভ মানের জন্য ছোট গ্রেডিয়েন্ট পাস করতে দেয়।
    </p>

    <h4>নিয়ম ৪: ReLU (জেনারেটরে)</h4>
    <p>
      জেনারেটরে সব লেয়ারে ReLU ব্যবহার করুন, শুধু আউটপুট লেয়ারে Tanh ব্যবহার করুন।
    </p>

    <h4>নিয়ম ৫: ফুলি-কানেক্টেড লেয়ার নয়</h4>
    <p>
      কনভোলিউশনাল ফিচারের উপরে global average pooling বা সরাসরি কনভোলিউশন ব্যবহার করুন।
    </p>

    <h3>৩. ট্রান্সপোজড কনভোলিউশন (Transposed Convolution)</h3>
    <p>
      জেনারেটর ছোট ফিচার ম্যাপ থেকে বড় ছবি তৈরি করে। এই আপস্যাম্পলিং-এর জন্য <strong>ট্রান্সপোজড কনভোলিউশন</strong> (deconvolution বা fractionally-strided convolution) ব্যবহার হয়।
    </p>
    <p>
      সাধারণ কনভোলিউশন ছবি ছোট করে (H → H/2), ট্রান্সপোজড কনভোলিউশন ছবি বড় করে (H → H*2)। এটি learnable upsampling, যা nearest-neighbor বা bilinear interpolation-এর চেয়ে ভালো।
    </p>
    <pre><code># আউটপুট আকার গণনা:
# ConvTranspose2d:
#   H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
#
# উদাহরণ: 4x4 input, kernel=4, stride=2, padding=1
#   H_out = (4-1)*2 - 2*1 + 4 = 6 - 2 + 4 = 8
#
# তাই: 4x4 → 8x8 → 16x16 → 32x32 → 64x64

import torch
import torch.nn as nn

x = torch.randn(1, 100, 1, 1)  # latent vector (100-dim)
up1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0)
out1 = up1(x)
print(f"After ConvTranspose 1: {out1.shape}")  # [1, 512, 4, 4]

up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
out2 = up2(out1)
print(f"After ConvTranspose 2: {out2.shape}")  # [1, 256, 8, 8]</code></pre>

    <h3>৪. DCGAN আর্কিটেকচার ডায়াগ্রাম</h3>
    <pre><code"># DCGAN Generator আর্কিটেকচার (64x64 আউটপুটের জন্য):
#
# Input: z (100,)
#    ↓ Reshape
# (100, 1, 1)
#    ↓ ConvTranspose2d(100→512, k=4, s=1, p=0) + BatchNorm + ReLU
# (512, 4, 4)
#    ↓ ConvTranspose2d(512→256, k=4, s=2, p=1) + BatchNorm + ReLU
# (256, 8, 8)
#    ↓ ConvTranspose2d(256→128, k=4, s=2, p=1) + BatchNorm + ReLU
# (128, 16, 16)
#    ↓ ConvTranspose2d(128→64, k=4, s=2, p=1) + BatchNorm + ReLU
# (64, 32, 32)
#    ↓ ConvTranspose2d(64→3, k=4, s=2, p=1) + Tanh
# (3, 64, 64) ← RGB আউটপুট

# DCGAN Discriminator আর্কিটেকচার:
#
# Input: (3, 64, 64) RGB ছবি
#    ↓ Conv2d(3→64, k=4, s=2, p=1) + LeakyReLU(0.2)
# (64, 32, 32)
#    ↓ Conv2d(64→128, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)
# (128, 16, 16)
#    ↓ Conv2d(128→256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)
# (256, 8, 8)
#    ↓ Conv2d(256→512, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2)
# (512, 4, 4)
#    ↓ Conv2d(512→1, k=4, s=1, p=0) + Sigmoid
# (1, 1, 1) ← real/fake probability</code></pre>

    <h3>৫. PyTorch DCGAN ইমপ্লিমেন্টেশন</h3>
    <pre><code>import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# হাইপারপ্যারামিটার
LATENT_DIM   = 100
NGF          = 64   # জেনারেটর ফিচার ম্যাপ সংখ্যা
NDF          = 64   # ডিসক্রিমিনেটর ফিচার ম্যাপ সংখ্যা
NUM_CHANNELS = 1    # MNIST = 1 channel (grayscale)
IMAGE_SIZE   = 64
BATCH_SIZE   = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS   = 30
BETA1        = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')</code></pre>

    <pre><code># ওজন ইনিশিয়ালাইজেশন (DCGAN পেপার অনুযায়ী)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# DCGAN জেনারেটর
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, ngf, num_channels):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # ইনপুট: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 32, 32)

            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # (num_channels, 64, 64)
        )

    def forward(self, z):
        return self.main(z)</code></pre>

    <pre><code># DCGAN ডিসক্রিমিনেটর
class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf, num_channels):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # ইনপুট: (num_channels, 64, 64)
            # প্রথম লেয়ারে BatchNorm নেই
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 32, 32)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4)

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # (1, 1, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)</code></pre>

    <pre><code># মডেল তৈরি ও ইনিশিয়ালাইজ
G = DCGANGenerator(LATENT_DIM, NGF, NUM_CHANNELS).to(device)
D = DCGANDiscriminator(NDF, NUM_CHANNELS).to(device)

G.apply(weights_init)
D.apply(weights_init)

print("Generator:")
print(G)
print(f"\nGenerator parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")

# অপটিমাইজার ও লস
optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
criterion   = nn.BCELoss()

# ফিক্সড নয়েজ (প্রগ্রেস মনিটরিং)
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1).to(device)</code></pre>

    <pre><code># DCGAN ট্রেনিং লুপ
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset    = torchvision.datasets.MNIST('./data', train=True,
                                        download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G_losses, D_losses = [], []

for epoch in range(NUM_EPOCHS):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs  = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.full((batch_size,), 1.0, device=device)
        fake_labels = torch.full((batch_size,), 0.0, device=device)

        # ---- ডিসক্রিমিনেটর আপডেট ----
        D.zero_grad()
        out_real  = D(real_imgs)
        loss_real = criterion(out_real, real_labels)
        loss_real.backward()

        z         = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake_imgs = G(z).detach()
        out_fake  = D(fake_imgs)
        loss_fake = criterion(out_fake, fake_labels)
        loss_fake.backward()

        d_loss = loss_real + loss_fake
        optimizer_D.step()

        # ---- জেনারেটর আপডেট ----
        G.zero_grad()
        z         = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake_imgs = G(z)
        out       = D(fake_imgs)
        g_loss    = criterion(out, real_labels)
        g_loss.backward()
        optimizer_G.step()

    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    if (epoch + 1) % 5 == 0:
        # প্রগ্রেস চেক
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        grid = torchvision.utils.make_grid(fake[:16], nrow=4, normalize=True)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"D: {d_loss.item():.4f}  G: {g_loss.item():.4f}")</code></pre>

    <h3>৬. স্থিতিশীল ট্রেনিংয়ের টিপস</h3>
    <p>
      <strong>লার্নিং রেট ব্যালেন্স:</strong> ডিসক্রিমিনেটর যদি অনেক দ্রুত শিখে, জেনারেটরের জন্য লার্নিং রেট বাড়ান বা D-এর আপডেট কমান।
    </p>
    <p>
      <strong>ইমেজ নর্মালাইজেশন:</strong> ছবিকে [-1, 1]-এ নর্মালাইজ করুন এবং জেনারেটরে Tanh ব্যবহার করুন — এই দুটি সামঞ্জস্যপূর্ণ।
    </p>
    <p>
      <strong>ব্যাচ সাইজ:</strong> বড় ব্যাচ সাইজ (128-256) ব্যাচ নর্মালাইজেশনের জন্য ভালো।
    </p>
    <p>
      <strong>ইনিশিয়ালাইজেশন:</strong> DCGAN পেপার অনুযায়ী N(0, 0.02) দিয়ে ওজন ইনিশিয়ালাইজ করুন।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      DCGAN হলো GAN-এর কনভোলিউশনাল সংস্করণ যা স্থানীয় ভিজ্যুয়াল কাঠামো শিখতে পারে। পুলিং-মুক্ত কনভোলিউশন, ব্যাচ নর্মালাইজেশন, এবং Leaky ReLU-এর সমন্বয়ে অনেক বেশি স্থিতিশীল ও মানসম্পন্ন ছবি তৈরি সম্ভব। এই আর্কিটেকচার পরবর্তী সমস্ত উন্নত GAN-এর ভিত্তি হিসেবে কাজ করেছে।
    </p>
  `
};
