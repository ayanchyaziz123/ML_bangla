export const transformer_6_vit = {
  slug: 'transformer-6-vit',
  title: 'Vision Transformer (ViT): ছবি বোঝার নতুন পদ্ধতি',
  description: 'Transformer-কে image classification-এ প্রয়োগ — Patch Embedding, Positional Encoding for images, ViT vs CNN comparison এবং সম্পূর্ণ Python implementation।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 17,
  content: `
    <h3>১. ছবিতে Transformer কেন?</h3>
    <p>২০২০ সালের আগে image classification-এর জন্য CNN ছিল অপ্রতিদ্বন্দ্বী। কিন্তু CNN-এর কিছু সীমাবদ্ধতা আছে:</p>
    <ul>
      <li><strong>Local receptive field:</strong> একটি neuron শুধু local neighborhood দেখে — distant pixels-এর মধ্যে direct interaction নেই।</li>
      <li><strong>Inductive biases:</strong> Translation invariance hardwired — কিছু ক্ষেত্রে এটি পজিটিভ, কিছু ক্ষেত্রে নেগেটিভ।</li>
      <li><strong>Large images-এ computation:</strong> Deep CNN-এ computation অনেক বেশি।</li>
    </ul>
    <p>২০২০ সালে Google Brain-এর "An Image is Worth 16x16 Words" পেপারে <strong>Vision Transformer (ViT)</strong> প্রকাশিত হয়। এটি image-কে text sequence-এর মতো treat করে — ছবিকে patch-এ ভেঙে প্রতিটি patch-কে একটি "token" হিসেবে Transformer-এ দেওয়া হয়।</p>

    <h3>২. Patch Embedding: ছবিকে Sequence করা</h3>
    <p>একটি H×W×C image কে P×P patch-এ ভাগ করা হয়।</p>
    <p>মোট patch সংখ্যা: <strong>N = (H/P) × (W/P)</strong></p>
    <p>উদাহরণ: 224×224 image, P=16 → N = 14×14 = 196 patches</p>
    <p>প্রতিটি patch: P×P×C = 16×16×3 = 768 pixels → linear projection → d_model dimensional vector।</p>
    <p>এটা ঠিক NLP-এর মতো: "words" এর পরিবর্তে "patches"।</p>
    <pre><code class="language-python">import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Image -> Patch Tokens
    Input:  (batch, C, H, W)
    Output: (batch, num_patches, d_model)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # একটি Conv2D দিয়ে patch extraction + linear projection
        # kernel_size=stride=patch_size => non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels,    # input channels (3 for RGB)
            d_model,        # output channels = embedding dim
            kernel_size=patch_size,
            stride=patch_size
        )
        # এই Conv2D প্রতিটি patch-কে d_model dim vector-এ project করে

    def forward(self, x):
        """
        x: (batch, C, H, W)
        """
        # Conv2D: (batch, d_model, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims: (batch, d_model, num_patches)
        x = x.flatten(2)
        # Transpose: (batch, num_patches, d_model)
        x = x.transpose(1, 2)
        return x

# Test
img_size, patch_size, in_ch, d_model = 32, 8, 3, 64
patch_embed = PatchEmbedding(img_size, patch_size, in_ch, d_model)

images = torch.randn(4, 3, 32, 32)   # batch=4, RGB, 32x32
patches = patch_embed(images)

print(f"Image input:      {images.shape}")
print(f"Patch output:     {patches.shape}")
# num_patches = (32/8)^2 = 16
# patches: (4, 16, 64)

print(f"Number of patches: {patch_embed.num_patches}")
print(f"Patch embedding parameters: {sum(p.numel() for p in patch_embed.parameters()):,}")
</code></pre>

    <h3>৩. [CLS] Token ও Positional Encoding</h3>
    <p>NLP-এর BERT-এর মতো ViT-এও একটি learnable <strong>[CLS] token</strong> prepend করা হয়। Final [CLS] representation দিয়ে classification করা হয়।</p>
    <p>Positional encoding-এও ViT sinusoidal নয়, <strong>learned 1D position embedding</strong> ব্যবহার করে। 2D position (row, col) থেকে 1D-তে flatten করা হয়।</p>
    <pre><code class="language-python">class ViTEmbedding(nn.Module):
    """Patch embedding + [CLS] token + positional encoding"""
    def __init__(self, img_size=32, patch_size=8, in_ch=3,
                 d_model=64, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, d_model)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Learned positional embedding (+1 for [CLS])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # Initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)           # (B, N, d_model)

        # [CLS] token prepend
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, d_model)

        # Positional encoding যোগ
        x = x + self.pos_embed                   # (B, N+1, d_model)
        return self.dropout(x)

# Test
vit_emb = ViTEmbedding()
images = torch.randn(4, 3, 32, 32)
emb_out = vit_emb(images)
print(f"Images:      {images.shape}")
print(f"Embedding:   {emb_out.shape}")
# (4, 17, 64) — 16 patches + 1 [CLS]
</code></pre>

    <h3>৪. ViT Encoder Block</h3>
    <p>ViT-এর Transformer block মূলত BERT-এর Encoder layer-এর মতোই। পার্থক্য:</p>
    <ul>
      <li><strong>Pre-LN:</strong> Layer Norm আগে apply হয় (Post-LN-এর চেয়ে stable)।</li>
      <li><strong>GELU activation:</strong> ReLU-র পরিবর্তে।</li>
      <li><strong>MLP ratio:</strong> FFN inner dim = 4 × d_model।</li>
    </ul>
    <pre><code class="language-python">class ViTBlock(nn.Module):
    """ViT Transformer encoder block (Pre-LN)"""
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN Self-Attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)

        # Pre-LN MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_ch=3,
                 num_classes=10, d_model=64, depth=6,
                 num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embedding = ViTEmbedding(
            img_size, patch_size, in_ch, d_model, dropout
        )
        self.blocks = nn.ModuleList([
            ViTBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Classification head — [CLS] token-এর উপর
        self.head = nn.Linear(d_model, num_classes)

        # Weight init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embedding(x)             # (B, N+1, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]                 # [CLS] token (B, d_model)
        return self.head(cls_out)         # (B, num_classes)

# Test
vit = VisionTransformer(
    img_size=32, patch_size=8, num_classes=10,
    d_model=64, depth=6, num_heads=4
)
images = torch.randn(4, 3, 32, 32)
logits = vit(images)
print(f"Input:  {images.shape}")
print(f"Output: {logits.shape}")  # (4, 10) — 10 classes

total_params = sum(p.numel() for p in vit.parameters())
print(f"Parameters: {total_params:,}")
</code></pre>

    <h3>৫. ViT বনাম CNN: বিস্তারিত তুলনা</h3>
    <table>
      <thead>
        <tr><th>বৈশিষ্ট্য</th><th>CNN</th><th>ViT</th></tr>
      </thead>
      <tbody>
        <tr><td>Inductive bias</td><td>Strong (local, translation invariant)</td><td>Minimal (data-driven)</td></tr>
        <tr><td>Global context</td><td>Layer stack করে gradually</td><td>প্রতিটি layer-এ সরাসরি</td></tr>
        <tr><td>Small data</td><td>ভালো (inductive bias সাহায্য করে)</td><td>দুর্বল (বেশি data দরকার)</td></tr>
        <tr><td>Large data (ImageNet-21K)</td><td>CNN-এর সাথে comparable</td><td>CNN-কে ছাড়িয়ে যায়</td></tr>
        <tr><td>Scalability</td><td>Limited</td><td>Data ও model scale-এ ভালো</td></tr>
        <tr><td>Interpretability</td><td>Grad-CAM visualizable</td><td>Attention map visualizable</td></tr>
        <tr><td>Computation</td><td>O(N) in image pixels</td><td>O(N²) in patches</td></tr>
      </tbody>
    </table>
    <p>Key finding: ViT ছোট dataset-এ CNN-এর চেয়ে দুর্বল, কিন্তু JFT-300M বা ImageNet-21K-এ pre-train করলে CNN-কে ছাড়িয়ে যায়।</p>

    <h3>৬. Attention Map Visualization</h3>
    <p>ViT-এ প্রতিটি patch কোন অন্য patch-এ attend করছে তা visualize করলে model কী দেখছে বোঝা যায়।</p>
    <pre><code class="language-python">import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_attention_map(model, image, patch_size=8, layer_idx=-1):
    """
    ViT-এর [CLS] token-এর attention map visualize করা।
    image: (1, C, H, W)
    """
    model.eval()
    attention_maps = []

    # Hook দিয়ে attention weights capture করুন
    hooks = []
    def hook_fn(module, input, output):
        # output[1] = attention weights
        if isinstance(output, tuple):
            attention_maps.append(output[1].detach())

    for block in model.blocks:
        h = block.attn.register_forward_hook(hook_fn)
        hooks.append(h)

    with torch.no_grad():
        _ = model(image)

    for h in hooks:
        h.remove()

    # Specified layer-এর attention
    attn = attention_maps[layer_idx]  # (batch, heads, N+1, N+1)
    attn = attn.mean(dim=1)           # average over heads: (batch, N+1, N+1)

    # [CLS] token-এর attention to patches
    cls_attn = attn[0, 0, 1:]         # (N,) — [CLS]'s attention to patches

    # Patch grid-এ reshape
    H_patches = W_patches = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(H_patches, W_patches).numpy()

    # Upsample to original image size
    import cv2
    img_size = image.shape[-1]
    attn_resized = cv2.resize(attn_map, (img_size, img_size))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original image
    img_np = image[0].permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention map
    im = axes[1].imshow(attn_resized, cmap='hot')
    axes[1].set_title(f'[CLS] Attention Map (Layer {layer_idx})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('vit_attention_map.png', dpi=100)
    print("Saved: vit_attention_map.png")
    return attn_map

# Test
vit_small = VisionTransformer(img_size=32, patch_size=8,
                               num_classes=10, d_model=64,
                               depth=4, num_heads=4)
dummy_image = torch.randn(1, 3, 32, 32)
attn_map = visualize_attention_map(vit_small, dummy_image)
print(f"Attention map shape: {attn_map.shape}")  # (4, 4) for 32x32, patch=8
</code></pre>

    <h3>৭. ViT Variants ও আধুনিক উন্নয়ন</h3>
    <p>Original ViT থেকে অনেক improved variant এসেছে:</p>
    <ul>
      <li><strong>DeiT (2020):</strong> Data-Efficient Image Transformer — knowledge distillation দিয়ে ImageNet-only training।</li>
      <li><strong>Swin Transformer (2021):</strong> Shifted window attention — hierarchical feature maps, CNN-like structure।</li>
      <li><strong>BEiT (2021):</strong> BERT-style pre-training for ViT — masked image modeling।</li>
      <li><strong>MAE (2022):</strong> Masked Autoencoders — 75% patches mask করে reconstruct করা।</li>
    </ul>
    <p>Swin Transformer বিশেষভাবে গুরুত্বপূর্ণ কারণ এটি object detection ও segmentation-এও ভালো করে।</p>

    <h3>৮. সারসংক্ষেপ: পুরো Series</h3>
    <p>এই ৬-পর্বের Transformer series-এ আমরা শিখলাম:</p>
    <ul>
      <li><strong>পর্ব ১:</strong> Attention mechanism — Q/K/V, scaled dot-product attention।</li>
      <li><strong>পর্ব ২:</strong> Self-attention ও Multi-head attention।</li>
      <li><strong>পর্ব ৩:</strong> পূর্ণ Encoder-Decoder architecture, Positional Encoding।</li>
      <li><strong>পর্ব ৪:</strong> BERT — bidirectional pre-training, MLM, fine-tuning।</li>
      <li><strong>পর্ব ৫:</strong> GPT — causal LM, autoregressive generation।</li>
      <li><strong>পর্ব ৬:</strong> ViT — image patches as tokens, global self-attention for vision।</li>
    </ul>
    <p>Transformer architecture আজ NLP থেকে Computer Vision, Speech, Protein folding (AlphaFold) পর্যন্ত সর্বত্র ব্যবহার হচ্ছে। "Attention Is All You Need" — সত্যিই এই একটি mechanism দিয়ে AI জগৎ পাল্টে গেছে।</p>
  `,
};
