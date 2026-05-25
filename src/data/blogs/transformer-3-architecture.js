export const transformer_3_architecture = {
  slug: 'transformer-3-architecture',
  title: 'ট্রান্সফর্মার আর্কিটেকচার: এনকোডার-ডিকোডার',
  description: 'পূর্ণ Transformer architecture — Encoder stack, Decoder stack, Positional Encoding (sin/cos), Layer Normalization, Residual connections এবং PyTorch implementation।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 18,
  content: `
    <h3>১. Transformer-এর সামগ্রিক চিত্র</h3>
    <p>২০১৭ সালে "Attention Is All You Need" পেপারে প্রকাশিত Transformer architecture-এর দুটি মূল অংশ:</p>
    <ul>
      <li><strong>Encoder:</strong> Input sequence পড়ে একটি rich representation তৈরি করে।</li>
      <li><strong>Decoder:</strong> Encoder-এর output ব্যবহার করে target sequence generate করে।</li>
    </ul>
    <p>Original Transformer-এর hyperparameters:</p>
    <table>
      <thead>
        <tr><th>Parameter</th><th>মান</th><th>ব্যাখ্যা</th></tr>
      </thead>
      <tbody>
        <tr><td>d_model</td><td>512</td><td>Embedding dimension</td></tr>
        <tr><td>N (layers)</td><td>6</td><td>Encoder ও Decoder-এ layer সংখ্যা</td></tr>
        <tr><td>h (heads)</td><td>8</td><td>Attention heads</td></tr>
        <tr><td>d_ff</td><td>2048</td><td>FFN inner dimension</td></tr>
        <tr><td>d_k = d_v</td><td>64</td><td>d_model / h</td></tr>
        <tr><td>Dropout</td><td>0.1</td><td>Regularization</td></tr>
      </tbody>
    </table>

    <h3>২. Positional Encoding: Sequence-এ Position জানানো</h3>
    <p>Attention mechanism-এ কোনো inherent order নেই — "cat sat mat" ও "mat sat cat" একই attention score পাবে যদি positional information না থাকে। তাই প্রতিটি token-এর embedding-এ position information যোগ করতে হয়।</p>
    <p>Original Transformer-এ sinusoidal positional encoding ব্যবহার হয়:</p>
    <p><strong>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</strong></p>
    <p><strong>PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</strong></p>
    <p>এখানে pos = position index, i = dimension index। জোড় dimension-এ sin, বিজোড়-এ cos।</p>
    <p>এই encoding-এর সুবিধা:</p>
    <ul>
      <li>যেকোনো length-এর sequence handle করতে পারে।</li>
      <li>PE(pos + k) কে PE(pos)-এর linear function হিসেবে express করা যায় — relative position শেখা সহজ।</li>
      <li>Different frequency-তে oscillation ভিন্ন ভিন্ন scale-এ position encode করে।</li>
    </ul>
    <pre><code class="language-python">import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term: 10000^(2i/d_model) এর inverse
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # জোড় index: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # বিজোড় index: cos

        # (1, max_len, d_model) — batch dimension যোগ
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # model parameter নয়, কিন্তু state-এ থাকে

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Test
d_model = 16
pos_enc = PositionalEncoding(d_model, max_len=100)
x = torch.zeros(2, 10, d_model)  # (batch=2, seq=10, d_model=16)
out = pos_enc(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")  # Same shape

# Position 0 ও 1-এর encoding compare করুন
print("\nPosition 0 encoding (first 8 dims):")
print(pos_enc.pe[0, 0, :8].numpy().round(3))
print("Position 1 encoding (first 8 dims):")
print(pos_enc.pe[0, 1, :8].numpy().round(3))
</code></pre>

    <h3>৩. Encoder: পূর্ণ Implementation</h3>
    <p>Encoder-এ N টি identical layer stack করা হয়। প্রতিটি layer-এ:</p>
    <ol>
      <li>Multi-Head Self-Attention (+ residual + layer norm)</li>
      <li>Position-wise FFN (+ residual + layer norm)</li>
    </ol>
    <pre><code class="language-python">import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionPT(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch = Q.size(0)

        # Linear projection + split heads
        def project_split(linear, x):
            x = linear(x)  # (batch, seq, d_model)
            return x.view(batch, -1, self.h, self.d_k).transpose(1, 2)
            # -> (batch, h, seq, d_k)

        Q = project_split(self.W_Q, Q)
        K = project_split(self.W_K, K)
        V = project_split(self.W_V, V)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = attn @ V  # (batch, h, seq, d_k)

        # Concatenate heads + output projection
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.W_O(context), attn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionPT(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sub-layer 1: Self-attention + residual
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: FFN + residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

# Test Encoder
vocab_size, d_model, N, h, d_ff = 1000, 32, 2, 4, 64
encoder = Encoder(vocab_size, d_model, N, h, d_ff)
src = torch.randint(0, vocab_size, (2, 10))  # (batch=2, seq=10)
enc_out = encoder(src)
print(f"Encoder input (token ids): {src.shape}")
print(f"Encoder output:            {enc_out.shape}")  # (2, 10, 32)
</code></pre>

    <h3>৪. Decoder: তিনটি Sub-layer</h3>
    <p>Decoder-এ প্রতিটি layer-এ তিনটি sub-layer:</p>
    <ol>
      <li><strong>Masked Multi-Head Self-Attention:</strong> Future tokens দেখতে পাবে না।</li>
      <li><strong>Multi-Head Cross-Attention:</strong> Q = decoder state, K = V = encoder output।</li>
      <li><strong>Position-wise FFN।</strong></li>
    </ol>
    <pre><code class="language-python">class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Sub-layer 1: Masked self-attention
        self.self_attn = MultiHeadAttentionPT(d_model, num_heads, dropout)
        # Sub-layer 2: Cross-attention with encoder output
        self.cross_attn = MultiHeadAttentionPT(d_model, num_heads, dropout)
        # Sub-layer 3: FFN
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention
        attn1, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 2. Cross-attention: Q from decoder, K/V from encoder
        attn2, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 3. FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

def make_causal_mask(seq_len):
    """Upper triangular mask — future tokens block করে"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

# Test Decoder Layer
d_model, num_heads, d_ff = 32, 4, 64
decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
tgt = torch.randn(2, 7, d_model)   # decoder input
enc_out = torch.randn(2, 10, d_model)  # encoder output
tgt_mask = make_causal_mask(7)

dec_out = decoder_layer(tgt, enc_out, tgt_mask=tgt_mask)
print(f"Decoder input:  {tgt.shape}")
print(f"Encoder output: {enc_out.shape}")
print(f"Decoder output: {dec_out.shape}")  # (2, 7, 32)
</code></pre>

    <h3>৫. পূর্ণ Transformer Model</h3>
    <pre><code class="language-python">class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6,
                 num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, num_heads, d_ff, dropout)

        # Decoder components
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.tgt_pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Final output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.tgt_pos_enc(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)  # (batch, tgt_seq, tgt_vocab)

# Full model test
src_vocab, tgt_vocab = 1000, 800
model = Transformer(src_vocab, tgt_vocab, d_model=32, N=2, num_heads=4, d_ff=64)

src = torch.randint(0, src_vocab, (2, 10))
tgt = torch.randint(0, tgt_vocab, (2, 7))
tgt_mask = make_causal_mask(7)

logits = model(src, tgt, tgt_mask=tgt_mask)
print(f"Output logits: {logits.shape}")  # (2, 7, 800)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
</code></pre>

    <h3>৬. Embedding Scaling কেন?</h3>
    <p>কোডে দেখুন <code>embedding(src) * math.sqrt(d_model)</code> — এটি কেন?</p>
    <p>Positional encoding-এর magnitude সাধারণত [-1, 1]-এর মধ্যে থাকে। Embedding-কে √d_model দিয়ে scale করলে embedding ও positional encoding-এর magnitude আপেক্ষিকভাবে balanced থাকে। এতে training-এ PE signal হারিয়ে যায় না।</p>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>পূর্ণ Transformer architecture:</p>
    <ul>
      <li><strong>Positional Encoding:</strong> sin/cos function দিয়ে position information এমবেড করা।</li>
      <li><strong>Encoder (N layers):</strong> Self-attention + FFN + Add&amp;Norm।</li>
      <li><strong>Decoder (N layers):</strong> Masked self-attention + Cross-attention + FFN + Add&amp;Norm।</li>
      <li><strong>Output projection:</strong> Final linear + softmax → vocabulary probability।</li>
    </ul>
    <p>পরের পর্বে দেখব কীভাবে এই Transformer-কে base করে BERT তৈরি হলো — শুধু Encoder ব্যবহার করে masked language modeling।</p>
  `,
};
