export const transformer_5_gpt = {
  slug: 'transformer-5-gpt',
  title: 'GPT: অটোরিগ্রেসিভ ল্যাঙ্গুয়েজ মডেল',
  description: 'GPT Decoder-only architecture, Causal self-attention, autoregressive text generation, GPT vs BERT comparison এবং HuggingFace দিয়ে text generation।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 16,
  content: `
    <h3>১. GPT কী? Decoder-Only Architecture</h3>
    <p><strong>GPT (Generative Pre-trained Transformer)</strong> হলো OpenAI-এর তৈরি language model যা Transformer-এর শুধু Decoder অংশ ব্যবহার করে। BERT যেখানে Encoder-only, GPT সেখানে Decoder-only।</p>
    <p>GPT পরিবারের evolution:</p>
    <table>
      <thead>
        <tr><th>Model</th><th>Parameters</th><th>Layers</th><th>d_model</th><th>প্রধান বৈশিষ্ট্য</th></tr>
      </thead>
      <tbody>
        <tr><td>GPT-1 (2018)</td><td>117M</td><td>12</td><td>768</td><td>First GPT, supervised fine-tuning</td></tr>
        <tr><td>GPT-2 (2019)</td><td>1.5B</td><td>48</td><td>1600</td><td>Zero-shot, few-shot generalization</td></tr>
        <tr><td>GPT-3 (2020)</td><td>175B</td><td>96</td><td>12288</td><td>In-context learning, few-shot</td></tr>
        <tr><td>GPT-4 (2023)</td><td>~1T (est)</td><td>—</td><td>—</td><td>Multimodal, RLHF</td></tr>
      </tbody>
    </table>

    <h3>২. Causal Self-Attention: ভবিষ্যৎ দেখা যাবে না</h3>
    <p>GPT-এর Decoder-এ <strong>Causal (Masked) Self-Attention</strong> ব্যবহার হয়। প্রতিটি token শুধু নিজে এবং তার আগের tokens দেখতে পারে — future tokens দেখতে পারে না।</p>
    <p>কেন এই restriction? কারণ GPT <strong>autoregressive</strong> — একটি token generate করে, তারপর সেটা দিয়ে পরেরটা, এভাবে চলে। যদি future দেখতে পারত তাহলে generation meaningless হয়ে যেত।</p>
    <p>Causal mask (4×4 example):</p>
    <pre><code>Position:  0  1  2  3
    0:    [1  0  0  0]   token 0 শুধু নিজেকে দেখে
    1:    [1  1  0  0]   token 1 দেখে: 0, 1
    2:    [1  1  1  0]   token 2 দেখে: 0, 1, 2
    3:    [1  1  1  1]   token 3 দেখে: 0, 1, 2, 3
</code></pre>
    <pre><code class="language-python">import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention"""
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads

        # Q, K, V একসাথে — efficient
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask (lower triangular) — buffer হিসেবে register
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask.view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()  # batch, seq_len, d_model

        # Q, K, V একসাথে compute করুন
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        Q, K, V = qkv.split(C, dim=2)

        # Split heads: (B, h, T, d_k)
        def split_h(t):
            return t.view(B, T, self.h, self.d_k).transpose(1, 2)
        Q, K, V = split_h(Q), split_h(K), split_h(V)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # Causal mask apply করুন
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, float('-inf')
        )
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Output
        out = attn @ V                              # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))

# Test
B, T, d_model, h = 2, 8, 16, 4
csa = CausalSelfAttention(d_model, h, max_len=50)
x = torch.randn(B, T, d_model)
out = csa(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")  # (2, 8, 16) — same shape
</code></pre>

    <h3>৩. GPT Block ও পূর্ণ Architecture</h3>
    <p>BERT-এর Encoder layer-এর সাথে GPT Block-এর পার্থক্য:</p>
    <ul>
      <li>Cross-attention নেই (no encoder-decoder attention)।</li>
      <li>Causal mask — future দেখা যায় না।</li>
      <li>GPT-2 থেকে Pre-LN (Layer Norm-first) pattern।</li>
    </ul>
    <pre><code class="language-python">class GPTBlock(nn.Module):
    """GPT-2 style transformer block (Pre-LN)"""
    def __init__(self, d_model, num_heads, d_ff, max_len=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),          # GPT-2 GELU ব্যবহার করে, ReLU নয়
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN: LayerNorm আগে, তারপর residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads,
                 max_len=1024, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # Learned position embedding
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, 4*d_model, max_len, dropout)
            for _ in range(N)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Language model head (tied weights with embedding — common in GPT)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        idx: (batch, seq_len) — token indices
        Returns: (batch, seq_len, vocab_size) — logits
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# Test
vocab_size, d_model, N, h = 500, 32, 4, 4
gpt = GPT(vocab_size, d_model, N, h, max_len=128)
idx = torch.randint(0, vocab_size, (2, 20))  # (batch=2, seq=20)
logits = gpt(idx)
print(f"Input indices: {idx.shape}")
print(f"Output logits: {logits.shape}")  # (2, 20, 500)

total_params = sum(p.numel() for p in gpt.parameters())
print(f"Parameters:    {total_params:,}")
</code></pre>

    <h3>৪. Autoregressive Text Generation</h3>
    <p>GPT-এর text generation হয় autoregressive পদ্ধতিতে — একটি token generate করে, সেটা দিয়ে আবার পরেরটা।</p>
    <p>বিভিন্ন sampling strategy:</p>
    <ul>
      <li><strong>Greedy decoding:</strong> প্রতিটি step-এ সর্বোচ্চ probability token নেওয়া। Fast কিন্তু repetitive।</li>
      <li><strong>Temperature sampling:</strong> Logits / temperature। Low temperature = more deterministic, high = more random।</li>
      <li><strong>Top-k sampling:</strong> Top k tokens-এর মধ্যে sample করা।</li>
      <li><strong>Top-p (nucleus) sampling:</strong> Cumulative probability p পর্যন্ত tokens-এর মধ্যে sample।</li>
    </ul>
    <pre><code class="language-python">import torch
import torch.nn.functional as F

def generate(model, prompt_ids, max_new_tokens=50,
             temperature=1.0, top_k=None, top_p=None):
    """
    GPT-style autoregressive generation।
    prompt_ids: (1, prompt_len) tensor
    """
    model.eval()
    generated = prompt_ids.clone()  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Model forward pass — শুধু last token-এর logits দরকার
            logits = model(generated)          # (1, seq, vocab)
            next_logits = logits[:, -1, :]     # (1, vocab) — শেষ position

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                # Top-k-এর বাইরে -inf দিন
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < min_val, float('-inf'))

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(
                    next_logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1)
                # Cumulative prob > p হলে remove
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float('-inf')
                # Original order-এ ফিরিয়ে দিন
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            # Sample করুন
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # EOS token হলে stop (ধরুন EOS = 1)
            if next_token.item() == 1:
                break

    return generated

# Test generation
vocab_size, d_model, N, h = 100, 16, 2, 2
model = GPT(vocab_size, d_model, N, h, max_len=64)
prompt = torch.tensor([[5, 12, 33]])  # 3 prompt tokens

# Different strategies
greedy = generate(model, prompt, max_new_tokens=10, temperature=0.01)
temp   = generate(model, prompt, max_new_tokens=10, temperature=0.8)
topk   = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5)

print(f"Prompt length:  {prompt.shape[1]}")
print(f"Greedy output:  {greedy.shape[1]} tokens: {greedy[0].tolist()}")
print(f"Temp=0.8:       {temp.shape[1]} tokens: {temp[0].tolist()}")
print(f"Top-k=5:        {topk.shape[1]} tokens: {topk[0].tolist()}")
</code></pre>

    <h3>৫. HuggingFace দিয়ে GPT-2 ব্যবহার</h3>
    <pre><code class="language-python">from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def generate_text(prompt, max_length=100, temperature=0.8,
                  top_k=50, top_p=0.9, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    return generated_texts

prompt = "The future of artificial intelligence"
results = generate_text(prompt, max_length=80)
for i, text in enumerate(results):
    print(f"Generated {i+1}:\n{text}\n")

# GPT-2 model info
print(f"GPT-2 parameters: {sum(p.numel() for p in model.parameters()):,}")
# ~117M for gpt2 (small)
</code></pre>

    <h3>৬. GPT বনাম BERT: কখন কোনটি?</h3>
    <table>
      <thead>
        <tr><th>বৈশিষ্ট্য</th><th>BERT</th><th>GPT</th></tr>
      </thead>
      <tbody>
        <tr><td>Architecture</td><td>Encoder-only</td><td>Decoder-only</td></tr>
        <tr><td>Context</td><td>Bidirectional</td><td>Left-to-right (causal)</td></tr>
        <tr><td>Pre-training</td><td>MLM + NSP</td><td>Causal LM (next token prediction)</td></tr>
        <tr><td>Text generation</td><td>দুর্বল</td><td>শক্তিশালী</td></tr>
        <tr><td>Classification</td><td>শক্তিশালী</td><td>সম্ভব (but not ideal)</td></tr>
        <tr><td>Understanding</td><td>শক্তিশালী (NER, QA)</td><td>কম শক্তিশালী</td></tr>
        <tr><td>Best use cases</td><td>Sentence classification, NER, QA</td><td>Text generation, completion, dialogue</td></tr>
      </tbody>
    </table>

    <h3>৭. সারসংক্ষেপ</h3>
    <ul>
      <li>GPT = Decoder-only Transformer, causal self-attention দিয়ে।</li>
      <li>Autoregressive generation: প্রতিটি step-এ একটি token produce করে।</li>
      <li>Temperature, top-k, top-p — generation diversity control করে।</li>
      <li>BERT = understanding tasks, GPT = generation tasks।</li>
      <li>Weight tying (lm_head ও embedding একই weight) — common optimization।</li>
    </ul>
    <p>পরের ও শেষ পর্বে দেখব Vision Transformer (ViT) — কীভাবে Transformer ছবি বুঝতে পারে।</p>
  `,
};
