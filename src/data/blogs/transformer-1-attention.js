export const transformer_1_attention = {
  slug: 'transformer-1-attention',
  title: 'অ্যাটেনশন মেকানিজম: ট্রান্সফর্মারের ভিত্তি',
  description: 'অ্যাটেনশন মেকানিজম কীভাবে কাজ করে এবং কেন এটি আধুনিক NLP-এর মূল ভিত্তি — Query-Key-Value intuition ও Scaled Dot-Product Attention সম্পূর্ণ গণিত সহ।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 15,
  content: `
    <h3>১. RNN-এর সীমাবদ্ধতা: কেন অ্যাটেনশন দরকার হলো?</h3>
    <p>২০১৭ সালের আগে পর্যন্ত NLP-এর রাজত্ব ছিল <strong>Recurrent Neural Network (RNN)</strong> এবং <strong>LSTM</strong>-এর। কিন্তু এদের একটি মৌলিক সমস্যা ছিল — <strong>দীর্ঘ sequence-এ তথ্য হারিয়ে যায়।</strong></p>
    <p>ধরুন একটি বাংলা বাক্য: <em>"রাহেলা, যে গতকাল ঢাকা থেকে এসেছিল এবং সারাদিন অফিসে কাজ করেছিল, সে এখন ক্লান্ত।"</em></p>
    <p>এই বাক্যে "সে" কে বোঝার জন্য মডেলকে অনেক পেছনে "রাহেলা" পর্যন্ত যেতে হবে। RNN-এ information একটি hidden state-এর মধ্য দিয়ে প্রবাহিত হয় — sequence লম্বা হলে প্রথম দিকের তথ্য <strong>vanish</strong> হয়ে যায়।</p>
    <p>RNN-এর আরও দুটি বড় সমস্যা:</p>
    <ul>
      <li><strong>Sequential processing:</strong> প্রতিটি token পর্যায়ক্রমে process করতে হয়, parallel করা যায় না — training অনেক ধীরগতি।</li>
      <li><strong>Fixed-size context vector:</strong> Encoder-Decoder architecture-এ পুরো input-কে একটি vector-এ compress করতে হয় — long sequence-এ bottleneck তৈরি হয়।</li>
    </ul>
    <p><strong>Attention mechanism</strong> এই সমস্যাগুলো সমাধান করে। প্রতিটি output token তৈরির সময় model সরাসরি input-এর যেকোনো position-এ "মনোযোগ" দিতে পারে।</p>

    <h3>২. Attention-এর মূল Intuition: Query, Key, Value</h3>
    <p>Attention mechanism-এর ধারণাটা একটি সহজ উপমা দিয়ে বোঝা যায় — <strong>Library search।</strong></p>
    <p>আপনি library-তে গিয়ে একটি বই খুঁজছেন (এটি আপনার <strong>Query</strong>)। Library-তে প্রতিটি বইয়ের সাথে একটি catalogue card আছে (এটি <strong>Key</strong>)। আপনি আপনার Query-কে প্রতিটি Key-এর সাথে match করেন, সবচেয়ে relevant বইগুলো বেশি মনোযোগ পায়। তারপর সেই relevant বইগুলোর content (এটি <strong>Value</strong>) combine করে উত্তর পান।</p>
    <table>
      <thead>
        <tr><th>Component</th><th>ভূমিকা</th><th>উদাহরণ</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Query (Q)</strong></td><td>আমি কী খুঁজছি?</td><td>Decoder-এর current state</td></tr>
        <tr><td><strong>Key (K)</strong></td><td>আমি কী offer করতে পারি?</td><td>Encoder-এর প্রতিটি token-এর identity</td></tr>
        <tr><td><strong>Value (V)</strong></td><td>আমার actual content কী?</td><td>Encoder-এর প্রতিটি token-এর information</td></tr>
      </tbody>
    </table>
    <p>Query ও Key-এর similarity দিয়ে attention weight নির্ধারিত হয়। High similarity = বেশি attention। তারপর সেই weights দিয়ে Value-গুলো weighted sum করা হয়।</p>

    <h3>৩. Scaled Dot-Product Attention: গণিত</h3>
    <p>Attention mechanism-এর formal definition:</p>
    <p><strong>Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V</strong></p>
    <p>প্রতিটি অংশ বিশ্লেষণ:</p>
    <ul>
      <li><strong>QKᵀ</strong> — Query matrix Q (shape: n×d_k) এবং Key matrix K (shape: m×d_k)-এর dot product। Result shape: n×m। প্রতিটি query-key pair-এর similarity score।</li>
      <li><strong>/ √d_k</strong> — Scaling factor। d_k হলো key dimension। এটি ছাড়া dot product-এর magnitude বড় হলে softmax-এর gradient vanish হতে পারে।</li>
      <li><strong>softmax(...)</strong> — Scores-কে probability distribution-এ convert করে। প্রতিটি row sum = 1। এটাই attention weight।</li>
      <li><strong>· V</strong> — Attention weights দিয়ে Value matrix-এর weighted sum।</li>
    </ul>
    <p>কেন √d_k দিয়ে scale করা হয়? ধরুন d_k = 64। দুটি random unit vector-এর dot product-এর variance হবে d_k = 64। তাই standard deviation = √64 = 8। Scale না করলে large values softmax-এ extreme হয়ে যায় এবং gradient ছোট হয়ে পড়ে।</p>

    <h3>৪. Python দিয়ে Attention Implement করা (NumPy)</h3>
    <pre><code class="language-python">import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    mask: optional (batch, seq_q, seq_k) boolean mask

    Returns:
        output: (batch, seq_q, d_v)
        attention_weights: (batch, seq_q, seq_k)
    """
    d_k = Q.shape[-1]

    # Step 1: QK^T / sqrt(d_k)
    # Q: (batch, seq_q, d_k), K^T: (batch, d_k, seq_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    # scores shape: (batch, seq_q, seq_k)

    # Step 2: Masking (optional) — future tokens mask করা
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Step 3: Softmax — attention weights
    attention_weights = softmax(scores, axis=-1)
    # attention_weights shape: (batch, seq_q, seq_k)

    # Step 4: Weighted sum of Values
    output = np.matmul(attention_weights, V)
    # output shape: (batch, seq_q, d_v)

    return output, attention_weights

# উদাহরণ: ছোট sequence দিয়ে test
batch_size = 1
seq_len = 5       # 5 tokens
d_k = 4           # key/query dimension
d_v = 4           # value dimension

np.random.seed(42)
Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Query shape:  {Q.shape}")       # (1, 5, 4)
print(f"Key shape:    {K.shape}")       # (1, 5, 4)
print(f"Value shape:  {V.shape}")       # (1, 5, 4)
print(f"Output shape: {output.shape}")  # (1, 5, 4)
print(f"\nAttention weights (প্রথম query-র জন্য):")
print(np.round(weights[0, 0, :], 4))
print(f"Sum: {weights[0, 0, :].sum():.4f}")  # সবসময় 1.0
</code></pre>

    <h3>৫. Attention Weight Visualization</h3>
    <p>একটি বাক্যে attention weight visualize করলে বোঝা যায় প্রতিটি token অন্য কোন token-কে বেশি গুরুত্ব দিচ্ছে। নিচের উদাহরণে দেখুন "it" token কীভাবে "animal" এর দিকে বেশি attend করে।</p>
    <pre><code class="language-python">import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_attention(tokens, attention_matrix, title="Attention Weights"):
    """
    tokens: list of token strings
    attention_matrix: (seq_len, seq_len) numpy array
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_matrix, cmap='Blues')

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Key (Source)')
    ax.set_ylabel('Query (Target)')
    ax.set_title(title)

    # Cell-এ value লেখা
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f'{attention_matrix[i,j]:.2f}',
                   ha='center', va='center', fontsize=9)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('attention_viz.png', dpi=100)
    print("attention_viz.png সেভ হয়েছে")
    return fig

# Toy example: "The animal did not cross because it was tired"
tokens = ["The", "animal", "did", "not", "cross", "it", "was", "tired"]
n = len(tokens)

# Random attention weights (real model থেকে নেওয়া হবে)
np.random.seed(10)
raw_scores = np.random.randn(n, n)

# "it" (index 5) কে "animal" (index 1) এর দিকে manually bias করি
raw_scores[5, 1] += 3.0

# Softmax
def softmax_2d(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

attn_weights = softmax_2d(raw_scores)

print("'it' token-এর attention weights:")
for tok, w in zip(tokens, attn_weights[5]):
    bar = '#' * int(w * 30)
    print(f"  {tok:10s}: {w:.3f} {bar}")
</code></pre>

    <h3>৬. Attention-এর তিন প্রকার</h3>
    <p>Transformer-এ তিন ধরনের attention ব্যবহার হয়:</p>
    <ul>
      <li><strong>Encoder Self-Attention:</strong> Q = K = V = encoder input। প্রতিটি input token অন্য সব input token-এর সাথে attend করে।</li>
      <li><strong>Decoder Self-Attention (Masked):</strong> Q = K = V = decoder input, কিন্তু future tokens mask করা হয়। Autoregressive generation নিশ্চিত করে।</li>
      <li><strong>Cross-Attention (Encoder-Decoder Attention):</strong> Q = decoder state, K = V = encoder output। Decoder জানে input sequence-এর কোন অংশ relevant।</li>
    </ul>
    <pre><code class="language-python">import numpy as np

def create_causal_mask(seq_len):
    """
    Decoder self-attention-এর জন্য causal (future) mask।
    position i তে শুধু position 0..i পর্যন্ত দেখতে পাবে।
    """
    # True মানে attend করতে পারবে, False মানে block
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask

mask = create_causal_mask(5)
print("Causal Mask (5x5):")
print(mask.astype(int))
# Output:
# [[1 0 0 0 0]
#  [1 1 0 0 0]
#  [1 1 1 0 0]
#  [1 1 1 1 0]
#  [1 1 1 1 1]]
# প্রথম row: শুধু position 0 দেখতে পায়
# শেষ row: সব position দেখতে পায়

# Masked attention test
seq_len = 5
d_k = 8
Q = np.random.randn(1, seq_len, d_k)
K = np.random.randn(1, seq_len, d_k)
V = np.random.randn(1, seq_len, d_k)

# batch dimension-এ mask expand করুন
mask_batch = mask[np.newaxis, :, :]  # (1, seq_len, seq_len)

output, weights = scaled_dot_product_attention(Q, K, V, mask=mask_batch)
print(f"\nMasked attention weights (upper triangle should be ~0):")
print(np.round(weights[0], 3))
</code></pre>

    <h3>৭. সারসংক্ষেপ ও পরবর্তী পদক্ষেপ</h3>
    <p>Attention mechanism-এর মূল কথা:</p>
    <ul>
      <li>RNN-এর sequential bottleneck এবং long-range dependency সমস্যা সমাধান করে।</li>
      <li><strong>Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V</strong> — এই সূত্রটি মনে রাখুন।</li>
      <li>Query-Key similarity দিয়ে কোথায় "মনোযোগ" দেবে ঠিক হয়, Value দিয়ে actual information নেওয়া হয়।</li>
      <li>√d_k scaling gradient vanishing রোধ করে।</li>
      <li>Masking দিয়ে decoder-এ future information leak ঠেকানো হয়।</li>
    </ul>
    <p>পরের পর্বে আমরা দেখব <strong>Multi-Head Attention</strong> — একই সাথে একাধিক attention head চালিয়ে বিভিন্ন ধরনের relationship capture করা।</p>
  `,
};
