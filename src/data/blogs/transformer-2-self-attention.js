export const transformer_2_self_attention = {
  slug: 'transformer-2-self-attention',
  title: 'সেলফ-অ্যাটেনশন ও মাল্টি-হেড অ্যাটেনশন',
  description: 'Self-attention mechanics, Multi-head attention-এ parallel attention heads, position-wise feed-forward network এবং সম্পূর্ণ Python implementation।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 16,
  content: `
    <h3>১. Self-Attention কী?</h3>
    <p><strong>Self-Attention</strong> হলো attention mechanism-এর বিশেষ রূপ যেখানে Query, Key এবং Value — তিনটিই একই sequence থেকে আসে। অর্থাৎ একটি sequence নিজেই নিজের সাথে attend করে।</p>
    <p>উদাহরণ: "The bank by the river bank was steep" — এখানে "bank" শব্দটি দুইবার আছে, দুটির অর্থ ভিন্ন। Self-attention-এ প্রতিটি "bank" অন্য সব শব্দের সাথে interact করতে পারে, ফলে context থেকে সঠিক অর্থ বোঝা যায়।</p>
    <p>প্রতিটি input token x_i থেকে তিনটি vector তৈরি হয় learned weight matrix দিয়ে:</p>
    <ul>
      <li><strong>q_i = x_i · W_Q</strong> — query vector</li>
      <li><strong>k_i = x_i · W_K</strong> — key vector</li>
      <li><strong>v_i = x_i · W_V</strong> — value vector</li>
    </ul>
    <p>W_Q, W_K, W_V হলো learnable weight matrices। Dimensions: x_i ∈ ℝ^d_model, W_Q/W_K ∈ ℝ^(d_model × d_k), W_V ∈ ℝ^(d_model × d_v)।</p>

    <h3>২. Self-Attention Step-by-Step</h3>
    <p>ধরুন আমাদের input sequence "I love NLP" — 3 tokens, প্রতিটি 4-dimensional embedding।</p>
    <pre><code class="language-python">import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

class SingleHeadSelfAttention:
    def __init__(self, d_model, d_k, d_v):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # Learned weight matrices (random initialization)
        np.random.seed(42)
        self.W_Q = np.random.randn(d_model, d_k) * 0.1
        self.W_K = np.random.randn(d_model, d_k) * 0.1
        self.W_V = np.random.randn(d_model, d_v) * 0.1

    def forward(self, X):
        """
        X: (seq_len, d_model) — input embeddings
        Returns: (seq_len, d_v) — context-aware representations
        """
        # Step 1: Query, Key, Value তৈরি
        Q = X @ self.W_Q   # (seq_len, d_k)
        K = X @ self.W_K   # (seq_len, d_k)
        V = X @ self.W_V   # (seq_len, d_v)

        # Step 2: Scaled dot-product scores
        scores = Q @ K.T / np.sqrt(self.d_k)   # (seq_len, seq_len)

        # Step 3: Softmax — attention weights
        attn_weights = softmax(scores)          # (seq_len, seq_len)

        # Step 4: Weighted sum of values
        output = attn_weights @ V              # (seq_len, d_v)

        return output, attn_weights

# Test
d_model, d_k, d_v = 4, 3, 3
seq_len = 3  # "I love NLP"

X = np.array([
    [0.1, 0.2, 0.3, 0.4],   # "I"
    [0.5, 0.6, 0.7, 0.8],   # "love"
    [0.9, 0.1, 0.2, 0.3],   # "NLP"
])

attn = SingleHeadSelfAttention(d_model, d_k, d_v)
output, weights = attn.forward(X)

print("Input X shape:", X.shape)
print("Output shape:", output.shape)
print("\nAttention weights matrix:")
print(np.round(weights, 3))
print("\n(প্রতিটি row sum = 1.0)")
print("Row sums:", np.round(weights.sum(axis=1), 4))
</code></pre>

    <h3>৩. Multi-Head Attention: কেন এবং কীভাবে?</h3>
    <p>Single-head attention-এ সমস্যা হলো মডেল একটি নির্দিষ্ট ধরনের relationship শেখে। কিন্তু language-এ বিভিন্ন ধরনের dependency আছে — syntactic, semantic, coreference ইত্যাদি।</p>
    <p><strong>Multi-Head Attention</strong>-এ h সংখ্যক attention head parallel-এ চালানো হয়, প্রতিটি head আলাদা W_Q, W_K, W_V দিয়ে। প্রতিটি head ভিন্ন ধরনের pattern শিখতে পারে।</p>
    <p>Multi-Head Attention formula:</p>
    <p><strong>MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O</strong></p>
    <p><strong>head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)</strong></p>
    <p>প্রতিটি head-এর dimension: d_k = d_v = d_model / h। তাই total computation single-head-এর মতোই।</p>
    <table>
      <thead>
        <tr><th>Parameter</th><th>মান (Original Transformer)</th><th>ব্যাখ্যা</th></tr>
      </thead>
      <tbody>
        <tr><td>d_model</td><td>512</td><td>Model dimension</td></tr>
        <tr><td>h (heads)</td><td>8</td><td>Attention heads সংখ্যা</td></tr>
        <tr><td>d_k = d_v</td><td>64</td><td>512 / 8 = 64 per head</td></tr>
        <tr><td>W_O shape</td><td>512 × 512</td><td>Output projection</td></tr>
      </tbody>
    </table>
    <pre><code class="language-python">import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads   # per-head dimension

        np.random.seed(0)
        scale = 0.1
        # প্রতিটি head-এর জন্য আলাদা weight (অথবা একসাথে একটি বড় matrix)
        self.W_Q = np.random.randn(d_model, d_model) * scale  # (d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale  # output projection

    def split_heads(self, X, batch_size):
        """
        X: (batch, seq, d_model) -> (batch, h, seq, d_k)
        """
        seq_len = X.shape[1]
        X = X.reshape(batch_size, seq_len, self.h, self.d_k)
        return X.transpose(0, 2, 1, 3)  # (batch, h, seq, d_k)

    def forward(self, Q_in, K_in, V_in):
        """
        Q_in, K_in, V_in: (batch, seq, d_model)
        """
        batch = Q_in.shape[0]

        # Linear projections
        Q = Q_in @ self.W_Q  # (batch, seq, d_model)
        K = K_in @ self.W_K
        V = V_in @ self.W_V

        # Split into h heads
        Q = self.split_heads(Q, batch)  # (batch, h, seq, d_k)
        K = self.split_heads(K, batch)
        V = self.split_heads(V, batch)

        # Scaled dot-product attention per head
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # (batch, h, seq_q, seq_k)
        attn_weights = softmax(scores, axis=-1)
        context = attn_weights @ V  # (batch, h, seq, d_k)

        # Concatenate heads: (batch, h, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose(0, 2, 1, 3)   # (batch, seq, h, d_k)
        context = context.reshape(batch, -1, self.d_model)

        # Final output projection
        output = context @ self.W_O  # (batch, seq, d_model)
        return output, attn_weights

# Test
batch, seq_len, d_model, num_heads = 2, 6, 8, 2
X = np.random.randn(batch, seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
output, weights = mha.forward(X, X, X)  # self-attention

print(f"Input:  {X.shape}")
print(f"Output: {output.shape}")         # (2, 6, 8) — same as input
print(f"Attn weights per head: {weights.shape}")  # (2, 2, 6, 6)
</code></pre>

    <h3>৪. Position-wise Feed-Forward Network</h3>
    <p>Multi-head attention-এর পর প্রতিটি Transformer layer-এ একটি <strong>Feed-Forward Network (FFN)</strong> থাকে। এটি প্রতিটি position-এ independently apply হয় (তাই "position-wise"):</p>
    <p><strong>FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2</strong></p>
    <p>Original Transformer-এ: d_model = 512, inner dimension d_ff = 2048 (4× বড়)। এই expansion-contraction pattern network-কে বেশি expressive করে।</p>
    <pre><code class="language-python">import numpy as np

class PositionWiseFFN:
    def __init__(self, d_model, d_ff):
        """
        d_model: input/output dimension (e.g. 512)
        d_ff:    inner dimension (e.g. 2048)
        """
        np.random.seed(1)
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        # Linear -> ReLU -> Linear
        hidden = self.relu(x @ self.W1 + self.b1)  # (batch, seq, d_ff)
        output = hidden @ self.W2 + self.b2          # (batch, seq, d_model)
        return output

# Test
batch, seq_len, d_model, d_ff = 2, 10, 8, 32
x = np.random.randn(batch, seq_len, d_model)
ffn = PositionWiseFFN(d_model, d_ff)
out = ffn.forward(x)
print(f"FFN Input:  {x.shape}")   # (2, 10, 8)
print(f"FFN Output: {out.shape}") # (2, 10, 8) — same shape

# Parameter count
params_ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model
print(f"FFN parameters: {params_ffn:,}")
# 8*32 + 32 + 32*8 + 8 = 256 + 32 + 256 + 8 = 552
</code></pre>

    <h3>৫. Layer Normalization ও Residual Connection</h3>
    <p>Transformer-এর প্রতিটি sub-layer-এর পর দুটি important operation হয়:</p>
    <p><strong>Add &amp; Norm: output = LayerNorm(x + SubLayer(x))</strong></p>
    <p><strong>Residual Connection</strong> (Add): Input x সরাসরি output-এ যোগ হয়। Gradient flow উন্নত করে, deep network train করা সহজ হয়।</p>
    <p><strong>Layer Normalization</strong> (Norm): প্রতিটি token-এর feature dimension-এ normalize করে (Batch Norm-এর মতো নয় — এখানে batch dimension-এ নয়, feature dimension-এ):</p>
    <p><strong>LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β</strong></p>
    <pre><code class="language-python">import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)   # learnable scale
        self.beta  = np.zeros(d_model)  # learnable shift

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Normalize along last (feature) dimension
        """
        mean = x.mean(axis=-1, keepdims=True)       # (batch, seq, 1)
        var  = x.var(axis=-1, keepdims=True)         # (batch, seq, 1)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta       # (batch, seq, d_model)

# Transformer encoder sub-layer pattern
class EncoderSubLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Sub-layer 1: Multi-Head Self-Attention + Add & Norm
        attn_out, _ = self.mha.forward(x, x, x)
        x = self.norm1.forward(x + attn_out)   # residual + layer norm

        # Sub-layer 2: FFN + Add & Norm
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)    # residual + layer norm

        return x

# Test
batch, seq_len, d_model = 2, 6, 8
x = np.random.randn(batch, seq_len, d_model)
layer = EncoderSubLayer(d_model, num_heads=2, d_ff=16)
out = layer.forward(x)
print(f"Encoder Sub-layer Input:  {x.shape}")
print(f"Encoder Sub-layer Output: {out.shape}")  # Same shape
</code></pre>

    <h3>৬. Multi-Head Attention কী শেখে?</h3>
    <p>গবেষণায় দেখা গেছে বিভিন্ন attention head ভিন্ন ভিন্ন linguistic pattern শেখে:</p>
    <ul>
      <li><strong>Head 1:</strong> Subject-verb agreement শেখে</li>
      <li><strong>Head 2:</strong> Coreference resolution (it → animal)</li>
      <li><strong>Head 3:</strong> Adjacent token relationships</li>
      <li><strong>Head 4:</strong> Long-range syntactic dependencies</li>
    </ul>
    <p>এই বৈচিত্র্যের কারণে Transformer single-head-এর চেয়ে অনেক বেশি expressive।</p>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই পর্বে আমরা দেখলাম:</p>
    <ul>
      <li>Self-attention-এ Q, K, V একই sequence থেকে তৈরি হয়।</li>
      <li>Multi-head attention h টি attention head parallel-এ চালায়, প্রতিটি d_model/h dimension-এ কাজ করে।</li>
      <li>Position-wise FFN প্রতিটি token-এর representation independently transform করে।</li>
      <li>Residual connection ও Layer Normalization deep network-এ stable training নিশ্চিত করে।</li>
    </ul>
    <p>পরের পর্বে পূর্ণ Transformer architecture — Encoder stack, Decoder stack এবং Positional Encoding দেখব।</p>
  `,
};
