export const transformerEn = [
  {
    slug: 'transformer-1-attention',
    title: 'Attention Mechanism: Foundation of Transformers',
    description: 'How attention works, why RNNs fail for long sequences, and the scaled dot-product attention formula',
    category: 'Transformers',
    content: `
<h3>Why Attention?</h3>
<p>Recurrent Neural Networks process sequences step-by-step, struggling with long-range dependencies. The attention mechanism solves this by allowing every position to directly attend to every other position.</p>

<h3>Query, Key, Value Intuition</h3>
<p>Think of attention as a soft database lookup: a <strong>Query</strong> (what we're looking for) is matched against <strong>Keys</strong> (index), and the result is a weighted sum of <strong>Values</strong> (content).</p>

<h3>Scaled Dot-Product Attention</h3>
<pre><code class="language-python">import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V, weights

# Example
d_k = 64
Q = np.random.randn(5, d_k)
K = np.random.randn(10, d_k)
V = np.random.randn(10, 64)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights sum: {attn_weights.sum(axis=-1)}")
</code></pre>

<p>The <code>sqrt(d_k)</code> scaling prevents dot products from growing large in high dimensions, keeping softmax gradients well-behaved.</p>

<h3>PyTorch Implementation</h3>
<pre><code class="language-python">import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
</code></pre>
`
  },
  {
    slug: 'transformer-2-self-attention',
    title: 'Self-Attention and Multi-Head Attention',
    description: 'Deep dive into self-attention mechanics, multi-head attention, and positional encoding',
    category: 'Transformers',
    content: `
<h3>Self-Attention</h3>
<p>In self-attention, Q, K, and V all come from the same sequence. Each token attends to every other token in the same sequence, capturing contextual relationships.</p>

<pre><code class="language-python">import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
</code></pre>

<h3>Multi-Head Attention</h3>
<p>Run attention h times in parallel with different learned projections, then concatenate:</p>
<pre><code class="language-python">class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.split_heads(self.W_Q(x), batch_size)
        K = self.split_heads(self.W_K(x), batch_size)
        V = self.split_heads(self.W_V(x), batch_size)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_O(context)
</code></pre>
`
  },
  {
    slug: 'transformer-3-architecture',
    title: 'Transformer Architecture: Encoder-Decoder',
    description: 'Complete transformer with encoder stack, decoder stack, positional encoding, and layer normalization',
    category: 'Transformers',
    content: `
<h3>Positional Encoding</h3>
<p>Since attention has no inherent notion of order, we inject positional information using sinusoidal functions:</p>
<pre><code class="language-python">import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
</code></pre>

<h3>Encoder Block</h3>
<pre><code class="language-python">class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, src_mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.feed_forward(x))
        return x
</code></pre>
`
  },
  {
    slug: 'transformer-4-bert',
    title: 'BERT: Masked Language Modeling',
    description: 'BERT pre-training with masked LM, next sentence prediction, fine-tuning, and HuggingFace usage',
    category: 'Transformers',
    content: `
<h3>BERT Architecture</h3>
<p>BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack. It reads text in both directions simultaneously, unlike GPT which is left-to-right only.</p>

<h3>Pre-training Tasks</h3>
<ul>
<li><strong>Masked Language Modeling (MLM):</strong> 15% of tokens are masked; model predicts them</li>
<li><strong>Next Sentence Prediction (NSP):</strong> Predict if sentence B follows sentence A</li>
</ul>

<pre><code class="language-python">from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize input
text = "The movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1)
    print(f"Prediction: {'Positive' if pred.item() == 1 else 'Negative'}")
</code></pre>

<h3>Fine-tuning for Sentiment Analysis</h3>
<pre><code class="language-python">from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

dataset = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./bert_sentiment',
    num_train_epochs=3, per_device_train_batch_size=16,
    evaluation_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized['train'], eval_dataset=tokenized['test'])
trainer.train()
</code></pre>
`
  },
  {
    slug: 'transformer-5-gpt',
    title: 'GPT: Autoregressive Language Models',
    description: 'GPT decoder-only architecture, causal self-attention, and text generation with HuggingFace',
    category: 'Transformers',
    content: `
<h3>GPT vs BERT</h3>
<table>
<tr><th>Feature</th><th>BERT</th><th>GPT</th></tr>
<tr><td>Architecture</td><td>Encoder only</td><td>Decoder only</td></tr>
<tr><td>Attention</td><td>Bidirectional</td><td>Causal (left-to-right)</td></tr>
<tr><td>Pre-training</td><td>Masked LM + NSP</td><td>Next token prediction</td></tr>
<tr><td>Best for</td><td>Classification, NER</td><td>Text generation</td></tr>
</table>

<h3>Causal Self-Attention</h3>
<pre><code class="language-python">import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # Causal mask: upper triangular = -inf
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = self.mask[:seq_len, :seq_len]
        out, _ = self.attention(x, x, x, attn_mask=causal_mask.float() * -1e9)
        return out
</code></pre>

<h3>Text Generation with GPT-2</h3>
<pre><code class="language-python">from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Machine learning is transforming"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id
)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
</code></pre>
`
  },
  {
    slug: 'transformer-6-vit',
    title: 'Vision Transformer (ViT): Transformers for Images',
    description: 'Patch embeddings, positional encoding for images, ViT vs CNN comparison, and HuggingFace ViT',
    category: 'Transformers',
    content: `
<h3>From Text to Images</h3>
<p>ViT splits an image into fixed-size patches (e.g., 16×16 pixels), flattens them, and treats each patch as a "token" — just like words in a sentence.</p>

<h3>Patch Embedding</h3>
<pre><code class="language-python">import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # Single conv replaces flatten + linear
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])  # CLS token
</code></pre>

<h3>Using HuggingFace ViT</h3>
<pre><code class="language-python">from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

image = Image.open("cat.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")

import torch
with torch.no_grad():
    outputs = model(**inputs)
    pred_class = outputs.logits.argmax(-1).item()
    print(f"Predicted: {model.config.id2label[pred_class]}")
</code></pre>
`
  },
];
