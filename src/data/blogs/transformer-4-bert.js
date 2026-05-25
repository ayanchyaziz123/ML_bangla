export const transformer_4_bert = {
  slug: 'transformer-4-bert',
  title: 'BERT: মাস্কড ল্যাঙ্গুয়েজ মডেলিং',
  description: 'BERT architecture, Masked Language Modeling pre-training, Next Sentence Prediction, fine-tuning strategy এবং HuggingFace Transformers দিয়ে practical NLP application।',
  date: 'মে ২০২৬',
  category: 'ট্রান্সফর্মার',
  readTime: 17,
  content: `
    <h3>১. BERT কী এবং কেন এটি বৈপ্লবিক?</h3>
    <p>২০১৮ সালে Google-এর গবেষকরা <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong> প্রকাশ করেন। এটি NLP-এর ইতিহাসে একটি turning point।</p>
    <p>BERT-এর আগে language model ছিল <em>unidirectional</em> — বাম থেকে ডানে (GPT) বা ডান থেকে বামে। BERT প্রথমবার <strong>bidirectional</strong> context ব্যবহার করে — একটি token বোঝার জন্য তার বাম ও ডান উভয় দিকের context ব্যবহার করে।</p>
    <p>উদাহরণ: "He went to the [MASK] to deposit money" — এখানে [MASK] বোঝার জন্য "deposit money" (ডানে) ও "He went to" (বামে) দুটোই দরকার।</p>
    <p>BERT মূলত Transformer-এর <strong>Encoder-only</strong> অংশ ব্যবহার করে। দুটি variant:</p>
    <table>
      <thead>
        <tr><th>Model</th><th>Layers (N)</th><th>Heads (h)</th><th>d_model</th><th>Parameters</th></tr>
      </thead>
      <tbody>
        <tr><td>BERT-base</td><td>12</td><td>12</td><td>768</td><td>110M</td></tr>
        <tr><td>BERT-large</td><td>24</td><td>16</td><td>1024</td><td>340M</td></tr>
      </tbody>
    </table>

    <h3>২. BERT-এর তিন ধরনের Embedding</h3>
    <p>BERT-এ প্রতিটি token-এর final input embedding তিনটি embedding-এর sum:</p>
    <ul>
      <li><strong>Token Embedding:</strong> WordPiece vocabulary থেকে token-এর embedding (30,522 vocabulary size)।</li>
      <li><strong>Segment Embedding:</strong> Sentence A হলে E_A, Sentence B হলে E_B — দুটি sentence distinguish করতে।</li>
      <li><strong>Position Embedding:</strong> Sinusoidal নয়, learned positional embedding (max 512 position)।</li>
    </ul>
    <p>বিশেষ tokens: <strong>[CLS]</strong> — sequence-এর শুরুতে, classification task-এর জন্য। <strong>[SEP]</strong> — sentence separator।</p>

    <h3>৩. Pre-training Task ১: Masked Language Model (MLM)</h3>
    <p>MLM হলো BERT-এর প্রাথমিক pre-training objective। Input-এর 15% tokens randomly mask করা হয়, model-কে সেই tokens predict করতে হয়।</p>
    <p>কিন্তু সবসময় [MASK] token দিলে fine-tuning-এর সময় মিলবে না (কারণ real text-এ [MASK] নেই)। তাই 15% selected tokens-এর মধ্যে:</p>
    <ul>
      <li><strong>80%:</strong> [MASK] দিয়ে replace — "bank" → "[MASK]"</li>
      <li><strong>10%:</strong> Random token দিয়ে replace — "bank" → "apple"</li>
      <li><strong>10%:</strong> Unchanged রাখা — "bank" → "bank"</li>
    </ul>
    <pre><code class="language-python">import random

def apply_mlm_mask(tokens, mask_prob=0.15, mask_token="[MASK]", vocab=None):
    """
    BERT MLM masking strategy implement করা।
    tokens: list of token strings
    Returns: masked_tokens, labels (original tokens যেগুলো mask হয়েছে)
    """
    if vocab is None:
        vocab = ["[UNK]", "the", "cat", "sat", "on", "mat", "apple", "ran", "jumped"]

    masked_tokens = tokens.copy()
    labels = [None] * len(tokens)  # None = not masked

    for i, token in enumerate(tokens):
        # Special tokens skip করুন
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if random.random() < mask_prob:
            labels[i] = token  # Original token save করুন

            r = random.random()
            if r < 0.80:
                masked_tokens[i] = mask_token           # 80%: [MASK]
            elif r < 0.90:
                masked_tokens[i] = random.choice(vocab) # 10%: random
            # else: unchanged (10%)

    return masked_tokens, labels

# Example
random.seed(42)
sentence = ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]"]
masked, labels = apply_mlm_mask(sentence)

print("Original:", sentence)
print("Masked:  ", masked)
print("Labels:  ", labels)
print("\nMasked positions:")
for i, (orig, label) in enumerate(zip(masked, labels)):
    if label is not None:
        print(f"  position {i}: '{label}' -> '{orig}'")
</code></pre>

    <h3>৪. Pre-training Task ২: Next Sentence Prediction (NSP)</h3>
    <p>NSP task-এ model দুটি sentence দেওয়া হয়, model বলবে দ্বিতীয় sentence প্রথমটির পর আসে কিনা (IsNext / NotNext)।</p>
    <p>Training data-তে 50% ক্ষেত্রে consecutive sentences, 50% ক্ষেত্রে random sentence দেওয়া হয়।</p>
    <pre><code class="language-python">import torch
import torch.nn as nn
from transformers import BertTokenizer

# NSP input format দেখুন
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sent_a = "The cat sat on the mat."
sent_b_positive = "The mat was red and fluffy."   # IsNext
sent_b_negative = "Quantum physics is fascinating."  # NotNext

# Positive pair (IsNext)
enc_pos = tokenizer(sent_a, sent_b_positive,
                    return_tensors='pt', padding=True)
print("=== IsNext Pair ===")
print("Input IDs shape:", enc_pos['input_ids'].shape)
print("Tokens:", tokenizer.convert_ids_to_tokens(
    enc_pos['input_ids'][0].tolist()))
print("Token type IDs (segment):", enc_pos['token_type_ids'][0].tolist())
# 0 = Sentence A, 1 = Sentence B

# Negative pair (NotNext)
enc_neg = tokenizer(sent_a, sent_b_negative,
                    return_tensors='pt', padding=True)
print("\n=== NotNext Pair ===")
print("Tokens:", tokenizer.convert_ids_to_tokens(
    enc_neg['input_ids'][0].tolist()))
</code></pre>

    <h3>৫. HuggingFace দিয়ে BERT ব্যবহার</h3>
    <p>HuggingFace Transformers library ব্যবহার করে pre-trained BERT লোড করা এবং feature extraction করা অত্যন্ত সহজ।</p>
    <pre><code class="language-python">import torch
from transformers import BertTokenizer, BertModel

# Pre-trained BERT লোড
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_bert_embeddings(text, model, tokenizer):
    """
    Text-এর BERT embeddings বের করা।
    Returns: token embeddings ও [CLS] embedding
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt',
                       max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (batch, seq_len, 768)
    token_embeddings = outputs.last_hidden_state
    # [CLS] token = index 0, sentence-level representation
    cls_embedding = token_embeddings[:, 0, :]

    return token_embeddings, cls_embedding

text1 = "The bank by the river was steep."
text2 = "I deposited money at the bank."

emb1_tok, emb1_cls = get_bert_embeddings(text1, model, tokenizer)
emb2_tok, emb2_cls = get_bert_embeddings(text2, model, tokenizer)

print(f"Token embeddings shape: {emb1_tok.shape}")  # (1, seq_len, 768)
print(f"CLS embedding shape:    {emb1_cls.shape}")  # (1, 768)

# Cosine similarity — দুটি sentence কতটা similar?
cos_sim = torch.nn.functional.cosine_similarity(emb1_cls, emb2_cls)
print(f"\nCosine similarity between sentences: {cos_sim.item():.4f}")
</code></pre>

    <h3>৬. Fine-tuning: Downstream Tasks</h3>
    <p>BERT-এর শক্তি হলো pre-trained model-এ একটি small task-specific layer যোগ করে fine-tune করলেই excellent performance পাওয়া যায়।</p>
    <p>বিভিন্ন task-এ fine-tuning কৌশল:</p>
    <ul>
      <li><strong>Text Classification (Sentiment Analysis):</strong> [CLS] embedding-এর উপর Linear layer।</li>
      <li><strong>Named Entity Recognition (NER):</strong> প্রতিটি token embedding-এর উপর Linear layer।</li>
      <li><strong>Question Answering (SQuAD):</strong> Start ও End position predict করা।</li>
    </ul>
    <pre><code class="language-python">import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset

class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        # 768 = BERT-base hidden size
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # [CLS] token representation
        cls_out = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)             # (batch, num_classes)
        return logits

# Training loop example
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — transformer training-এ standard
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Fine-tuning tips:
# Learning rate: 2e-5 to 5e-5 (BERT paper recommendation)
# Epochs: 3-4 (over-fitting হয় বেশি হলে)
# Batch size: 16 বা 32
# Warmup steps: total steps-এর 10%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
texts = ["I love this movie!", "This was terrible.", "Great experience."]
labels_data = [1, 0, 1]  # positive/negative

encodings = tokenizer(texts, padding=True, truncation=True,
                      max_length=64, return_tensors='pt')
labels_tensor = torch.tensor(labels_data)

dataset = TensorDataset(
    encodings['input_ids'],
    encodings['attention_mask'],
    labels_tensor
)
loader = DataLoader(dataset, batch_size=2)

model = BertSentimentClassifier(num_classes=2)
# AdamW: weight decay সহ Adam (Transformer-এ standard)
optimizer = AdamW(model.parameters(), lr=2e-5)

print("Model architecture:")
print(f"  BERT parameters: {sum(p.numel() for p in model.bert.parameters()):,}")
print(f"  Classifier parameters: {sum(p.numel() for p in model.classifier.parameters()):,}")
</code></pre>

    <h3>৭. BERT-এর সীমাবদ্ধতা</h3>
    <p>BERT অত্যন্ত শক্তিশালী হলেও কিছু সীমাবদ্ধতা আছে:</p>
    <ul>
      <li><strong>Max sequence length 512:</strong> দীর্ঘ documents-এর জন্য সমস্যা।</li>
      <li><strong>Text generation-এ দুর্বল:</strong> Bidirectional হওয়ায় autoregressive generation করতে পারে না।</li>
      <li><strong>NSP task দুর্বল:</strong> RoBERTa দেখিয়েছে NSP ছাড়াও ভালো performance।</li>
      <li><strong>Computational cost:</strong> 110M parameters — inference slow।</li>
    </ul>
    <p>এই সীমাবদ্ধতা থেকে RoBERTa, DistilBERT, ALBERT, DeBERTa ইত্যাদি improved models এসেছে।</p>

    <h3>৮. সারসংক্ষেপ</h3>
    <ul>
      <li>BERT = Transformer Encoder-only, bidirectional context।</li>
      <li>MLM pre-training: 15% tokens mask করে predict করা।</li>
      <li>NSP pre-training: দুটি sentence consecutive কিনা।</li>
      <li>Fine-tuning: [CLS] embedding বা token embeddings-এর উপর task-specific head।</li>
      <li>HuggingFace-এ <code>BertModel.from_pretrained()</code> দিয়ে instantly ব্যবহার করা যায়।</li>
    </ul>
    <p>পরের পর্বে GPT — Decoder-only Transformer এবং autoregressive text generation।</p>
  `,
};
