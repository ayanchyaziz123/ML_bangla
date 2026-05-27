export const recsys_4_deep = {
  slug: 'recsys-4-deep',
  title: 'Two-Tower Model: Industry-Scale Recommendation',
  description: 'Two-Tower retrieval architecture, feature engineering for recommenders, Wide & Deep model, এবং real-world large-scale recommendation system design।',
  date: 'মে ২০২৬',
  category: 'রেকমেন্ডার সিস্টেম',
  readTime: 15,
  content: `
    <h3>১. Real-World Recommenders: দুটি স্তর</h3>
    <p>Netflix, YouTube, Amazon-এর recommendation pipeline সাধারণত দুটি পর্যায়ে হয়:</p>
    <ol>
      <li><strong>Retrieval (Candidate Generation):</strong> কোটি items থেকে কয়েক'শ relevant candidates খুব দ্রুত বের করা</li>
      <li><strong>Ranking:</strong> Candidates-কে detailed features দিয়ে rank করা</li>
    </ol>
    <p><strong>Two-Tower Model</strong> Retrieval stage-এ use হয়। User এবং Item-এর জন্য আলাদা neural network (tower) দিয়ে embeddings তৈরি করা হয়। High similarity = strong match।</p>

    <h3>২. Two-Tower Architecture</h3>
    <pre><code class="language-python">import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Tower(nn.Module):
    """Single tower: maps features to embedding space"""
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=32):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # L2 normalize for cosine similarity

class TwoTowerModel(nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, emb_dim=32):
        super().__init__()
        self.user_tower = Tower(user_feat_dim, [128, 64], emb_dim)
        self.item_tower = Tower(item_feat_dim, [128, 64], emb_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)  # Learnable temperature

    def forward(self, user_feats, item_feats):
        user_emb = self.user_tower(user_feats)
        item_emb = self.item_tower(item_feats)
        return user_emb, item_emb

    def similarity(self, user_feats, item_feats):
        u, i = self.forward(user_feats, item_feats)
        return (u * i).sum(dim=-1)  # Dot product similarity

class InteractionDataset(Dataset):
    def __init__(self, user_feats, item_feats, labels):
        self.user = torch.FloatTensor(user_feats)
        self.item = torch.FloatTensor(item_feats)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.user[idx], self.item[idx], self.labels[idx]

def in_batch_negative_loss(user_emb, item_emb, temperature=0.07):
    """
    In-batch negatives: all other items in the batch serve as negatives.
    Much more efficient than explicit negative sampling.
    """
    logits = torch.matmul(user_emb, item_emb.T) / temperature
    labels = torch.arange(len(user_emb), device=user_emb.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# Training
np.random.seed(42)
n_samples = 2000
user_feat_dim, item_feat_dim = 20, 15

user_feats = np.random.randn(n_samples, user_feat_dim).astype(np.float32)
item_feats = np.random.randn(n_samples, item_feat_dim).astype(np.float32)

model = TwoTowerModel(user_feat_dim, item_feat_dim, emb_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(InteractionDataset(user_feats, item_feats, np.ones(n_samples)),
                    batch_size=256, shuffle=True)

for epoch in range(10):
    total_loss = 0
    for u, i, _ in loader:
        u_emb, i_emb = model(u, i)
        loss = in_batch_negative_loss(u_emb, i_emb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
</code></pre>

    <h3>৩. Offline Serving with FAISS</h3>
    <pre><code class="language-python">import faiss

# Pre-compute all item embeddings (done offline, once)
n_items = 10000
all_item_feats = torch.FloatTensor(np.random.randn(n_items, item_feat_dim))

model.eval()
with torch.no_grad():
    # Process in batches for large catalogs
    item_embeddings = []
    for start in range(0, n_items, 512):
        batch = all_item_feats[start:start+512]
        _, emb = model(torch.zeros(len(batch), user_feat_dim), batch)
        item_embeddings.append(emb.numpy())
item_embeddings = np.vstack(item_embeddings).astype('float32')

# Build FAISS index for fast ANN search
index = faiss.IndexFlatIP(32)  # Inner product (dot similarity)
index.add(item_embeddings)
print(f"Index built: {index.ntotal} items")

# At query time: embed user, search index
def recommend(user_features, k=50):
    with torch.no_grad():
        u_emb, _ = model(torch.FloatTensor(user_features).unsqueeze(0),
                         torch.zeros(1, item_feat_dim))
    u_emb = u_emb.numpy().astype('float32')
    scores, indices = index.search(u_emb, k)
    return list(zip(indices[0], scores[0]))

recs = recommend(np.random.randn(user_feat_dim))
print(f"Top 5 candidates: {recs[:5]}")
</code></pre>

    <h3>৪. Feature Engineering for Recommenders</h3>
    <pre><code class="language-python">import pandas as pd, numpy as np

def build_user_features(user_df, interaction_df):
    features = user_df.copy()
    # Behavioral features
    stats = interaction_df.groupby('user_id').agg(
        n_interactions=('item_id', 'count'),
        avg_rating=('rating', 'mean'),
        rating_std=('rating', 'std'),
        days_active=('timestamp', lambda x: (x.max() - x.min()).days)
    ).fillna(0)
    features = features.join(stats, on='user_id')
    # Recency feature
    last_interaction = interaction_df.groupby('user_id')['timestamp'].max()
    features['days_since_last'] = (pd.Timestamp.now() - last_interaction).dt.days
    return features

print("Feature engineering pipeline ready")
print("Key user features: n_interactions, avg_rating, days_active, days_since_last")
print("Key item features: category, price_tier, avg_rating, n_ratings, days_since_added")
</code></pre>
  `
};
