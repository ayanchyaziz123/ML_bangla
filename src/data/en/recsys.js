export const recsysEn = [
  {
    slug: 'recsys-1-basics',
    title: 'Recommender Systems: Collaborative & Content-Based Filtering',
    description: 'User-item matrix, cosine similarity, user-based and item-based collaborative filtering, content-based recommendations',
    category: 'Recommender Systems',
    content: `
<h3>Types of Recommender Systems</h3>
<ul>
<li><strong>Collaborative Filtering:</strong> "Users like you also liked..." — based on interaction patterns</li>
<li><strong>Content-Based:</strong> "Because you liked X (similar features)..." — based on item attributes</li>
<li><strong>Hybrid:</strong> Combines both (Netflix, Spotify)</li>
</ul>

<h3>User-Based Collaborative Filtering</h3>
<pre><code class="language-python">import pandas as pd, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame({
    'Alice': [5, 4, 0, 0, 1],
    'Bob':   [4, 0, 4, 1, 0],
    'Carol': [0, 0, 5, 4, 0],
    'Dave':  [0, 1, 0, 5, 4],
}, index=['Titanic','Inception','Interstellar','The Matrix','Avengers'])

user_sim = pd.DataFrame(cosine_similarity(ratings.T),
    index=ratings.columns, columns=ratings.columns)

def predict(user, movie, n_neighbors=2):
    rated = ratings.columns[ratings.loc[movie] != 0].tolist()
    sims = user_sim.loc[user, rated].nlargest(n_neighbors)
    return (sims * ratings.loc[movie, sims.index]).sum() / sims.abs().sum()

print(f"Alice's predicted rating for Interstellar: {predict('Alice', 'Interstellar'):.2f}")
</code></pre>

<h3>Content-Based Filtering</h3>
<pre><code class="language-python">from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({'title': ['Titanic','Inception','Interstellar','The Matrix'],
    'genres': ['romance drama','thriller sci-fi','sci-fi space drama','sci-fi action']})
tfidf = TfidfVectorizer()
sim = cosine_similarity(tfidf.fit_transform(movies['genres']))
idx = movies[movies['title']=='Interstellar'].index[0]
scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:4]
print([(movies['title'][i], round(s,3)) for i, s in scores])
</code></pre>
`
  },
  {
    slug: 'recsys-2-matrix',
    title: 'Matrix Factorization: SVD and ALS',
    description: 'Funk SVD, latent factors, regularization, implicit feedback with ALS, Surprise library',
    category: 'Recommender Systems',
    content: `
<h3>Matrix Factorization</h3>
<p>Decompose the user-item matrix R (m×n) into P (m×k) and Q (n×k). Each latent factor k captures a hidden concept (e.g., "action preference", "genre taste").</p>
<p><strong>Loss = Σ(r_ui - p_u·q_i)² + λ(||p_u||² + ||q_i||²)</strong></p>

<pre><code class="language-python">from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
algo.fit(trainset)
preds = algo.test(testset)
print(f"RMSE: {accuracy.rmse(preds):.4f}, MAE: {accuracy.mae(preds):.4f}")

# Top-N recommendations
from collections import defaultdict
def top_n(predictions, n=10):
    user_recs = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_recs[uid].append((iid, est))
    return {u: sorted(r, key=lambda x: x[1], reverse=True)[:n] for u, r in user_recs.items()}
</code></pre>

<h3>ALS for Implicit Feedback</h3>
<pre><code class="language-python">from implicit import als
import scipy.sparse as sparse, numpy as np

interactions = sparse.random(1000, 500, density=0.05, format='csr')
interactions.data = np.random.randint(1, 20, interactions.nnz).astype('float32')

model = als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
model.fit(interactions)
ids, scores = model.recommend(0, interactions[0], N=5)
print("Top 5 for user 0:", list(zip(ids, scores.round(3))))
</code></pre>
`
  },
  {
    slug: 'recsys-3-neural',
    title: 'Neural Collaborative Filtering with Embeddings',
    description: 'NCF architecture, GMF path, MLP path, negative sampling, PyTorch implementation',
    category: 'Recommender Systems',
    content: `
<h3>Neural CF Architecture</h3>
<p>NCF combines two paths: GMF (Generalized Matrix Factorization — element-wise product) and MLP (deep non-linear interactions), then fuses for final prediction.</p>

<pre><code class="language-python">import torch, torch.nn as nn, torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64,32,16]):
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)
        mlp, in_dim = [], emb_dim * 2
        for out_dim in layers:
            mlp += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp)
        self.out = nn.Linear(emb_dim + layers[-1], 1)

    def forward(self, u, i):
        gmf = self.user_gmf(u) * self.item_gmf(i)
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=-1))
        return torch.sigmoid(self.out(torch.cat([gmf, mlp], dim=-1))).squeeze()

model = NCF(1000, 500)
u, i = torch.randint(0, 1000, (32,)), torch.randint(0, 500, (32,))
print(f"Output shape: {model(u, i).shape}")  # (32,)
</code></pre>

<h3>Negative Sampling</h3>
<pre><code class="language-python">def negative_sample(pos_pairs, n_items, k=4):
    samples = []
    pos_set = set(i for _, i in pos_pairs)
    for u, i in pos_pairs:
        samples.append((u, i, 1.0))
        for _ in range(k):
            neg = np.random.randint(n_items)
            while neg in pos_set: neg = np.random.randint(n_items)
            samples.append((u, neg, 0.0))
    return samples
</code></pre>
`
  },
  {
    slug: 'recsys-4-deep',
    title: 'Two-Tower Model: Industry-Scale Retrieval',
    description: 'Two-tower architecture, in-batch negatives, FAISS offline serving, feature engineering for recommenders',
    category: 'Recommender Systems',
    content: `
<h3>Two-Stage Recommendation Pipeline</h3>
<ol>
<li><strong>Retrieval:</strong> Two-Tower → FAISS ANN → 100-500 candidates (ms latency)</li>
<li><strong>Ranking:</strong> Wide&amp;Deep / DCN → precise scores → top 10-50</li>
</ol>

<pre><code class="language-python">import torch, torch.nn as nn, torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, in_dim, hidden=[128,64], out_dim=32):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, emb_dim=32):
        super().__init__()
        self.user_tower = Tower(user_dim, out_dim=emb_dim)
        self.item_tower = Tower(item_dim, out_dim=emb_dim)

def in_batch_loss(u_emb, i_emb, tau=0.07):
    logits = (u_emb @ i_emb.T) / tau
    labels = torch.arange(len(u_emb))
    return F.cross_entropy(logits, labels)
</code></pre>

<h3>FAISS Offline Serving</h3>
<pre><code class="language-python">import faiss, numpy as np

# Pre-compute all item embeddings (run nightly)
model = TwoTower(20, 15, emb_dim=32)
item_feats = torch.randn(10000, 15)
with torch.no_grad():
    item_emb = F.normalize(model.item_tower(item_feats), dim=-1).numpy().astype('float32')

index = faiss.IndexFlatIP(32)
index.add(item_emb)

# At query time
def recommend(user_feats, k=50):
    with torch.no_grad():
        u = F.normalize(model.user_tower(torch.FloatTensor(user_feats).unsqueeze(0)), dim=-1)
    scores, ids = index.search(u.numpy().astype('float32'), k)
    return list(zip(ids[0], scores[0]))
</code></pre>
`
  },
  {
    slug: 'recsys-5-project',
    title: 'RecSys Project: MovieLens End-to-End',
    description: 'Full pipeline comparison of CF, SVD, and NCF on MovieLens 100K — NDCG@K, Precision@K, MAP evaluation',
    category: 'Recommender Systems',
    content: `
<h3>Complete Comparison Pipeline</h3>
<pre><code class="language-python">from surprise import SVD, KNNBasic, Dataset, accuracy
from surprise.model_selection import train_test_split
import numpy as np
from collections import defaultdict

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

for name, algo in [
    ('User-CF', KNNBasic(sim_options={'user_based': True})),
    ('Item-CF', KNNBasic(sim_options={'user_based': False})),
    ('SVD',     SVD(n_factors=100, n_epochs=20, random_state=42)),
]:
    algo.fit(trainset)
    preds = algo.test(testset)
    print(f"{name}: RMSE={accuracy.rmse(preds,verbose=False):.4f}, MAE={accuracy.mae(preds,verbose=False):.4f}")
</code></pre>

<h3>NDCG@K Evaluation</h3>
<pre><code class="language-python">def ndcg_at_k(predictions, k=10):
    user_preds = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_preds[uid].append((est, true_r))
    scores = []
    for uid, ratings in user_preds.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        dcg  = sum((2**r-1)/np.log2(i+2) for i,(_, r) in enumerate(ratings[:k]))
        idcg = sum((2**r-1)/np.log2(i+2) for i, r in enumerate(sorted([r for _,r in ratings],reverse=True)[:k]))
        scores.append(dcg/idcg if idcg > 0 else 0)
    return np.mean(scores)

svd = SVD(n_factors=100, random_state=42)
svd.fit(trainset)
preds = svd.test(testset)
print(f"NDCG@10: {ndcg_at_k(preds):.4f}")
</code></pre>

<h3>Cold Start Strategy</h3>
<table>
<tr><th>Interactions</th><th>Strategy</th></tr>
<tr><td>0 (new user)</td><td>Popularity-based</td></tr>
<tr><td>1–5</td><td>Content-based on interacted items</td></tr>
<tr><td>5+</td><td>Full collaborative filtering</td></tr>
</table>
`
  },
];
