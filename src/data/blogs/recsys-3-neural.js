export const recsys_3_neural = {
  slug: 'recsys-3-neural',
  title: 'Neural Collaborative Filtering: Deep Learning দিয়ে Recommendation',
  description: 'Embedding layers দিয়ে user ও item representation, MLP দিয়ে interaction learning, এবং PyTorch-এ সম্পূর্ণ NCF implementation।',
  date: 'মে ২০২৬',
  category: 'রেকমেন্ডার সিস্টেম',
  readTime: 14,
  content: `
    <h3>১. কেন Neural CF?</h3>
    <p>Traditional Matrix Factorization শুধু linear interactions capture করে (dot product)। Neural CF non-linear, complex user-item relationships শিখতে পারে।</p>
    <p><strong>NCF Framework:</strong> User ও Item-কে Embedding layer দিয়ে dense vector-এ রূপান্তর করে, তারপর MLP দিয়ে interaction model করা হয়।</p>

    <pre><code class="language-python">import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
    def __len__(self): return len(self.ratings)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64, 32, 16]):
        super().__init__()
        # GMF path: element-wise product (linear interactions)
        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(n_items, emb_dim)
        # MLP path: deep non-linear interactions
        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.item_emb_mlp = nn.Embedding(n_items, emb_dim)
        # MLP layers
        mlp_layers = []
        in_size = emb_dim * 2
        for out_size in layers:
            mlp_layers += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(0.2)]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_layers)
        # Final prediction: GMF output + MLP output
        self.output = nn.Linear(emb_dim + layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user, item):
        # GMF: element-wise product
        gmf_out = self.user_emb_gmf(user) * self.item_emb_gmf(item)
        # MLP: concatenate then deep network
        mlp_in = torch.cat([self.user_emb_mlp(user), self.item_emb_mlp(item)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        # Combine and predict
        out = self.output(torch.cat([gmf_out, mlp_out], dim=-1))
        return self.sigmoid(out).squeeze()

def train_ncf():
    # Generate synthetic data
    np.random.seed(42)
    n_users, n_items = 500, 200
    n_samples = 5000

    users = np.random.randint(0, n_users, n_samples)
    items = np.random.randint(0, n_items, n_samples)
    # Normalize ratings to [0, 1]
    ratings = np.random.uniform(1, 5, n_samples).astype(np.float32)
    ratings_norm = (ratings - 1) / 4

    u_tr, u_te, i_tr, i_te, r_tr, r_te = train_test_split(
        users, items, ratings_norm, test_size=0.2, random_state=42)

    train_loader = DataLoader(RatingDataset(u_tr, i_tr, r_tr), batch_size=256, shuffle=True)
    test_loader  = DataLoader(RatingDataset(u_te, i_te, r_te), batch_size=256)

    model = NCF(n_users, n_items, emb_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(15):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            pred = model(u, i)
            loss = criterion(pred, r)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for u, i, r in test_loader:
                pred = model(u, i)
                test_loss += criterion(pred, r).item()
        if (epoch+1) % 5 == 0:
            train_rmse = np.sqrt(total_loss/len(train_loader)) * 4
            test_rmse  = np.sqrt(test_loss/len(test_loader)) * 4
            print(f"Epoch {epoch+1}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
    return model

model = train_ncf()
</code></pre>

    <h3>২. Implicit Feedback NCF (Binary Classification)</h3>
    <pre><code class="language-python">class NCFImplicit(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, layers=[64, 32, 16]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        in_size = emb_dim * 2
        mlp = []
        for out_size in layers:
            mlp += [nn.Linear(in_size, out_size), nn.BatchNorm1d(out_size), nn.ReLU(), nn.Dropout(0.2)]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp)
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.output(self.mlp(x)).squeeze()

# Negative sampling: for each positive interaction, sample k negatives
def negative_sampling(pos_pairs, n_items, k=4):
    samples = []
    item_set = set(i for _, i in pos_pairs)
    for u, i in pos_pairs:
        samples.append((u, i, 1.0))  # positive
        neg_count = 0
        while neg_count < k:
            neg_item = np.random.randint(n_items)
            if neg_item not in item_set:
                samples.append((u, neg_item, 0.0))
                neg_count += 1
    return samples

print("NCF with implicit feedback and negative sampling ready")
</code></pre>
  `
};
