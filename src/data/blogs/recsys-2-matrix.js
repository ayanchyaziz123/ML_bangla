export const recsys_2_matrix = {
  slug: 'recsys-2-matrix',
  title: 'Matrix Factorization: Latent Factor দিয়ে Recommendation',
  description: 'SVD, Funk SVD, ALS (Alternating Least Squares), regularization, implicit feedback handling — Netflix Prize-এর বিজয়ী পদ্ধতি বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রেকমেন্ডার সিস্টেম',
  readTime: 14,
  content: `
    <h3>১. Collaborative Filtering-এর সমস্যা</h3>
    <p>User-based CF-এর দুটি বড় সমস্যা:</p>
    <ul>
      <li><strong>Scalability:</strong> লক্ষ লক্ষ users-এর মধ্যে similarity compute করা expensive</li>
      <li><strong>Sparsity:</strong> Netflix-এ 99%+ ratings missing — cosine similarity বিশ্বস্ত নয়</li>
    </ul>
    <p><strong>Matrix Factorization</strong> এই সমস্যা সমাধান করে: User-Item matrix R (m×n) কে দুটি ছোট matrix-এ decompose করা: <strong>R ≈ P × Q<sup>T</sup></strong></p>
    <ul>
      <li>P (m×k): User-latent factor matrix</li>
      <li>Q (n×k): Item-latent factor matrix</li>
      <li>k = latent factors (e.g., 10-200), hidden features (genre preference, actor preference, etc.)</li>
    </ul>

    <h3>২. Funk SVD (Simon Funk, Netflix Prize)</h3>
    <p>2006 Netflix Prize-এ Simon Funk এই approach popular করেন। Gradient descent দিয়ে P ও Q optimize করা হয়।</p>
    <p><strong>Loss = Σ(r_ui - p_u · q_i)² + λ(||p_u||² + ||q_i||²)</strong></p>

    <pre><code class="language-python">import numpy as np
from sklearn.model_selection import train_test_split

# Sample ratings: (user_id, item_id, rating)
np.random.seed(42)
n_users, n_items = 100, 50
n_ratings = 1000

ratings = []
for _ in range(n_ratings):
    u = np.random.randint(n_users)
    i = np.random.randint(n_items)
    r = np.random.uniform(1, 5)
    ratings.append((u, i, r))

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

class MatrixFactorization:
    def __init__(self, n_users, n_items, k=20, lr=0.01, reg=0.02):
        self.P = np.random.normal(0, 0.1, (n_users, k))
        self.Q = np.random.normal(0, 0.1, (n_items, k))
        self.bu = np.zeros(n_users)  # user bias
        self.bi = np.zeros(n_items)  # item bias
        self.mu = 0                  # global mean
        self.lr = lr
        self.reg = reg

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def train(self, data, epochs=20):
        self.mu = np.mean([r for _, _, r in data])
        for epoch in range(epochs):
            np.random.shuffle(data)
            total_loss = 0
            for u, i, r in data:
                pred = self.predict(u, i)
                err = r - pred
                # SGD updates
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                total_loss += err**2
            if (epoch+1) % 5 == 0:
                rmse = self.evaluate(test_data)
                print(f"Epoch {epoch+1}: Train Loss={total_loss/len(data):.4f}, Test RMSE={rmse:.4f}")

    def evaluate(self, data):
        errors = [(r - self.predict(u, i))**2 for u, i, r in data]
        return np.sqrt(np.mean(errors))

mf = MatrixFactorization(n_users, n_items, k=20)
mf.train(train_data, epochs=20)
print(f"Final Test RMSE: {mf.evaluate(test_data):.4f}")
</code></pre>

    <h3>৩. Surprise Library দিয়ে Matrix Factorization</h3>
    <pre><code class="language-python">from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split as sur_split
import pandas as pd

# Load MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = sur_split(data, test_size=0.2, random_state=42)

# SVD (Matrix Factorization)
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
algo.fit(trainset)
predictions = algo.test(testset)

rmse = accuracy.rmse(predictions)
mae  = accuracy.mae(predictions)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Cross-validation
cv_results = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Get top-N recommendations for a user
from collections import defaultdict

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)
user_id = list(top_n.keys())[0]
print(f"\\nTop 5 recommendations for user {user_id}:")
for iid, score in top_n[user_id]:
    print(f"  Item {iid}: {score:.3f}")
</code></pre>

    <h3>৪. Implicit Feedback (views, clicks)</h3>
    <p>অনেক real-world system-এ explicit ratings নেই — শুধু clicks, views, purchases আছে। এই binary data-কে <strong>implicit feedback</strong> বলে।</p>
    <pre><code class="language-python">from implicit import als
import scipy.sparse as sparse
import numpy as np

# Implicit feedback: user-item interaction counts
n_users, n_items = 1000, 500
interactions = sparse.random(n_users, n_items, density=0.05, format='csr')
interactions.data = np.random.randint(1, 20, size=interactions.nnz).astype('float32')

# ALS (Alternating Least Squares) for implicit feedback
model = als.AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=20,
    use_gpu=False
)
model.fit(interactions)

# Recommend for user 0
user_id = 0
item_ids, scores = model.recommend(user_id, interactions[user_id], N=5)
print("Top 5 recommendations:", list(zip(item_ids, scores.round(3))))

# Similar items
similar_items = model.similar_items(0, N=5)
print("Items similar to item 0:", similar_items)
</code></pre>
  `
};
