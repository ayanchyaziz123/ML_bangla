export const dimred_4_umap = {
  slug: 'dimred-4-umap',
  title: 'UMAP: দ্রুত ও উন্নত ডাইমেনশন রিডাকশন',
  description: 'UMAP-এর topological intuition, t-SNE-র সাথে গতি ও গুণমান তুলনা, n_neighbors ও min_dist parameter, এবং Python umap-learn দিয়ে বাস্তব উদাহরণ।',
  date: 'মে ২০২৫',
  category: 'ডাইমেনশনালিটি রিডাকশন',
  readTime: 14,
  content: `
    <h3>১. UMAP কী?</h3>
    <p>
      UMAP (Uniform Manifold Approximation and Projection) ২০১৮ সালে Leland McInnes, John Healy এবং James Melville প্রস্তাব করেন। এটি t-SNE-এর একটি শক্তিশালী বিকল্প যা:
    </p>
    <ul>
      <li>অনেক দ্রুত (বড় ডেটাসেটে ১০-১০০ গুণ দ্রুত)</li>
      <li>Global structure আরও ভালো সংরক্ষণ করে</li>
      <li>Out-of-sample transform সমর্থন করে</li>
      <li>২ মাত্রার বেশিতেও কাজ করে (general dimensionality reduction)</li>
    </ul>
    <p>
      UMAP শুধু visualization নয়, general-purpose dimensionality reduction হিসেবেও ব্যবহার করা যায় — যেখানে PCA linear structure, t-SNE শুধু visualization, সেখানে UMAP দুটোর মাঝামাঝি।
    </p>

    <h3>২. UMAP-এর অন্তর্দৃষ্টি: Topological Data Analysis</h3>
    <p>
      UMAP-এর তাত্ত্বিক ভিত্তি <strong>Riemannian geometry</strong> এবং <strong>algebraic topology</strong>। কিন্তু সহজভাবে বোঝার জন্য এই intuition কাজে আসে:
    </p>
    <p>
      ধরুন আপনার ডেটা আসলে একটি উচ্চমাত্রার space-এর ভেতরে একটি lower-dimensional <strong>manifold</strong>-এ আছে। যেমন একটি বলের surface 2D manifold (3D space-এ)।
    </p>
    <p>
      UMAP এই manifold-এর shape (topology) শিখতে চায় এবং তারপর সেই shape কে ২ মাত্রায় "unfold" করে।
    </p>

    <h4>UMAP-এর মূল ধাপ:</h4>
    <ol>
      <li><strong>Graph তৈরি:</strong> উচ্চমাত্রায় প্রতিটি পয়েন্টের k nearest neighbors খোঁজো এবং weighted graph তৈরি করো। Weight হলো পড়শির fuzzy membership।</li>
      <li><strong>Low-dimensional layout:</strong> এই graph-এর একটি ২D layout তৈরি করো যা graph structure সংরক্ষণ করে। Attractive forces (neighbors কাছে টানে) এবং repulsive forces (non-neighbors দূরে ঠেলে) balance করে।</li>
    </ol>

    <h3>৩. মূল Hyperparameters</h3>

    <h4>n_neighbors (default: 15)</h4>
    <p>
      প্রতিটি পয়েন্টের কতজন nearest neighbor বিবেচনা করা হবে।
    </p>
    <ul>
      <li><strong>ছোট মান (2-10):</strong> Very local structure, fine-grained clusters।</li>
      <li><strong>বড় মান (50-200):</strong> More global structure, smoother embedding।</li>
    </ul>
    <p>t-SNE-এর perplexity-র মতো কিন্তু আরও intuitive।</p>

    <h4>min_dist (default: 0.1)</h4>
    <p>
      Low-dimensional space-এ পয়েন্টগুলো কতটা কাছে আসতে পারবে।
    </p>
    <ul>
      <li><strong>ছোট মান (0.0-0.1):</strong> Tight clusters, dense packing।</li>
      <li><strong>বড় মান (0.5-0.99):</strong> Loose structure, ছড়িয়ে পড়া।</li>
    </ul>

    <h4>metric (default: 'euclidean')</h4>
    <p>
      Distance metric। Options: 'euclidean', 'cosine', 'manhattan', 'correlation', 'hamming', ইত্যাদি।
    </p>

    <h3>৪. UMAP Install ও Basic Usage</h3>
    <pre><code># Install করুন
# pip install umap-learn

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Data লোড করুন
digits = load_digits()
X = digits.data
y = digits.target

# Standardize করুন
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP apply করুন
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=True
)

X_umap = reducer.fit_transform(X_scaled)
print(f"UMAP output shape: {X_umap.shape}")  # (1797, 2)

# ভিজুয়ালাইজ করুন
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                      c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter, label='Digit')
plt.title('UMAP Visualization of MNIST Digits')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. n_neighbors এবং min_dist-এর প্রভাব</h3>
    <pre><code># Hyperparameter grid দেখুন
n_neighbors_list = [5, 15, 50]
min_dist_list = [0.0, 0.1, 0.5]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, nn in enumerate(n_neighbors_list):
    for j, md in enumerate(min_dist_list):
        reducer = umap.UMAP(
            n_neighbors=nn,
            min_dist=md,
            random_state=42
        )
        X_emb = reducer.fit_transform(X_scaled[:800])

        axes[i, j].scatter(X_emb[:, 0], X_emb[:, 1],
                            c=y[:800], cmap='tab10', alpha=0.7, s=10)
        axes[i, j].set_title(f'n_neighbors={nn}, min_dist={md}')
        axes[i, j].axis('off')

plt.suptitle('UMAP: Effect of n_neighbors and min_dist', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. UMAP vs t-SNE: গতি তুলনা</h3>
    <pre><code>import time
from sklearn.manifold import TSNE
import umap

# বড় ডেটাসেট তৈরি করুন
from sklearn.datasets import make_blobs
np.random.seed(42)
X_large, y_large = make_blobs(n_samples=10000, n_features=50,
                               centers=10, random_state=42)
scaler = StandardScaler()
X_large_scaled = scaler.fit_transform(X_large)

# t-SNE timing
print("Running t-SNE...")
start = time.time()
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_large_scaled)
tsne_time = time.time() - start
print(f"t-SNE time: {tsne_time:.2f}s")

# UMAP timing
print("Running UMAP...")
start = time.time()
reducer = umap.UMAP(n_components=2, n_neighbors=15,
                    min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_large_scaled)
umap_time = time.time() - start
print(f"UMAP time: {umap_time:.2f}s")

print(f"\nUMAP is {tsne_time/umap_time:.1f}x faster than t-SNE")

# তুলনামূলক visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=y_large, cmap='tab10', alpha=0.3, s=5)
axes[0].set_title(f't-SNE ({tsne_time:.1f}s)')

axes[1].scatter(X_umap[:, 0], X_umap[:, 1],
                c=y_large, cmap='tab10', alpha=0.3, s=5)
axes[1].set_title(f'UMAP ({umap_time:.1f}s)')

plt.suptitle('t-SNE vs UMAP on 10,000 samples', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৭. Out-of-Sample Transform: UMAP-এর বিশেষ সুবিধা</h3>
    <pre><code># UMAP train data-তে fit করুন, test data-তে transform করুন
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Training data-তে fit
reducer = umap.UMAP(n_components=2, random_state=42)
X_train_umap = reducer.fit_transform(X_train)

# Test data-তে transform (fit ছাড়া)
X_test_umap = reducer.transform(X_test)

print(f"Train UMAP shape: {X_train_umap.shape}")
print(f"Test UMAP shape: {X_test_umap.shape}")

# Plot করুন
plt.figure(figsize=(10, 8))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
            c=y_train, cmap='tab10', alpha=0.4, s=10, label='Train')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
            c=y_test, cmap='tab10', alpha=1.0, s=50, marker='*', label='Test')
plt.title('UMAP: Out-of-Sample Transform')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# t-SNE করতে পারে না! এটি try করলে নতুন embedding তৈরি হয়
# tsne.transform(X_test)  # AttributeError!</code></pre>

    <h3>৮. UMAP supervised mode</h3>
    <pre><code># Labels ব্যবহার করে class-aware embedding
reducer_supervised = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

# Labels দিয়ে fit করুন
X_supervised = reducer_supervised.fit_transform(X_scaled, y=y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Unsupervised
reducer_unsup = umap.UMAP(n_components=2, random_state=42)
X_unsupervised = reducer_unsup.fit_transform(X_scaled)

scatter1 = axes[0].scatter(X_unsupervised[:, 0], X_unsupervised[:, 1],
                            c=y, cmap='tab10', alpha=0.7, s=10)
axes[0].set_title('UMAP Unsupervised')

scatter2 = axes[1].scatter(X_supervised[:, 0], X_supervised[:, 1],
                            c=y, cmap='tab10', alpha=0.7, s=10)
axes[1].set_title('UMAP Supervised (with labels)')

plt.colorbar(scatter2, ax=axes[1], label='Digit')
plt.suptitle('Unsupervised vs Supervised UMAP', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৯. UMAP vs t-SNE: সারসংক্ষেপ তুলনা</h3>
    <p>কোনটি ব্যবহার করবেন তা নির্ধারণ করতে এই তুলনা দেখুন:</p>
    <ul>
      <li><strong>গতি:</strong> UMAP অনেক দ্রুত, বিশেষত বড় ডেটাসেটে।</li>
      <li><strong>Global structure:</strong> UMAP বেশি ভালো সংরক্ষণ করে।</li>
      <li><strong>Local structure:</strong> দুটোই ভালো, t-SNE একটু বেশি।</li>
      <li><strong>Out-of-sample:</strong> শুধু UMAP সমর্থন করে।</li>
      <li><strong>General DR:</strong> UMAP (২-এর বেশি মাত্রায়), t-SNE নয়।</li>
      <li><strong>Reproducibility:</strong> UMAP বেশি stable।</li>
      <li><strong>Theoretical basis:</strong> UMAP শক্তিশালী (topology)।</li>
    </ul>
    <p>
      পরবর্তী ও শেষ পর্বে আমরা একটি পূর্ণ project করব যেখানে PCA, t-SNE এবং UMAP তিনটিই একসাথে তুলনা করব।
    </p>
  `
};
