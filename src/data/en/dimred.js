export const dimredEn = [
  {
    slug: 'dimred-1-pca',
    title: 'PCA: Principal Component Analysis',
    description: 'Curse of dimensionality, eigenvectors, explained variance, and Python implementation',
    category: 'Dimensionality Reduction',
    content: `
<h3>The Curse of Dimensionality</h3>
<p>As dimensions increase, data becomes increasingly sparse, distances lose meaning, and models overfit. PCA combats this by projecting data onto a lower-dimensional subspace that captures maximum variance.</p>

<h3>PCA Algorithm</h3>
<ol>
<li>Center the data: X_centered = X - mean(X)</li>
<li>Compute covariance matrix: C = X^T X / (n-1)</li>
<li>Compute eigenvectors and eigenvalues of C</li>
<li>Sort by eigenvalue (descending) — these are principal components</li>
<li>Project: Z = X_centered @ W (top-k eigenvectors)</li>
</ol>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(f"Original shape: {X.shape}")  # (1797, 64)

# Apply PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print(f"Reduced shape: {X_2d.shape}")

# Explained variance
pca_full = PCA()
pca_full.fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Components for 95% variance: {n_95}")

# Visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Digits Dataset - PCA 2D')
plt.show()
</code></pre>
`
  },
  {
    slug: 'dimred-2-svd',
    title: 'SVD: Singular Value Decomposition and PCA',
    description: 'Matrix factorization A=UΣVᵀ, its connection to PCA, and image compression',
    category: 'Dimensionality Reduction',
    content: `
<h3>Singular Value Decomposition</h3>
<p>Any matrix A (m×n) can be factored as A = UΣVᵀ where U is orthogonal (left singular vectors), Σ is diagonal (singular values), and Vᵀ is orthogonal (right singular vectors).</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load grayscale image
img = np.array(Image.open('photo.jpg').convert('L'), dtype=float)
print(f"Image shape: {img.shape}")

# SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# Reconstruct with k components
def reconstruct(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, k in zip(axes, [5, 20, 50, len(S)]):
    recon = reconstruct(U, S, Vt, k)
    ax.imshow(recon, cmap='gray')
    ratio = k * (img.shape[0] + img.shape[1] + 1) / img.size * 100
    ax.set_title(f'k={k}\n{ratio:.1f}% storage')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Connection to PCA: right singular vectors of X are eigenvectors of X^T X
</code></pre>
`
  },
  {
    slug: 'dimred-3-tsne',
    title: 't-SNE: Visualizing High-Dimensional Data',
    description: 'How t-SNE preserves local structure, perplexity parameter, and practical visualization tips',
    category: 'Dimensionality Reduction',
    content: `
<h3>t-SNE Intuition</h3>
<p>t-SNE (t-distributed Stochastic Neighbor Embedding) minimizes the KL divergence between high-dimensional joint probabilities and low-dimensional t-distribution probabilities, preserving local neighborhood structure.</p>

<pre><code class="language-python">from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

X, y = load_digits(return_X_y=True)

# t-SNE (perplexity controls local vs global balance)
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
            init='pca', random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title('Digits - t-SNE (perplexity=30)')
plt.show()

# Compare perplexity values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, perp in zip(axes, [5, 30, 100]):
    tsne_p = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_p = tsne_p.fit_transform(X)
    ax.scatter(X_p[:, 0], X_p[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    ax.set_title(f'Perplexity={perp}')
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'dimred-4-umap',
    title: 'UMAP: Fast and Powerful Dimensionality Reduction',
    description: 'Topological data analysis, UMAP algorithm, speed vs t-SNE, n_neighbors and min_dist',
    category: 'Dimensionality Reduction',
    content: `
<h3>UMAP vs t-SNE</h3>
<table>
<tr><th>Feature</th><th>t-SNE</th><th>UMAP</th></tr>
<tr><td>Speed</td><td>Slow</td><td>Fast</td></tr>
<tr><td>Scalability</td><td>~50K points</td><td>Millions</td></tr>
<tr><td>Global structure</td><td>Poor</td><td>Better preserved</td></tr>
<tr><td>Reproducibility</td><td>Varies with init</td><td>More stable</td></tr>
<tr><td>New point projection</td><td>No</td><td>Yes (transform)</td></tr>
</table>

<pre><code class="language-python">import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import time

X, y = load_digits(return_X_y=True)

# UMAP
start = time.time()
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X)
print(f"UMAP time: {time.time() - start:.2f}s")

plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar()
plt.title('Digits - UMAP')
plt.show()

# n_neighbors effect
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, n in zip(axes, [5, 15, 50]):
    reducer_n = umap.UMAP(n_neighbors=n, random_state=42)
    X_n = reducer_n.fit_transform(X)
    ax.scatter(X_n[:, 0], X_n[:, 1], c=y, cmap='tab10', s=5, alpha=0.6)
    ax.set_title(f'n_neighbors={n}')
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'dimred-5-project',
    title: 'Dimensionality Reduction: Comparative Project',
    description: 'Hands-on comparison of PCA, t-SNE, and UMAP on MNIST with classification pipeline',
    category: 'Dimensionality Reduction',
    content: `
<h3>Complete Comparison Pipeline</h3>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import time

# MNIST subset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data[:10000] / 255.0, mnist.target[:10000]

results = {}
for name, reducer, dims in [
    ('PCA-50',   PCA(n_components=50), 50),
    ('PCA-2',    PCA(n_components=2), 2),
    ('t-SNE-2',  TSNE(n_components=2, random_state=42), 2),
    ('UMAP-2',   umap.UMAP(n_components=2, random_state=42), 2),
    ('UMAP-10',  umap.UMAP(n_components=10, random_state=42), 10),
]:
    start = time.time()
    X_red = reducer.fit_transform(X)
    t = time.time() - start

    clf = LogisticRegression(max_iter=500)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_red, y, cv=3, scoring='accuracy')
    results[name] = {'time': t, 'accuracy': scores.mean(), 'dims': dims}

print(f"{'Method':<12} {'Dims':>5} {'Accuracy':>10} {'Time (s)':>10}")
print('-' * 40)
for name, res in results.items():
    print(f"{name:<12} {res['dims']:>5} {res['accuracy']:>10.3f} {res['time']:>10.1f}")
</code></pre>

<h4>Key Takeaways</h4>
<ul>
<li>PCA with 50 components retains most classification accuracy while reducing computation</li>
<li>2D embeddings are for visualization — classification performance drops</li>
<li>UMAP-10 often matches PCA-50 accuracy while using fewer dimensions</li>
<li>Use PCA as first step before t-SNE/UMAP for large datasets</li>
</ul>
`
  },
];
