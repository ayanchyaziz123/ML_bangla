export const dimred_2_svd = {
  slug: 'dimred-2-svd',
  title: 'SVD: ম্যাট্রিক্স ফ্যাক্টরাইজেশন ও PCA-র সম্পর্ক',
  description: 'Singular Value Decomposition (A=UΣVᵀ), PCA-র সাথে গভীর সম্পর্ক, truncated SVD বড় ম্যাট্রিক্সের জন্য এবং image compression-এর বাস্তব উদাহরণ।',
  date: 'মে ২০২৫',
  category: 'ডাইমেনশনালিটি রিডাকশন',
  readTime: 13,
  content: `
    <h3>১. SVD কী এবং কেন এটি গুরুত্বপূর্ণ?</h3>
    <p>
      Singular Value Decomposition বা SVD হলো লিনিয়ার অ্যালজেব্রার অন্যতম শক্তিশালী উপকরণ। এটি যেকোনো ম্যাট্রিক্সকে তিনটি বিশেষ ম্যাট্রিক্সের গুণফলে ভাঙতে পারে।
    </p>
    <p>
      SVD শুধু ডাইমেনশন রিডাকশনে নয়, আরও অনেক জায়গায় ব্যবহৃত হয়:
    </p>
    <ul>
      <li>Recommendation systems (Netflix, Amazon)</li>
      <li>Natural Language Processing (LSA - Latent Semantic Analysis)</li>
      <li>Image compression</li>
      <li>Pseudo-inverse computation</li>
      <li>Least squares problem সমাধান</li>
      <li>PCA-র efficient computation</li>
    </ul>

    <h3>২. SVD-এর গাণিতিক সংজ্ঞা: A = UΣVᵀ</h3>
    <p>
      যেকোনো m×n ম্যাট্রিক্স A-কে লেখা যায়:
    </p>
    <p>
      <strong>A = U Σ Vᵀ</strong>
    </p>
    <p>যেখানে:</p>
    <ul>
      <li><strong>U</strong>: m×m orthogonal matrix (left singular vectors)। U-এর কলামগুলো হলো AAᵀ-এর eigenvectors।</li>
      <li><strong>Σ (Sigma)</strong>: m×n diagonal matrix (singular values)। Diagonal elements σ₁ ≥ σ₂ ≥ ... ≥ 0 হ্রাসমান ক্রমে সাজানো।</li>
      <li><strong>Vᵀ</strong>: n×n orthogonal matrix-এর transpose (right singular vectors)। V-এর কলামগুলো হলো AᵀA-এর eigenvectors।</li>
    </ul>
    <p>
      <strong>Orthogonal matrix:</strong> UᵀU = UUᵀ = I (সরলভাবে: rows এবং columns গুলো unit vectors এবং পরস্পর লম্ব)।
    </p>

    <h4>ছোট উদাহরণ:</h4>
    <pre><code>import numpy as np

# একটি সহজ ম্যাট্রিক্স
A = np.array([[3, 1, 1],
              [-1, 3, 1]])
print("A shape:", A.shape)  # (2, 3)

# SVD করুন
U, sigma, Vt = np.linalg.svd(A, full_matrices=True)

print("U shape:", U.shape)       # (2, 2)
print("sigma:", sigma)            # [3.464, 3.162] (singular values)
print("Vt shape:", Vt.shape)     # (3, 3)

# Reconstruction করুন
Sigma = np.zeros(A.shape)
Sigma[:len(sigma), :len(sigma)] = np.diag(sigma)
A_reconstructed = U @ Sigma @ Vt

print("\nOriginal A:\n", A)
print("Reconstructed A:\n", A_reconstructed.round(10))</code></pre>

    <h3>৩. Geometric Interpretation</h3>
    <p>
      SVD-কে geometrically বোঝা যায় এভাবে: যেকোনো linear transformation কে তিনটি মৌলিক অপারেশনে ভাঙা যায়:
    </p>
    <ol>
      <li><strong>Vᵀ:</strong> Input space-এ একটি rotation (বা reflection)</li>
      <li><strong>Σ:</strong> Scaling (প্রতিটি axis আলাদাভাবে stretching)</li>
      <li><strong>U:</strong> Output space-এ আরেকটি rotation</li>
    </ol>
    <p>
      Singular values (σᵢ) বলে দেয় প্রতিটি dimension কতটুকু "গুরুত্বপূর্ণ"। বড় singular value = ঐ direction-এ বেশি information।
    </p>

    <h3>৪. Truncated SVD: Low-Rank Approximation</h3>
    <p>
      Full SVD-তে সব singular values রাখা হয়। কিন্তু যদি শুধু প্রথম k টি রাখি, তাহলে পাই <strong>best rank-k approximation</strong>:
    </p>
    <p>
      Aₖ = U[:, :k] × Σ[:k, :k] × Vᵀ[:k, :]
    </p>
    <p>
      Eckart-Young theorem প্রমাণ করে: এটিই হলো A-এর সবচেয়ে কাছের rank-k matrix (Frobenius norm অনুযায়ী)।
    </p>

    <h3>৫. Image Compression দিয়ে হাতে-কলমে SVD</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Grayscale ছবি তৈরি করুন (বা লোড করুন)
# এখানে একটি synthetic ছবি ব্যবহার করি
np.random.seed(42)
# একটি structured ছবি তৈরি করুন
x = np.linspace(0, 4 * np.pi, 256)
y = np.linspace(0, 4 * np.pi, 256)
X, Y = np.meshgrid(x, y)
img_array = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.float64)

print(f"Image shape: {img_array.shape}")  # (256, 256)
print(f"Original size: {img_array.nbytes / 1024:.1f} KB")

# SVD করুন
U, sigma, Vt = np.linalg.svd(img_array, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"sigma shape: {sigma.shape}")
print(f"Vt shape: {Vt.shape}")
print(f"Top 5 singular values: {sigma[:5].round(2)}")

# বিভিন্ন rank-এ reconstruct করুন
ranks = [1, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Original
axes[0].imshow(img_array, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original (rank=256)')
axes[0].axis('off')

for idx, k in enumerate(ranks):
    # Truncated SVD
    Uk = U[:, :k]
    sigma_k = sigma[:k]
    Vtk = Vt[:k, :]

    # Reconstruct
    img_approx = Uk @ np.diag(sigma_k) @ Vtk

    # Error calculate করুন
    error = np.linalg.norm(img_array - img_approx, 'fro')
    compression = (k * (256 + 256 + 1)) / (256 * 256) * 100

    axes[idx + 1].imshow(img_approx, cmap='gray', vmin=0, vmax=255)
    axes[idx + 1].set_title(f'k={k}\nCompression: {compression:.1f}%')
    axes[idx + 1].axis('off')

axes[-1].axis('off')
plt.suptitle('SVD Image Compression', fontsize=14)
plt.tight_layout()
plt.show()

# Singular values-এর decay দেখুন
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(sigma[:50])
plt.xlabel('Rank')
plt.ylabel('Singular Value')
plt.title('Singular Value Decay')
plt.grid(True)

plt.subplot(1, 2, 2)
cumulative_energy = np.cumsum(sigma**2) / np.sum(sigma**2) * 100
plt.plot(cumulative_energy[:100])
plt.axhline(y=95, color='r', linestyle='--', label='95%')
plt.axhline(y=99, color='g', linestyle='--', label='99%')
plt.xlabel('Rank')
plt.ylabel('Cumulative Energy (%)')
plt.title('Cumulative Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. SVD এবং PCA-র গভীর সম্পর্ক</h3>
    <p>
      PCA এবং SVD আসলে একই জিনিস, শুধু দৃষ্টিভঙ্গি আলাদা। চলুন প্রমাণ করি:
    </p>
    <p>
      Centered ডেটা matrix X-এর covariance matrix:
      C = XᵀX / (n-1)
    </p>
    <p>
      X-এর SVD: X = UΣVᵀ
    </p>
    <p>
      তাহলে:
      XᵀX = (UΣVᵀ)ᵀ(UΣVᵀ) = VΣᵀUᵀUΣVᵀ = VΣ²Vᵀ
    </p>
    <p>
      অর্থাৎ C = VΣ²Vᵀ/(n-1)
    </p>
    <p>
      এটি covariance matrix-এর eigendecomposition! তার মানে:
    </p>
    <ul>
      <li>V-এর columns = PCA-র eigenvectors (principal components)</li>
      <li>σᵢ² / (n-1) = i-তম eigenvalue = i-তম PC-এর variance</li>
      <li>U-এর columns = normalized principal component scores</li>
    </ul>
    <pre><code>from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data

# Center করুন
X_centered = X - X.mean(axis=0)

# Method 1: SVD সরাসরি
U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
print("SVD right singular vectors (V columns):")
print(Vt.T[:, :2].round(4))

# Method 2: PCA
pca = PCA(n_components=2)
pca.fit(X_centered)
print("\nPCA components:")
print(pca.components_.T.round(4))

# SVD-based PCA scores
Z_svd = X_centered @ Vt.T[:, :2]
Z_pca = pca.transform(X_centered)

# দুটো প্রায় একই (sign ছাড়া)
print("\nMax difference:", np.max(np.abs(np.abs(Z_svd) - np.abs(Z_pca))))</code></pre>

    <h3>৭. Randomized SVD: বড় ডেটার জন্য</h3>
    <p>
      Full SVD-এর complexity O(min(m,n)×m×n)। বড় ম্যাট্রিক্সের জন্য এটি অনেক ধীর। <strong>Randomized SVD</strong> এই সমস্যা সমাধান করে।
    </p>
    <pre><code>from sklearn.utils.extmath import randomized_svd
import time
import numpy as np

# বড় random matrix
np.random.seed(42)
big_matrix = np.random.randn(5000, 1000)

# Full SVD timing
start = time.time()
U_full, s_full, Vt_full = np.linalg.svd(big_matrix, full_matrices=False)
print(f"Full SVD time: {time.time() - start:.3f}s")

# Randomized SVD timing (শুধু top 50 components)
start = time.time()
U_rand, s_rand, Vt_rand = randomized_svd(big_matrix, n_components=50, random_state=42)
print(f"Randomized SVD time: {time.time() - start:.3f}s")

# Accuracy compare করুন
print(f"\nTop singular values comparison:")
print(f"Full SVD: {s_full[:5].round(2)}")
print(f"Randomized SVD: {s_rand[:5].round(2)}")

# TruncatedSVD (sklearn) - sparse matrices-এর জন্য আদর্শ
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

# Sparse matrix তৈরি করুন
sparse_X = sparse_random(10000, 5000, density=0.01, random_state=42)
print(f"\nSparse matrix shape: {sparse_X.shape}")
print(f"Non-zero elements: {sparse_X.nnz} ({sparse_X.nnz/sparse_X.size*100:.2f}%)")

tsvd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = tsvd.fit_transform(sparse_X)
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {tsvd.explained_variance_ratio_.sum()*100:.1f}%")</code></pre>

    <h3>৮. Latent Semantic Analysis (LSA): Text-এ SVD</h3>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Sample documents
documents = [
    "machine learning algorithms for data analysis",
    "deep learning neural networks classification",
    "python programming data science tutorial",
    "natural language processing text mining",
    "computer vision image recognition deep learning",
    "statistical machine learning regression analysis",
    "neural network training optimization gradient",
    "text classification sentiment analysis NLP",
]

# TF-IDF matrix তৈরি করুন
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(documents)
print(f"TF-IDF shape: {X_tfidf.shape}")

# LSA: SVD দিয়ে topics বের করুন
n_topics = 3
lsa = TruncatedSVD(n_components=n_topics, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)

# প্রতিটি topic-এ top words
feature_names = vectorizer.get_feature_names_out()
for i, component in enumerate(lsa.components_):
    top_words_idx = component.argsort()[-5:][::-1]
    top_words = [feature_names[j] for j in top_words_idx]
    print(f"Topic {i+1}: {', '.join(top_words)}")</code></pre>

    <h3>৯. SVD-এর সারসংক্ষেপ</h3>
    <p>
      SVD হলো লিনিয়ার অ্যালজেব্রার "Swiss Army Knife"। এর মূল বৈশিষ্ট্যগুলো:
    </p>
    <ul>
      <li><strong>Universal:</strong> যেকোনো m×n matrix-এ কাজ করে (square হতে হবে না)।</li>
      <li><strong>Optimal:</strong> Low-rank approximation-এর জন্য theoretically best।</li>
      <li><strong>Stable:</strong> Numerically stable algorithm।</li>
      <li><strong>Versatile:</strong> Compression, recommendation, NLP, PCA সব কিছুতে।</li>
    </ul>
    <p>
      পরবর্তী পর্বে আমরা দেখব t-SNE — যা non-linear structure ধরতে পারে এবং উচ্চমাত্রার ডেটার সুন্দর visualization তৈরি করতে পারে।
    </p>
  `
};
