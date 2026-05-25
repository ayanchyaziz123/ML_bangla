export const dimred_3_tsne = {
  slug: 'dimred-3-tsne',
  title: 't-SNE: উচ্চমাত্রার ডেটা ভিজুয়ালাইজেশন',
  description: 't-SNE কীভাবে local structure সংরক্ষণ করে, perplexity parameter, ধাপে ধাপে অ্যালগরিদম, PCA-র সাথে তুলনা এবং Python দিয়ে সুন্দর visualization।',
  date: 'মে ২০২৫',
  category: 'ডাইমেনশনালিটি রিডাকশন',
  readTime: 15,
  content: `
    <h3>১. t-SNE কী এবং কেন এটি আলাদা?</h3>
    <p>
      t-SNE (t-distributed Stochastic Neighbor Embedding) ২০০৮ সালে Laurens van der Maaten এবং Geoffrey Hinton প্রস্তাব করেন। এটি উচ্চমাত্রার ডেটাকে ২ বা ৩ মাত্রায় নামিয়ে visualize করার জন্য সবচেয়ে জনপ্রিয় পদ্ধতি।
    </p>
    <p>
      PCA-র সাথে t-SNE-এর মূল পার্থক্য হলো:
    </p>
    <ul>
      <li><strong>PCA:</strong> Global structure সংরক্ষণ করে (variance maximize করে), linear।</li>
      <li><strong>t-SNE:</strong> Local structure সংরক্ষণ করে (কাছের পয়েন্টগুলো কাছে রাখে), non-linear।</li>
    </ul>
    <p>
      ফলে t-SNE দিয়ে ডেটার clusters অনেক স্পষ্টভাবে দেখা যায়, যেখানে PCA অনেক সময় ব্যর্থ হয়।
    </p>

    <h3>২. মূল অন্তর্দৃষ্টি: পড়শি সংরক্ষণ করা</h3>
    <p>
      t-SNE-এর মূল ধারণাটি বোঝা যাক:
    </p>
    <p>
      কল্পনা করুন আপনার কাছে ১০০০ মাত্রার ডেটা আছে। t-SNE বলে:
    </p>
    <blockquote>
      "উচ্চমাত্রায় যে দুটো পয়েন্ট কাছাকাছি ছিল, ২ মাত্রায়ও তারা কাছাকাছি থাকুক। যারা দূরে ছিল, তারা দূরেই থাকুক।"
    </blockquote>
    <p>
      এটি করার জন্য t-SNE probability distributions ব্যবহার করে। উচ্চমাত্রায় প্রতিটি পয়েন্টের কাছের পড়শির একটি probability distribution তৈরি করা হয়, এবং ২ মাত্রায়ও একই রকম distribution তৈরি করার চেষ্টা করা হয়।
    </p>

    <h3>৩. t-SNE অ্যালগরিদম: ধাপে ধাপে</h3>

    <h4>ধাপ ১: উচ্চমাত্রায় Similarity পরিমাপ</h4>
    <p>
      প্রতিটি জোড়া পয়েন্ট (i, j)-এর জন্য conditional probability p(j|i) হিসাব করা হয় — xᵢ-এর "neighborhood"-এ xⱼ থাকার সম্ভাবনা:
    </p>
    <p>
      p(j|i) = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ≠ᵢ exp(-||xᵢ - xₖ||² / 2σᵢ²)
    </p>
    <p>
      এটি মূলত Gaussian (normal) distribution। σᵢ হলো i-তম পয়েন্টের জন্য bandwidth যা perplexity থেকে নির্ধারিত হয়।
    </p>
    <p>Symmetric করার জন্য: pᵢⱼ = (p(j|i) + p(i|j)) / 2n</p>

    <h4>ধাপ ২: নিম্নমাত্রায় Similarity পরিমাপ</h4>
    <p>
      ২ মাত্রার embedding-এ (yᵢ, yⱼ)-এর জন্য t-distribution ব্যবহার করা হয় (এখানেই "t" আসে):
    </p>
    <p>
      qᵢⱼ = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖ≠ₗ (1 + ||yₖ - yₗ||²)⁻¹
    </p>
    <p>
      <strong>কেন t-distribution?</strong> Gaussian হলে দূরের পয়েন্টগুলো একে অপরের উপর "চাপা" পড়ে (crowding problem)। t-distribution-এর heavy tails এই সমস্যা সমাধান করে — দূরের পয়েন্টগুলো আরও ছড়িয়ে যায়।
    </p>

    <h4>ধাপ ৩: KL Divergence Minimize করা</h4>
    <p>
      লক্ষ্য হলো p এবং q distribution যতটা সম্ভব কাছাকাছি আনা। Cost function হলো KL Divergence:
    </p>
    <p>
      C = Σᵢ KL(Pᵢ || Qᵢ) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
    </p>
    <p>
      Gradient descent দিয়ে এই cost minimize করা হয়:
    </p>
    <p>
      ∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
    </p>

    <h3>৪. Perplexity: সবচেয়ে গুরুত্বপূর্ণ Hyperparameter</h3>
    <p>
      Perplexity মূলত "কতজন effective nearest neighbor বিবেচনা করা হবে" তা নির্ধারণ করে। এটি σᵢ নিয়ন্ত্রণ করে।
    </p>
    <p>
      Perplexity = 2^H(Pᵢ) যেখানে H হলো Shannon entropy।
    </p>
    <p>
      সাধারণ guideline:
    </p>
    <ul>
      <li><strong>Low perplexity (5-10):</strong> Very local structure, ছোট tight clusters।</li>
      <li><strong>Medium perplexity (30-50):</strong> Balance, বেশিরভাগ ক্ষেত্রে ভালো।</li>
      <li><strong>High perplexity (100+):</strong> More global structure, blurry clusters।</li>
    </ul>
    <p>
      <strong>Important:</strong> t-SNE-তে clusters-এর relative আকার বা clusters-এর মধ্যে দূরত্ব সরাসরি ব্যাখ্যাযোগ্য নয়।
    </p>

    <h3>৫. Python দিয়ে t-SNE</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# MNIST digits লোড করুন
digits = load_digits()
X = digits.data
y = digits.target
print(f"Data shape: {X.shape}")  # (1797, 64)
print(f"Classes: {np.unique(y)}")  # 0-9

# Standardize করুন
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE apply করুন
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    learning_rate='auto',
    init='pca',       # PCA দিয়ে initialize করুন (recommended)
    random_state=42,
    verbose=1
)
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE output shape: {X_tsne.shape}")  # (1797, 2)

# ভিজুয়ালাইজ করুন
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE Visualization of MNIST Digits (perplexity=30)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. Perplexity-র প্রভাব তুলনা করুন</h3>
    <pre><code># ভিন্ন perplexity তুলনা
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, perp in zip(axes.flatten(), perplexities):
    tsne = TSNE(n_components=2, perplexity=perp,
                n_iter=1000, random_state=42, init='pca',
                learning_rate='auto')
    X_emb = tsne.fit_transform(X_scaled[:500])  # দ্রুত করতে 500 sample

    scatter = ax.scatter(X_emb[:, 0], X_emb[:, 1],
                         c=y[:500], cmap='tab10', alpha=0.7, s=15)
    ax.set_title(f'Perplexity = {perp}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

plt.suptitle('Effect of Perplexity on t-SNE', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৭. t-SNE vs PCA তুলনা</h3>
    <pre><code># PCA এবং t-SNE পাশাপাশি দেখুন
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                c=y, cmap='tab10', alpha=0.6, s=15)
axes[0].set_title(f'PCA (EVR: {sum(pca.explained_variance_ratio_)*100:.1f}%)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            n_iter=1000, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                           c=y, cmap='tab10', alpha=0.6, s=15)
axes[1].set_title('t-SNE (perplexity=30)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.colorbar(scatter, ax=axes[1], label='Digit')
plt.suptitle('PCA vs t-SNE on MNIST Digits', fontsize=14)
plt.tight_layout()
plt.show()

# KNN accuracy দিয়ে quantitative comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

for name, X_red in [('PCA', X_pca), ('t-SNE', X_tsne), ('Original', X_scaled)]:
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_red, y, cv=5)
    print(f"{name}: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")</code></pre>

    <h3>৮. t-SNE-এর সঠিক ব্যাখ্যা</h3>
    <p>
      t-SNE ব্যবহার করার সময় কিছু গুরুত্বপূর্ণ বিষয় মনে রাখতে হবে:
    </p>
    <ul>
      <li><strong>Cluster আকার meaningless:</strong> বড় cluster মানে বেশি variance নয়।</li>
      <li><strong>Cluster দূরত্ব meaningless:</strong> দুই cluster কাছে থাকা মানে তারা similar নয়।</li>
      <li><strong>Random results:</strong> Different random seed দিলে ভিন্ন layout আসতে পারে।</li>
      <li><strong>Topology preserved:</strong> কিছু cluster হয়তো পাশাপাশি থাকে কারণ তারা উচ্চমাত্রায় পাশাপাশি ছিল।</li>
    </ul>

    <h3>৯. কখন t-SNE ব্যবহার করবেন?</h3>
    <p><strong>t-SNE উপযুক্ত যখন:</strong></p>
    <ul>
      <li>Visualization-এর জন্য (শুধু ২-৩ মাত্রায়)।</li>
      <li>Cluster structure explore করতে চাইলে।</li>
      <li>ডেটাতে non-linear patterns আছে।</li>
      <li>পরিমাণমতো ডেটা (১০,০০০ এর কম পয়েন্ট)।</li>
    </ul>
    <p><strong>t-SNE উপযুক্ত নয় যখন:</strong></p>
    <ul>
      <li>নতুন ডেটার জন্য transform করতে চাইলে (out-of-sample extension নেই)।</li>
      <li>বড় ডেটাসেট (১ লক্ষের বেশি পয়েন্ট — এর জন্য UMAP ভালো)।</li>
      <li>Downstream ML task-এর জন্য (পরবর্তীতে model train করতে)।</li>
      <li>Global structure বোঝার জন্য।</li>
    </ul>
    <p>
      পরবর্তী পর্বে আমরা দেখব UMAP — যা t-SNE-এর অনেক সীমাবদ্ধতা কাটিয়ে উঠেছে এবং আরও দ্রুত কাজ করে।
    </p>
  `
};
