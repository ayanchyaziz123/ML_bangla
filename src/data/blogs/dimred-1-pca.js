export const dimred_1_pca = {
  slug: 'dimred-1-pca',
  title: 'PCA: ডাইমেনশন কমানোর রাজা',
  description: 'ডাইমেনশনালিটি রিডাকশন কী, Curse of Dimensionality, PCA-র অন্তর্দৃষ্টি, কোভেরিয়েন্স ম্যাট্রিক্স, eigenvectors/eigenvalues এবং Python দিয়ে সম্পূর্ণ বাস্তব উদাহরণ।',
  date: 'মে ২০২৫',
  category: 'ডাইমেনশনালিটি রিডাকশন',
  readTime: 14,
  content: `
    <h3>১. ডাইমেনশনালিটি রিডাকশন কী?</h3>
    <p>
      মেশিন লার্নিং-এ আমরা প্রায়ই এমন ডেটাসেটের মুখোমুখি হই যেখানে শত বা হাজার হাজার ফিচার (column) থাকে। উদাহরণ হিসেবে ধরুন একটি ছবি — ১০০×১০০ পিক্সেলের একটি ছবিতে ১০,০০০ ফিচার আছে। এত বেশি ফিচার নিয়ে কাজ করা অনেক সমস্যা তৈরি করে।
    </p>
    <p>
      <strong>ডাইমেনশনালিটি রিডাকশন</strong> হলো সেই প্রক্রিয়া যেখানে আমরা ডেটার গুরুত্বপূর্ণ তথ্য যতটা সম্ভব ধরে রেখে ফিচারের সংখ্যা কমিয়ে আনি। এটি দুইভাবে করা যায়:
    </p>
    <ul>
      <li><strong>Feature Selection:</strong> বিদ্যমান ফিচারগুলো থেকে সেরাগুলো বেছে নেওয়া।</li>
      <li><strong>Feature Extraction:</strong> নতুন, কম সংখ্যক ফিচার তৈরি করা যা মূল ডেটার সারমর্ম ধারণ করে।</li>
    </ul>
    <p>PCA (Principal Component Analysis) হলো Feature Extraction-এর সবচেয়ে বিখ্যাত পদ্ধতি।</p>

    <h3>২. Curse of Dimensionality</h3>
    <p>
      Richard Bellman ১৯৬১ সালে "Curse of Dimensionality" বা মাত্রার অভিশাপ ধারণাটি প্রবর্তন করেন। এই অভিশাপ বোঝার জন্য একটি সহজ উদাহরণ দেখি:
    </p>
    <p>
      কল্পনা করুন একটি ১×১ বর্গে কিছু বিন্দু আছে। ২ মাত্রায় ১০০ বিন্দু দিয়ে সমান কভারেজ পাওয়া যায়। কিন্তু ১০ মাত্রায় একই কভারেজের জন্য দরকার ১০০¹⁰ = ১০²⁰ বিন্দু। এটি স্পষ্টতই অসম্ভব।
    </p>
    <p>উচ্চমাত্রায় কিছু অদ্ভুত ঘটনা ঘটে:</p>
    <ul>
      <li><strong>Sparsity:</strong> ডেটা পয়েন্টগুলো একে অপর থেকে অনেক দূরে সরে যায়।</li>
      <li><strong>Distance concentration:</strong> সব পয়েন্টের মধ্যে দূরত্ব প্রায় সমান হয়ে যায়, ফলে nearest neighbor অর্থহীন হয়।</li>
      <li><strong>Overfitting:</strong> মডেল noise শিখে ফেলে কারণ ট্রেনিং ডেটা পর্যাপ্ত নয়।</li>
      <li><strong>Computational cost:</strong> গণনা অনেক বেশি সময় নেয়।</li>
    </ul>
    <p>
      উদাহরণ: একটি ছবি classification task-এ যদি প্রতিটি পিক্সেল একটি ফিচার হয়, তাহলে ২৮×২৮ MNIST ছবিতে ৭৮৪ ফিচার আছে। PCA দিয়ে এটি ৫০ বা ১০০ ফিচারে নামিয়ে আনলেও ৯৫% এর বেশি accuracy ধরে রাখা যায়।
    </p>

    <h3>৩. PCA-র অন্তর্দৃষ্টি: সর্বোচ্চ ভেরিয়েন্সের দিক</h3>
    <p>
      PCA-র মূল ধারণা অত্যন্ত সুন্দর। ধরুন আপনার ২ মাত্রার ডেটা আছে (x, y) এবং আপনি এটিকে ১ মাত্রায় প্রজেক্ট করতে চান। কোন দিকে প্রজেক্ট করলে সবচেয়ে বেশি তথ্য ধরে রাখা যাবে?
    </p>
    <p>
      উত্তর হলো: <strong>যে দিকে ডেটার variance সবচেয়ে বেশি, সেই দিকে।</strong>
    </p>
    <p>
      যদি সব ডেটা একটি সরলরেখায় থাকে, তাহলে সেই রেখার দিকে প্রজেক্ট করলে কোনো তথ্য হারাবে না। কিন্তু যদি সেই রেখার লম্ব দিকে প্রজেক্ট করি, সব ডেটা একটি বিন্দুতে মিলে যাবে — সব তথ্য হারিয়ে যাবে।
    </p>
    <p>PCA ঠিক এই কাজটিই করে — ক্রমানুসারে সর্বোচ্চ variance-এর দিকগুলো খুঁজে বের করে:</p>
    <ul>
      <li><strong>PC1 (1st Principal Component):</strong> সবচেয়ে বেশি variance-এর দিক।</li>
      <li><strong>PC2 (2nd Principal Component):</strong> PC1-এর লম্ব, দ্বিতীয় সর্বোচ্চ variance-এর দিক।</li>
      <li><strong>PC3, PC4, ...</strong> একইভাবে।</li>
    </ul>
    <p>
      গুরুত্বপূর্ণ বিষয়: প্রতিটি principal component পূর্ববর্তীগুলোর সাথে <strong>orthogonal (লম্ব)</strong>। এর মানে তারা পরস্পর থেকে স্বাধীন।
    </p>

    <h3>৪. গণিত: Covariance Matrix থেকে Eigenvectors</h3>
    <p>PCA-র গণিতটি চারটি ধাপে বোঝা যায়:</p>

    <h4>ধাপ ১: Data Standardization</h4>
    <p>
      প্রথমে ডেটা normalize করতে হয় (mean = 0, std = 1):
      z = (x - μ) / σ
    </p>
    <p>
      এটি জরুরি কারণ ভিন্ন scale-এর ফিচার (যেমন বয়স ১-১০০, বেতন ১০০০-১০০০০০) থাকলে বেশি মানের ফিচার PCA-কে প্রভাবিত করবে।
    </p>

    <h4>ধাপ ২: Covariance Matrix তৈরি</h4>
    <p>
      দুটি ফিচার x এবং y-এর covariance হলো:
      Cov(x, y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n-1)
    </p>
    <p>
      Covariance positive হলে x বাড়লে y-ও বাড়ে। Negative হলে উল্টো। Zero হলে কোনো linear সম্পর্ক নেই।
    </p>
    <p>
      n ফিচারের জন্য Covariance Matrix হয় n×n:
      C = XᵀX / (n-1)
    </p>
    <p>
      ৩ ফিচারের উদাহরণ (x₁, x₂, x₃):
    </p>
    <pre><code>C = | Cov(x₁,x₁)  Cov(x₁,x₂)  Cov(x₁,x₃) |
    | Cov(x₂,x₁)  Cov(x₂,x₂)  Cov(x₂,x₃) |
    | Cov(x₃,x₁)  Cov(x₃,x₂)  Cov(x₃,x₃) |</code></pre>
    <p>Diagonal elements হলো প্রতিটি ফিচারের variance।</p>

    <h4>ধাপ ৩: Eigendecomposition</h4>
    <p>
      Covariance matrix C-এর জন্য আমরা eigenvalues (λ) এবং eigenvectors (v) খুঁজি যেখানে:
      Cv = λv
    </p>
    <p>
      <strong>Eigenvector:</strong> একটি দিক (direction) যা matrix দ্বারা transform হলেও দিক পরিবর্তন হয় না, শুধু magnitude পরিবর্তন হয়।
    </p>
    <p>
      <strong>Eigenvalue:</strong> কতটুকু magnitude পরিবর্তন হয় — এটি সেই দিকে ডেটার variance পরিমাপ করে।
    </p>
    <p>
      সবচেয়ে বড় eigenvalue-এর eigenvector হলো PC1 (সর্বোচ্চ variance), দ্বিতীয় বড়টি PC2, ইত্যাদি।
    </p>

    <h4>ধাপ ৪: Projection</h4>
    <p>
      k টি principal component রাখলে নতুন ডেটা:
      Z = X × W
    </p>
    <p>যেখানে W হলো k টি eigenvector-এর matrix (n×k)। Z হলো reduced data (m×k)।</p>

    <h3>৫. Explained Variance Ratio</h3>
    <p>
      কতটুকু তথ্য ধরে রাখা হয়েছে সেটি <strong>Explained Variance Ratio</strong> দিয়ে পরিমাপ করা হয়:
    </p>
    <p>
      EVR(i) = λᵢ / Σλⱼ
    </p>
    <p>
      উদাহরণ: যদি eigenvalues হয় [8.5, 3.2, 1.1, 0.5, 0.2], তাহলে:
      মোট variance = 13.5
      PC1 ব্যাখ্যা করে: 8.5/13.5 = 62.9%
      PC1 + PC2 ব্যাখ্যা করে: (8.5+3.2)/13.5 = 86.7%
    </p>
    <p>
      সাধারণত ৯৫% বা ৯৯% explained variance ধরে রাখার জন্য কতটি PC দরকার সেটি নির্ধারণ করা হয়। <strong>Scree Plot</strong> এই সিদ্ধান্ত নিতে সাহায্য করে।
    </p>

    <h3>৬. Python দিয়ে PCA</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ডেটা লোড করুন
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(f"Original shape: {X.shape}")  # (150, 4)

# ধাপ ১: Standardize করুন
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ধাপ ২: PCA apply করুন
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance দেখুন
evr = pca.explained_variance_ratio_
print("Explained Variance Ratio:", evr)
print("Cumulative EVR:", np.cumsum(evr))

# ধাপ ৩: Scree Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(evr)+1), evr * 100)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot')
plt.xticks(range(1, len(evr)+1))

plt.subplot(1, 2, 2)
plt.plot(range(1, len(evr)+1), np.cumsum(evr) * 100, 'bo-')
plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# ধাপ ৪: 2D-তে reduce করুন (visualization)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"Reduced shape: {X_2d.shape}")  # (150, 2)
print(f"Variance retained: {sum(pca_2d.explained_variance_ratio_)*100:.1f}%")

# ভিজুয়ালাইজ করুন
colors = ['red', 'green', 'blue']
labels = iris.target_names

plt.figure(figsize=(8, 6))
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=label, alpha=0.7, s=60)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Iris Dataset: PCA 2D Projection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ধাপ ৫: Loading (কোন original feature কোন PC-তে কতটা contribute করে)
loadings = pca_2d.components_
print("\nPC Loadings:")
for i, pc in enumerate(loadings):
    print(f"PC{i+1}: {dict(zip(feature_names, pc.round(3)))}")</code></pre>

    <h3>৭. PCA দিয়ে Noise Reduction</h3>
    <pre><code># Noisy MNIST digits reconstruct করা
from sklearn.datasets import load_digits

digits = load_digits()
X_digits = digits.data  # (1797, 64)

# Noise যোগ করুন
np.random.seed(42)
X_noisy = X_digits + np.random.normal(0, 2, X_digits.shape)

# PCA দিয়ে 40 components রাখুন
pca_denoise = PCA(n_components=40)
X_reduced = pca_denoise.fit_transform(X_noisy)
X_reconstructed = pca_denoise.inverse_transform(X_reduced)

print(f"Variance retained: {sum(pca_denoise.explained_variance_ratio_)*100:.1f}%")

# আগে ও পরে দেখুন
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(X_noisy[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    axes[2, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('Original', rotation=90, labelpad=30)
axes[1, 0].set_ylabel('Noisy', rotation=90, labelpad=30)
axes[2, 0].set_ylabel('Denoised', rotation=90, labelpad=30)
plt.suptitle('PCA Denoising')
plt.tight_layout()
plt.show()</code></pre>

    <h3>৮. Incremental PCA (বড় ডেটাসেটের জন্য)</h3>
    <pre><code>from sklearn.decomposition import IncrementalPCA

# বড় ডেটা batch-এ process করুন
batch_size = 100
ipca = IncrementalPCA(n_components=2, batch_size=batch_size)

# mini-batch-এ fit করুন
for batch_start in range(0, len(X_scaled), batch_size):
    batch = X_scaled[batch_start:batch_start + batch_size]
    ipca.partial_fit(batch)

# Transform করুন
X_ipca = ipca.transform(X_scaled)
print(f"IncrementalPCA shape: {X_ipca.shape}")
print(f"Variance retained: {sum(ipca.explained_variance_ratio_)*100:.1f}%")</code></pre>

    <h3>৯. PCA-র সীমাবদ্ধতা</h3>
    <p>PCA অনেক শক্তিশালী হলেও কিছু সীমাবদ্ধতা আছে:</p>
    <ul>
      <li><strong>Linear only:</strong> PCA শুধু linear relationships ধরতে পারে। Non-linear structure (যেমন spiral, circle) ধরতে পারে না।</li>
      <li><strong>Interpretability:</strong> নতুন components মূল ফিচারের combination, তাই সরাসরি ব্যাখ্যা করা কঠিন।</li>
      <li><strong>Scaling sensitivity:</strong> Standardization না করলে ভুল ফলাফল আসে।</li>
      <li><strong>Outlier sensitive:</strong> Outlier থাকলে variance-based PCA প্রভাবিত হয়।</li>
    </ul>
    <p>
      Non-linear ডেটার জন্য Kernel PCA, t-SNE, বা UMAP বেশি উপযুক্ত, যেগুলো পরবর্তী পর্বে আলোচনা করা হবে।
    </p>

    <h3>১০. কখন PCA ব্যবহার করবেন?</h3>
    <ul>
      <li>High-dimensional ডেটা (৫০+ ফিচার) নিয়ে কাজ করার সময়।</li>
      <li>ফিচারগুলোর মধ্যে multicollinearity থাকলে।</li>
      <li>Visualization-এর জন্য ২-৩ মাত্রায় নামাতে চাইলে।</li>
      <li>Model training আগে preprocessing হিসেবে।</li>
      <li>Image/text data compression-এ।</li>
    </ul>
    <p>
      মনে রাখবেন: PCA supervised learning নয়, তাই এটি class labels সম্পর্কে কিছু জানে না। যদি class separation গুরুত্বপূর্ণ হয়, তাহলে LDA (Linear Discriminant Analysis) বিবেচনা করুন।
    </p>
  `
};
