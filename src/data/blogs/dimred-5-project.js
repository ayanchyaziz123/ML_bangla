export const dimred_5_project = {
  slug: 'dimred-5-project',
  title: 'ডাইমেনশনালিটি রিডাকশন: তুলনামূলক প্রজেক্ট',
  description: 'MNIST ও Iris ডেটাসেটে PCA, t-SNE এবং UMAP-এর পূর্ণ তুলনামূলক বিশ্লেষণ, visualization, pipeline integration এবং সঠিক পদ্ধতি বেছে নেওয়ার গাইড।',
  date: 'মে ২০২৫',
  category: 'ডাইমেনশনালিটি রিডাকশন',
  readTime: 16,
  content: `
    <h3>১. প্রজেক্ট পরিচিতি</h3>
    <p>
      এই পর্বে আমরা একটি comprehensive project করব যেখানে PCA, t-SNE এবং UMAP তিনটি পদ্ধতি একসাথে apply করব এবং তুলনা করব। আমরা দুটো ডেটাসেট ব্যবহার করব:
    </p>
    <ul>
      <li><strong>Iris Dataset:</strong> ছোট, ব্যাখ্যাযোগ্য (৪ ফিচার, ৩ class)।</li>
      <li><strong>MNIST Digits:</strong> বড়, উচ্চমাত্রার (৬৪ ফিচার, ১০ class)।</li>
    </ul>
    <p>আমরা দেখব:</p>
    <ul>
      <li>কোন পদ্ধতি কোথায় ভালো কাজ করে।</li>
      <li>গতি ও গুণমানের trade-off।</li>
      <li>Downstream classification-এ ডাইমেনশন রিডাকশনের প্রভাব।</li>
      <li>ML pipeline-এ integration।</li>
    </ul>

    <h3>২. Setup এবং Data Preparation</h3>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not installed. Run: pip install umap-learn")
    UMAP_AVAILABLE = False

# ============================================
# Dataset 1: Iris
# ============================================
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
iris_names = iris.target_names
iris_features = iris.feature_names

print("=" * 50)
print("IRIS DATASET")
print("=" * 50)
print(f"Shape: {X_iris.shape}")
print(f"Classes: {iris_names}")
print(f"Features: {iris_features}")
print(f"Class distribution: {np.bincount(y_iris)}")

# ============================================
# Dataset 2: MNIST Digits
# ============================================
digits = load_digits()
X_mnist = digits.data
y_mnist = digits.target

print("\n" + "=" * 50)
print("MNIST DIGITS DATASET")
print("=" * 50)
print(f"Shape: {X_mnist.shape}")
print(f"Classes: {np.unique(y_mnist)}")
print(f"Class distribution: {np.bincount(y_mnist)}")

# Standardize করুন
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

scaler_mnist = StandardScaler()
X_mnist_scaled = scaler_mnist.fit_transform(X_mnist)</code></pre>

    <h3>৩. Iris Dataset-এ তিনটি পদ্ধতি Apply করুন</h3>
    <pre><code>def run_all_methods(X, y, dataset_name, target_names=None):
    """PCA, t-SNE, UMAP একসাথে চালান এবং timing রেকর্ড করুন"""
    results = {}

    # PCA
    print(f"Running PCA on {dataset_name}...")
    start = time.time()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_time = time.time() - start
    evr = pca.explained_variance_ratio_.sum()
    results['PCA'] = {
        'X': X_pca, 'time': pca_time, 'evr': evr, 'model': pca
    }
    print(f"  PCA done in {pca_time:.3f}s | EVR: {evr*100:.1f}%")

    # t-SNE
    print(f"Running t-SNE on {dataset_name}...")
    start = time.time()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    tsne_time = time.time() - start
    results['t-SNE'] = {'X': X_tsne, 'time': tsne_time, 'model': tsne}
    print(f"  t-SNE done in {tsne_time:.3f}s")

    # UMAP
    if UMAP_AVAILABLE:
        print(f"Running UMAP on {dataset_name}...")
        start = time.time()
        reducer = umap.UMAP(n_components=2, n_neighbors=15,
                            min_dist=0.1, random_state=42)
        X_umap_emb = reducer.fit_transform(X)
        umap_time = time.time() - start
        results['UMAP'] = {'X': X_umap_emb, 'time': umap_time, 'model': reducer}
        print(f"  UMAP done in {umap_time:.3f}s")

    return results

# Iris results
iris_results = run_all_methods(X_iris_scaled, y_iris, "Iris", iris_names)

# Visualization
fig, axes = plt.subplots(1, len(iris_results), figsize=(15, 5))
colors = ['red', 'green', 'blue']

for ax, (name, res) in zip(axes, iris_results.items()):
    X_emb = res['X']
    for cls_idx, (cls_name, color) in enumerate(zip(iris_names, colors)):
        mask = y_iris == cls_idx
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1],
                   c=color, label=cls_name, alpha=0.8, s=60)
    title = f"{name} ({res['time']:.3f}s)"
    if 'evr' in res:
        title += f"\nEVR: {res['evr']*100:.1f}%"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Iris Dataset: PCA vs t-SNE vs UMAP', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. MNIST Digits-এ তুলনা</h3>
    <pre><code># MNIST results
mnist_results = run_all_methods(X_mnist_scaled, y_mnist, "MNIST")

# Visualization
fig, axes = plt.subplots(1, len(mnist_results), figsize=(18, 6))

for ax, (name, res) in zip(axes, mnist_results.items()):
    X_emb = res['X']
    scatter = ax.scatter(X_emb[:, 0], X_emb[:, 1],
                         c=y_mnist, cmap='tab10', alpha=0.6, s=5)
    title = f"{name} ({res['time']:.1f}s)"
    if 'evr' in res:
        title += f"\nEVR: {res['evr']*100:.1f}%"
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.colorbar(scatter, ax=axes[-1], label='Digit')
plt.suptitle('MNIST Digits: PCA vs t-SNE vs UMAP', fontsize=14)
plt.tight_layout()
plt.show()

# Time comparison bar chart
methods = list(mnist_results.keys())
times = [mnist_results[m]['time'] for m in methods]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, times, color=['#3498db', '#e74c3c', '#2ecc71'])
plt.ylabel('Time (seconds)')
plt.title('Speed Comparison on MNIST (1797 samples, 64 features)')
for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{t:.2f}s', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. Downstream Classification-এ প্রভাব</h3>
    <pre><code>def evaluate_classification(X_orig, y, reduction_results, dataset_name):
    """Different embeddings-এ KNN classification accuracy compare করুন"""
    print(f"\n{'='*50}")
    print(f"Classification on {dataset_name}")
    print(f"{'='*50}")

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    # Original
    scores_orig = cross_val_score(knn, X_orig, y, cv=5, scoring='accuracy')
    print(f"{'Original':<15} Accuracy: {scores_orig.mean()*100:.2f}% (+/- {scores_orig.std()*100:.2f}%)")

    # Different dimensionality reduction
    for name, res in reduction_results.items():
        X_emb = res['X']
        scores = cross_val_score(knn, X_emb, y, cv=5, scoring='accuracy')
        print(f"{name:<15} Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

# PCA দিয়ে আরও মাত্রায় কমানো এবং accuracy দেখা
print("\nPCA: Accuracy vs Number of Components (MNIST)")
print("-" * 50)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
n_components_list = [2, 5, 10, 20, 30, 40, 50]

for n in n_components_list:
    pca_n = PCA(n_components=n)
    X_pca_n = pca_n.fit_transform(X_mnist_scaled)
    scores = cross_val_score(knn, X_pca_n, y_mnist, cv=5, scoring='accuracy')
    evr = pca_n.explained_variance_ratio_.sum()
    print(f"  n={n:3d} | EVR: {evr*100:.1f}% | Accuracy: {scores.mean()*100:.2f}%")

evaluate_classification(X_iris_scaled, y_iris, iris_results, "Iris")
evaluate_classification(X_mnist_scaled, y_mnist, mnist_results, "MNIST")</code></pre>

    <h3>৬. ML Pipeline Integration</h3>
    <pre><code># sklearn Pipeline-এ PCA integrate করা
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# PCA + SVM pipeline
pipe_pca_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
])

# Only SVM pipeline
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
])

# MNIST train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42, stratify=y_mnist
)

# Train করুন এবং compare করুন
models = {
    'SVM only': pipe_svm,
    'PCA(30) + SVM': pipe_pca_svm
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"  Train time: {train_time:.3f}s | Predict time: {pred_time*1000:.1f}ms")
    print(f"  Test Accuracy: {acc*100:.2f}%")

# PCA components select করার জন্য GridSearch
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC(kernel='rbf', gamma='scale'))
])

param_grid = {
    'pca__n_components': [10, 20, 30, 40],
    'svm__C': [1, 10]
}

grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy',
                            n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_*100:.2f}%")
print(f"Test accuracy: {grid_search.score(X_test, y_test)*100:.2f}%")</code></pre>

    <h3>৭. Qualitative Analysis: কোন Digit কোথায়?</h3>
    <pre><code># MNIST-এ প্রতিটি digit-এর centroid দেখুন
def plot_digit_centroids(X_emb, y, title):
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1],
                          c=y, cmap='tab10', alpha=0.3, s=5)

    # প্রতিটি digit-এর centroid খুঁজুন এবং label দিন
    for digit in range(10):
        mask = y == digit
        centroid = X_emb[mask].mean(axis=0)
        plt.annotate(str(digit), centroid,
                     fontsize=20, fontweight='bold',
                     ha='center', va='center',
                     color='black',
                     bbox=dict(boxstyle='round', facecolor='white',
                               alpha=0.8, edgecolor='gray'))

    plt.title(title, fontsize=14)
    plt.colorbar(scatter, label='Digit')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# PCA
plot_digit_centroids(mnist_results['PCA']['X'], y_mnist,
                      'PCA: MNIST Digit Centroids')

# t-SNE
plot_digit_centroids(mnist_results['t-SNE']['X'], y_mnist,
                      't-SNE: MNIST Digit Centroids')

if UMAP_AVAILABLE:
    plot_digit_centroids(mnist_results['UMAP']['X'], y_mnist,
                          'UMAP: MNIST Digit Centroids')</code></pre>

    <h3>৮. সঠিক পদ্ধতি বেছে নেওয়ার গাইড</h3>
    <p>
      এই project থেকে আমরা যা শিখলাম তার উপর ভিত্তি করে একটি সিদ্ধান্ত গাইড:
    </p>

    <h4>PCA বেছে নিন যখন:</h4>
    <ul>
      <li>Model training-এর আগে preprocessing করতে চান।</li>
      <li>ডেটা mainly linear structure আছে।</li>
      <li>Speed এবং reproducibility দরকার।</li>
      <li>Interpretability গুরুত্বপূর্ণ (explained variance ratio)।</li>
      <li>Out-of-sample transform দরকার (production deployment)।</li>
      <li>Noise reduction করতে চান।</li>
    </ul>

    <h4>t-SNE বেছে নিন যখন:</h4>
    <ul>
      <li>Visualization-এর জন্য (কখনো ML task-এ ব্যবহার করবেন না)।</li>
      <li>ছোট ডেটাসেট (৫,০০০-১০,০০০ পয়েন্ট)।</li>
      <li>Non-linear cluster structure explore করতে চান।</li>
      <li>খুব tight, distinct clusters দেখতে চান।</li>
    </ul>

    <h4>UMAP বেছে নিন যখন:</h4>
    <ul>
      <li>বড় ডেটাসেট (১০,০০০+ পয়েন্ট) visualize করতে চান।</li>
      <li>Out-of-sample transform দরকার কিন্তু non-linear structure আছে।</li>
      <li>General dimensionality reduction (2D-এর বেশি মাত্রায়)।</li>
      <li>Global এবং local structure দুটোই সংরক্ষণ করতে চান।</li>
      <li>Speed গুরুত্বপূর্ণ।</li>
    </ul>

    <h3>৯. Final Summary Table</h3>
    <pre><code># Summary DataFrame
summary_data = {
    'Method': ['PCA', 't-SNE', 'UMAP'],
    'Type': ['Linear', 'Non-linear', 'Non-linear'],
    'Speed': ['Fast', 'Slow', 'Medium-Fast'],
    'Global Structure': ['Excellent', 'Poor', 'Good'],
    'Local Structure': ['Good', 'Excellent', 'Excellent'],
    'Out-of-sample': ['Yes', 'No', 'Yes'],
    'Scalability': ['Excellent', 'Poor', 'Good'],
    'Best Use Case': ['Preprocessing/DR', 'Visualization', 'Visualization/DR']
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("\nProject Complete! Key Takeaways:")
print("1. PCA = fast, linear, great for preprocessing")
print("2. t-SNE = beautiful clusters, only for visualization")
print("3. UMAP = best of both worlds, modern choice")</code></pre>

    <h3>১০. সিরিজের সমাপ্তি</h3>
    <p>
      এই সিরিজে আমরা ডাইমেনশনালিটি রিডাকশনের সম্পূর্ণ পরিবার সম্পর্কে জেনেছি:
    </p>
    <ul>
      <li><strong>PCA:</strong> সর্বোচ্চ variance-এর দিক খোঁজে linear projection।</li>
      <li><strong>SVD:</strong> Matrix factorization, PCA-এর কার্যকর implementation।</li>
      <li><strong>t-SNE:</strong> Local structure সংরক্ষণ করে সুন্দর visualization।</li>
      <li><strong>UMAP:</strong> Topological approach, দ্রুত ও versatile।</li>
    </ul>
    <p>
      বাস্তব ML project-এ সাধারণত এইভাবে শুরু করুন: প্রথমে PCA দিয়ে দ্রুত explore করুন, তারপর UMAP দিয়ে সুন্দর visualization তৈরি করুন, এবং ML pipeline-এ PCA বা UMAP ব্যবহার করুন।
    </p>
    <p>
      মনে রাখবেন: কোনো পদ্ধতিই সব ক্ষেত্রে সেরা নয়। ডেটার প্রকৃতি, dataset size, এবং আপনার লক্ষ্য অনুযায়ী সঠিক পদ্ধতি বেছে নিন।
    </p>
  `
};
