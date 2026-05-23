export const knn_4_distance = {
  title: "KNN Distance Metrics: কোনটি কখন ব্যবহার করবো?",
  description: "Euclidean, Manhattan, Minkowski, Chebyshev, Hamming এবং Cosine দূরত্ব মেট্রিক্সের বিস্তারিত বিশ্লেষণ এবং sklearn-এ প্রয়োগ।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 9,
  slug: "knn-distance-metrics",
  content: `
    <h3>১. কেন Distance Metric গুরুত্বপূর্ণ?</h3>
    <p>
      KNN-এর performance সম্পূর্ণ নির্ভর করে আমরা কিভাবে "দূরত্ব" পরিমাপ করি তার উপর।
      ভুল metric ব্যবহার করলে model সম্পূর্ণ ভুল প্রতিবেশী খুঁজে পাবে এবং prediction
      নির্ভরযোগ্য হবে না।
    </p>
    <p>
      Data-র প্রকৃতির উপর ভিত্তি করে metric নির্বাচন করতে হয়:
      continuous numerical data, categorical data, text data, বা binary data
      প্রত্যেকের জন্য আলাদা metric উপযুক্ত।
    </p>

    <h3>২. প্রধান Distance Metrics</h3>
    <p><strong>Euclidean Distance:</strong></p>
    <p>
      সরলরেখায় দুটি বিন্দুর মধ্যে দূরত্ব।
      Formula: d = sqrt(Σ(xi - yi)²)।
      এটি সবচেয়ে সাধারণ এবং continuous numerical data-র জন্য উপযুক্ত।
      তবে outlier-এর প্রতি sensitive কারণ square করা হয়।
    </p>
    <p><strong>Manhattan Distance:</strong></p>
    <p>
      Grid-এ চলার মতো দূরত্ব (শহরের রাস্তার মতো)।
      Formula: d = Σ|xi - yi|।
      Outlier-এ robust কারণ square করা হয় না।
      High-dimensional data-তে Euclidean-এর চেয়ে ভালো কাজ করে।
    </p>
    <p><strong>Minkowski Distance:</strong></p>
    <p>
      Euclidean ও Manhattan-এর সাধারণ রূপ।
      Formula: d = (Σ|xi - yi|^p)^(1/p)।
      p=1 → Manhattan, p=2 → Euclidean, p→∞ → Chebyshev।
      sklearn-এ p parameter দিয়ে control করা যায়।
    </p>
    <p><strong>Chebyshev Distance:</strong></p>
    <p>
      যেকোনো একটি dimension-এ maximum পার্থক্য।
      Formula: d = max(|xi - yi|)।
      Chess-এ king-এর movement-এর মতো।
      সব dimension-এ uniform movement প্রয়োজন হলে ব্যবহার।
    </p>
    <p><strong>Hamming Distance:</strong></p>
    <p>
      Categorical বা binary data-র জন্য।
      Formula: positions-এর fraction যেখানে দুটি string ভিন্ন।
      Text comparison, DNA sequence matching-এ ব্যবহার।
    </p>
    <p><strong>Cosine Distance:</strong></p>
    <p>
      দুটি vector-এর মধ্যে কোণ পরিমাপ করে।
      Cosine similarity = (A·B) / (||A|| × ||B||)।
      Cosine distance = 1 - Cosine similarity।
      Text/NLP, recommendation systems-এ সবচেয়ে জনপ্রিয়।
      Magnitude নয়, direction গুরুত্বপূর্ণ হলে ব্যবহার।
    </p>

    <h3>৩. Python দিয়ে সব Metric হিসাব</h3>
    <pre><code>import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev
from scipy.spatial.distance import hamming, cosine
import pandas as pd

# Numerical data
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 4, 1, 3, 5])

print("Numerical Distance Metrics:")
print(f"  Euclidean:  {euclidean(x1, x2):.4f}")
print(f"  Manhattan:  {cityblock(x1, x2):.4f}")
print(f"  Minkowski (p=3): {minkowski(x1, x2, p=3):.4f}")
print(f"  Chebyshev:  {chebyshev(x1, x2):.4f}")
print(f"  Cosine:     {cosine(x1, x2):.4f}")

# Categorical/Binary data (Hamming)
str1 = np.array([1, 0, 1, 1, 0, 1])  # binary features
str2 = np.array([1, 1, 1, 0, 0, 1])
print(f"\nHamming Distance (binary): {hamming(str1, str2):.4f}")
print(f"  (ভিন্ন position সংখ্যা: {np.sum(str1 != str2)}/{len(str1)})")

# Text similarity (Cosine)
doc1 = np.array([2, 1, 0, 3, 1])  # word frequencies
doc2 = np.array([1, 0, 1, 2, 2])
cos_sim = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
print(f"\nCosine Similarity (text): {cos_sim:.4f}")
print(f"Cosine Distance: {1-cos_sim:.4f}")

# Minkowski-এ p-এর প্রভাব
print("\nMinkowski-এ p-এর প্রভাব:")
for p in [1, 1.5, 2, 3, 5, 10]:
    dist = minkowski(x1, x2, p=p)
    label = f"p={p}" + (" (Manhattan)" if p==1 else " (Euclidean)" if p==2 else "")
    print(f"  {label}: {dist:.4f}")</code></pre>

    <h3>৪. Sklearn-এ Different Metrics তুলনা</h3>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Wine dataset (multi-class, 13 features)
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# বিভিন্ন metrics test করা
metrics_config = [
    ('euclidean', {}),
    ('manhattan', {}),
    ('minkowski', {'p': 3}),
    ('chebyshev', {}),
    ('cosine', {}),
]

results = []
for metric, kwargs in metrics_config:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric, **kwargs)
    cv_scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring='accuracy')

    # Test accuracy
    knn.fit(X_train_s, y_train)
    test_acc = knn.score(X_test_s, y_test)

    results.append({
        'Metric': metric,
        'CV Mean': round(cv_scores.mean(), 4),
        'CV Std': round(cv_scores.std(), 4),
        'Test Accuracy': round(test_acc, 4)
    })

df_results = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print("Wine Dataset-এ Distance Metrics তুলনা (K=5):")
print(df_results.to_string(index=False))

# Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = df_results['Metric'].values
test_accs = df_results['Test Accuracy'].values
cv_means = df_results['CV Mean'].values

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, cv_means, width, label='CV Accuracy',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
               color='coral', alpha=0.8)

ax.set_xlabel('Distance Metric')
ax.set_ylabel('Accuracy')
ax.set_title('Distance Metrics Comparison (Wine Dataset)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, rotation=15)
ax.legend()
ax.set_ylim(0.8, 1.0)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. Text Data-তে Cosine Distance</h3>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# Sample text classification
texts = [
    "machine learning python algorithm",
    "deep learning neural network",
    "python programming software",
    "neural network deep learning model",
    "algorithm data structure computer",
    "machine learning data science",
]
labels = [0, 1, 0, 1, 0, 0]  # 0=programming, 1=deep learning

# TF-IDF + KNN with Cosine distance
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('knn', KNeighborsClassifier(n_neighbors=3, metric='cosine'))
])

pipeline.fit(texts, labels)

# Test prediction
test_texts = [
    "python deep learning algorithm",
    "neural network machine",
]
predictions = pipeline.predict(test_texts)
for text, pred in zip(test_texts, predictions):
    label = "Deep Learning" if pred == 1 else "Programming"
    print(f"Text: '{text}' -> Class: {label}")</code></pre>

    <h3>৬. Distance Metrics তুলনা সারণী</h3>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Formula</th>
          <th>সেরা ব্যবহার</th>
          <th>Sklearn Parameter</th>
          <th>সীমাবদ্ধতা</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Euclidean</td>
          <td>sqrt(Σ(xi-yi)²)</td>
          <td>General continuous data</td>
          <td>metric='euclidean'</td>
          <td>Outlier-sensitive, scale-dependent</td>
        </tr>
        <tr>
          <td>Manhattan</td>
          <td>Σ|xi-yi|</td>
          <td>High-dim, outliers আছে</td>
          <td>metric='manhattan'</td>
          <td>Diagonal direction ignore করে</td>
        </tr>
        <tr>
          <td>Minkowski</td>
          <td>(Σ|xi-yi|^p)^(1/p)</td>
          <td>General purpose, tunable</td>
          <td>metric='minkowski', p=?</td>
          <td>p tune করতে হয়</td>
        </tr>
        <tr>
          <td>Chebyshev</td>
          <td>max(|xi-yi|)</td>
          <td>Chess, uniform movement</td>
          <td>metric='chebyshev'</td>
          <td>Single dimension dominates</td>
        </tr>
        <tr>
          <td>Hamming</td>
          <td>fraction of diff positions</td>
          <td>Categorical, binary data</td>
          <td>metric='hamming'</td>
          <td>Magnitude নেয় না</td>
        </tr>
        <tr>
          <td>Cosine</td>
          <td>1 - (A·B)/(||A||·||B||)</td>
          <td>Text, NLP, sparse data</td>
          <td>metric='cosine'</td>
          <td>Magnitude ignore করে</td>
        </tr>
      </tbody>
    </table>
  `,
};
