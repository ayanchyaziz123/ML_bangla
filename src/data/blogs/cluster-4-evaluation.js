export const cluster_4_evaluation = {
  title: "ক্লাস্টারিং মূল্যায়ন: Silhouette, Davies-Bouldin ও আরও মেট্রিক্স",
  description:
    "Unsupervised clustering কীভাবে evaluate করবেন? Inertia, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score — সব মেট্রিক্স কোড সহ বিস্তারিত।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 9,
  slug: "clustering-evaluation",
  content: `
    <h3>১. Unsupervised Learning Evaluation-এর চ্যালেঞ্জ</h3>
    <p>
      Supervised learning-এ মূল্যায়ন সহজ — ground truth label আছে, accuracy বা F1 score হিসাব করো। কিন্তু clustering-এ কোনো "সঠিক উত্তর" নেই। তাহলে কীভাবে বুঝবেন কোন clustering ভালো?
    </p>
    <p>
      ভালো clustering-এর দুটি মূলনীতি: <br/>
      <strong>Cohesion (সংহতি):</strong> একই cluster-এর পয়েন্টগুলো পরস্পরের কাছাকাছি হওয়া উচিত। <br/>
      <strong>Separation (বিচ্ছেদ):</strong> ভিন্ন cluster-এর পয়েন্টগুলো পরস্পর থেকে দূরে হওয়া উচিত।
    </p>
    <p>
      এই দুটি নীতির ভিত্তিতে বিভিন্ন evaluation metric তৈরি হয়েছে। চলুন সেগুলো দেখি।
    </p>

    <h3>২. Inertia / WCSS: সবচেয়ে সহজ মেট্রিক</h3>
    <p>
      <strong>Inertia</strong> বা WCSS (Within-Cluster Sum of Squares) হলো প্রতিটি পয়েন্ট থেকে তার cluster centroid-এর দূরত্বের বর্গের যোগফল। কম Inertia মানে cluster-গুলো আরও tight।
    </p>
    <p>
      <strong>সমস্যা:</strong> K বাড়লে Inertia সবসময় কমে — এমনকি K = n হলে Inertia = 0। তাই Inertia একা দিয়ে সর্বোত্তম K বলা যায় না। Elbow Method-এ কমার হার দেখতে হয়।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

# Synthetic dataset
X, y_true = make_blobs(n_samples=500, centers=4,
                       cluster_std=0.9, random_state=42)

# K=2 থেকে K=8 পর্যন্ত সব মেট্রিক্স
K_range = range(2, 9)
inertias = []
silhouette_scores = []
db_scores = []
ch_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++',
                n_init=10, random_state=42)
    labels = km.fit_predict(X)

    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))

print("K\tInertia\t\tSilhouette\tDavies-Bouldin\tCalinski-Harabasz")
for i, k in enumerate(K_range):
    print(f"{k}\t{inertias[i]:.1f}\t\t{silhouette_scores[i]:.3f}"
          f"\t\t{db_scores[i]:.3f}\t\t{ch_scores[i]:.1f}")</code></pre>

    <h3>৩. Silhouette Score: সেরা সর্বমুখী মেট্রিক</h3>
    <p>
      <strong>Silhouette Score</strong> প্রতিটি পয়েন্টের জন্য cohesion ও separation একসাথে মাপে। প্রতিটি পয়েন্ট i-এর জন্য:
    </p>
    <p>
      <strong>a(i)</strong> = i-এর সাথে তার নিজের cluster-এর অন্য পয়েন্টগুলোর গড় দূরত্ব (cohesion, কম হলে ভালো) <br/>
      <strong>b(i)</strong> = i-এর সাথে সবচেয়ে কাছের অন্য cluster-এর পয়েন্টগুলোর গড় দূরত্ব (separation, বেশি হলে ভালো) <br/>
      <strong>s(i)</strong> = (b(i) - a(i)) / max(a(i), b(i))
    </p>
    <p>
      Silhouette Score -1 থেকে +1 এর মধ্যে থাকে। +1 মানে নিখুঁত cluster, 0 মানে boundary-তে আছে, -1 মানে ভুল cluster-এ আছে।
    </p>
    <pre><code">from sklearn.metrics import silhouette_samples

# সেরা K খুঁজুন (Silhouette দিয়ে)
best_k = K_range[np.argmax(silhouette_scores)]
print(f"Silhouette Score সবচেয়ে বেশি K={best_k}-তে: "
      f"{max(silhouette_scores):.3f}")

# Silhouette plot
km_best = KMeans(n_clusters=best_k, init='k-means++',
                 n_init=10, random_state=42)
labels_best = km_best.fit_predict(X)
sample_silhouette = silhouette_samples(X, labels_best)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Silhouette analysis
y_lower = 10
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i in range(best_k):
    ith_silhouette = np.sort(sample_silhouette[labels_best == i])
    size_cluster_i = ith_silhouette.shape[0]
    y_upper = y_lower + size_cluster_i
    axes[0].fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_silhouette,
                          alpha=0.7, color=colors[i])
    axes[0].text(-0.05, y_lower + 0.5 * size_cluster_i, f'C{i+1}')
    y_lower = y_upper + 10

axes[0].axvline(x=max(silhouette_scores), color='red',
                linestyle='--', label=f'Avg={max(silhouette_scores):.3f}')
axes[0].set_title(f'Silhouette Analysis (K={best_k})', fontsize=12)
axes[0].set_xlabel('Silhouette Coefficient')
axes[0].set_ylabel('Cluster')
axes[0].legend()

# সব মেট্রিক্স plot
axes[1].plot(list(K_range), silhouette_scores, 'go-',
             label='Silhouette (higher=better)')
axes[1].set_title('Silhouette Score vs K', fontsize=12)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=150)
plt.show()</code></pre>

    <h3>৪. Davies-Bouldin ও Calinski-Harabasz Index</h3>
    <p>
      <strong>Davies-Bouldin Index (DBI):</strong> প্রতিটি cluster-এর ভেতরের ছড়িয়ে পড়া (within-cluster scatter) এবং cluster-গুলোর মধ্যে দূরত্বের ratio মাপে। কম DBI মানে ভালো clustering।
    </p>
    <p>
      <strong>Calinski-Harabasz Score (CH):</strong> Between-cluster variance এবং within-cluster variance-এর ratio। বেশি CH score মানে ভালো, compact এবং ভালোভাবে আলাদা clusters।
    </p>
    <pre><code"># সব মেট্রিক্স একসাথে visualization
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes[0, 0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_title('Inertia (WCSS) — Elbow Method', fontsize=12)
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(list(K_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=best_k, color='red', linestyle='--',
                   label=f'Best K={best_k}')
axes[0, 1].set_title('Silhouette Score (higher=better)', fontsize=12)
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(list(K_range), db_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_title('Davies-Bouldin Index (lower=better)', fontsize=12)
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('DBI')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(list(K_range), ch_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_title('Calinski-Harabasz Score (higher=better)', fontsize=12)
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('CH Score')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Clustering Evaluation Metrics Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('clustering_metrics.png', dpi=150)
plt.show()</code></pre>

    <h3>৫. মেট্রিক্স সারসংক্ষেপ ও কোনটি ব্যবহার করবেন?</h3>
    <table>
      <thead>
        <tr>
          <th>মেট্রিক্স</th>
          <th>Range</th>
          <th>ভালো মান</th>
          <th>সুবিধা</th>
          <th>অসুবিধা</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Inertia (WCSS)</td>
          <td>0 থেকে ∞</td>
          <td>কম হলে ভালো</td>
          <td>সহজ, দ্রুত</td>
          <td>K বাড়লে সবসময় কমে</td>
        </tr>
        <tr>
          <td>Silhouette Score</td>
          <td>-1 থেকে +1</td>
          <td>+1 কাছাকাছি</td>
          <td>Cohesion + Separation উভয়</td>
          <td>বড় ডেটাসেটে ধীর</td>
        </tr>
        <tr>
          <td>Davies-Bouldin</td>
          <td>0 থেকে ∞</td>
          <td>কম হলে ভালো</td>
          <td>দ্রুত গণনা</td>
          <td>শুধু centroid-ভিত্তিক</td>
        </tr>
        <tr>
          <td>Calinski-Harabasz</td>
          <td>0 থেকে ∞</td>
          <td>বেশি হলে ভালো</td>
          <td>দ্রুত, বড় ডেটাতেও কাজ করে</td>
          <td>Convex cluster ধরে নেয়</td>
        </tr>
      </tbody>
    </table>
    <p>
      বাস্তবে একটি মেট্রিক্সের উপর সম্পূর্ণ নির্ভর না করে একাধিক মেট্রিক্স একসাথে দেখুন। সাধারণত <strong>Silhouette Score</strong> সবচেয়ে নির্ভরযোগ্য। যদি ground truth label থাকে, তাহলে <strong>Adjusted Rand Index (ARI)</strong> বা <strong>Normalized Mutual Information (NMI)</strong> ব্যবহার করুন।
    </p>
  `,
};
