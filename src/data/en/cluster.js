export const clusterEn = [
  {
    title: "K-Means Clustering: A Step-by-Step Guide",
    description:
      "How the K-Means algorithm works, K-Means++ initialization, the Elbow Method for choosing K, and full sklearn examples with visualization.",
    date: "২৩ মে, ২০২৬",
    category: "Clustering",
    readTime: 11,
    slug: "clustering-kmeans",
    content: `
      <h3>1. What is Clustering and Why K-Means?</h3>
      <p>
        <strong>Clustering</strong> is an unsupervised machine learning technique where we group data into meaningful clusters without any labels. Imagine you have thousands of customer records — clustering can automatically segment these customers based on their behavior.
      </p>
      <p>
        <strong>K-Means</strong> is the most popular and straightforward clustering algorithm. It partitions data into K groups where each point is assigned to the nearest group center (centroid). The algorithm is iterative — it runs repeatedly until the clusters stabilize.
      </p>

      <h3>2. The K-Means Algorithm: Step by Step</h3>
      <p>K-Means works in four core steps:</p>
      <p>
        <strong>Step 1 — Initialization:</strong> Randomly pick K centroids. <br/>
        <strong>Step 2 — Assignment:</strong> Assign each data point to the nearest centroid's cluster. <br/>
        <strong>Step 3 — Update:</strong> Recompute each centroid as the mean of all points in its cluster. <br/>
        <strong>Step 4 — Repeat:</strong> Repeat steps 2 and 3 until centroids stop changing.
      </p>
      <p>
        Distance is typically measured using <strong>Euclidean distance</strong>:
        d(x, μ) = √Σ(xᵢ - μᵢ)²
      </p>

      <h3>3. K-Means++ Initialization</h3>
      <p>
        Standard K-Means picks centroids randomly, which can sometimes yield poor results. <strong>K-Means++</strong> solves this with smart initialization: the first centroid is random, and each subsequent centroid is chosen with probability proportional to its squared distance from the nearest existing centroid. This spreads centroids well and speeds up convergence.
      </p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create a synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.8, random_state=42)

# K-Means++ (default in sklearn: init='k-means++')
kmeans_pp = KMeans(n_clusters=4, init='k-means++',
                   max_iter=300, n_init=10, random_state=42)

# Random initialization
kmeans_rand = KMeans(n_clusters=4, init='random',
                     max_iter=300, n_init=10, random_state=42)

kmeans_pp.fit(X)
kmeans_rand.fit(X)

print(f"K-Means++ Inertia:     {kmeans_pp.inertia_:.2f}")
print(f"K-Means Random Inertia:{kmeans_rand.inertia_:.2f}")
print(f"K-Means++ Iterations:  {kmeans_pp.n_iter_}")
print(f"K-Means Random Iterations: {kmeans_rand.n_iter_}")</code></pre>

      <h3>4. The Elbow Method: How to Choose K</h3>
      <p>
        The biggest question in K-Means is: <em>what should K be?</em> The <strong>Elbow Method</strong> answers this by plotting <strong>WCSS (Within-Cluster Sum of Squares)</strong> — also called Inertia — against different values of K. As K increases, Inertia decreases. The point where the rate of decrease sharply slows down forms the "elbow" — that K is optimal.
      </p>
      <pre><code>inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++',
                max_iter=300, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K (Number of Clusters)', fontsize=13)
plt.ylabel('Inertia (WCSS)', fontsize=13)
plt.title('Elbow Method — Choosing Optimal K', fontsize=15)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()</code></pre>

      <h3>5. Full sklearn Example with Visualization</h3>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
plt.figure(figsize=(10, 7))

for i in range(4):
    mask = labels == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=colors[i], label=f'Cluster {i+1}',
                alpha=0.7, s=50, edgecolors='white', linewidths=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1],
            c='black', marker='X', s=200,
            label='Centroids', zorder=5)

plt.title('K-Means Clustering (K=4)', fontsize=15)
plt.xlabel('Feature 1 (scaled)', fontsize=12)
plt.ylabel('Feature 2 (scaled)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150)
plt.show()</code></pre>

      <h3>6. Limitations of K-Means</h3>
      <table>
        <thead>
          <tr>
            <th>Limitation</th>
            <th>Explanation</th>
            <th>Solution</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Must specify K upfront</td>
            <td>Wrong K yields poor results</td>
            <td>Elbow method, Silhouette score</td>
          </tr>
          <tr>
            <td>Only spherical clusters</td>
            <td>Cannot detect non-convex shapes</td>
            <td>Use DBSCAN or Spectral Clustering</td>
          </tr>
          <tr>
            <td>Sensitive to outliers</td>
            <td>Outliers pull centroids away</td>
            <td>Use K-Medoids or DBSCAN</td>
          </tr>
          <tr>
            <td>Local minimum risk</td>
            <td>Random init can give bad clusters</td>
            <td>Use K-Means++ and increase n_init</td>
          </tr>
        </tbody>
      </table>
      <p>
        K-Means works best when clusters are spherical, roughly equal-sized, and the data has few outliers. In the next post, we'll explore DBSCAN which overcomes many of these limitations.
      </p>
    `,
  },
  {
    title: "DBSCAN: Density-Based Clustering Complete Guide",
    description:
      "DBSCAN intuition, eps and min_samples parameters, core/border/noise points, k-distance graph for eps selection, and comparison with K-Means on non-spherical clusters.",
    date: "২৩ মে, ২০২৬",
    category: "Clustering",
    readTime: 10,
    slug: "clustering-dbscan",
    content: `
      <h3>1. What is DBSCAN and Why is it Special?</h3>
      <p>
        <strong>DBSCAN</strong> (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that is far more flexible than K-Means. With K-Means you must specify K upfront and it only finds spherical clusters. DBSCAN has two major advantages:
      </p>
      <p>
        <strong>1)</strong> No need to specify K — the algorithm determines the number of clusters automatically. <br/>
        <strong>2)</strong> Can find clusters of arbitrary shapes — crescents, rings, spirals, and more.
      </p>
      <p>
        The core idea is: if a region has high density of points, it forms a cluster. Points that are not in any dense region are labeled as <strong>noise</strong> (outliers).
      </p>

      <h3>2. Core Concepts: Three Types of Points</h3>
      <p>DBSCAN uses two hyperparameters:</p>
      <p>
        <strong>eps (ε):</strong> The radius of a point's "neighborhood." Points within eps distance are neighbors. <br/>
        <strong>min_samples:</strong> A point needs at least this many neighbors within eps to be a "core point."
      </p>
      <table>
        <thead>
          <tr>
            <th>Point Type</th>
            <th>Definition</th>
            <th>Label</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Core Point</td>
            <td>Has at least min_samples points within eps radius</td>
            <td>Any cluster label</td>
          </tr>
          <tr>
            <td>Border Point</td>
            <td>Within eps of a core point, but not a core itself</td>
            <td>That core's cluster label</td>
          </tr>
          <tr>
            <td>Noise Point</td>
            <td>Not a neighbor of any core point</td>
            <td>-1 (outlier)</td>
          </tr>
        </tbody>
      </table>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# make_moons: where K-Means fails, DBSCAN succeeds
X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric='euclidean'
)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise}")
print(f"Unique labels: {set(labels)}")</code></pre>

      <h3>3. Visualizing DBSCAN vs K-Means on Non-Spherical Data</h3>
      <pre><code>fig, axes = plt.subplots(1, 2, figsize=(14, 6))

unique_labels = set(labels)
colors_map = {-1: '#999999', 0: '#e74c3c', 1: '#3498db'}

for label in unique_labels:
    mask = labels == label
    color = colors_map.get(label, '#f39c12')
    marker = 'x' if label == -1 else 'o'
    size = 30 if label == -1 else 50
    name = 'Noise' if label == -1 else f'Cluster {label + 1}'
    axes[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=color, marker=marker, s=size, label=name, alpha=0.8)

axes[0].set_title('DBSCAN on make_moons', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)

for i, color in enumerate(['#e74c3c', '#3498db']):
    mask = km_labels == i
    axes[1].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=color, s=50, label=f'K-Means Cluster {i+1}', alpha=0.8)
axes[1].set_title('K-Means on make_moons — FAILS', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('DBSCAN vs K-Means: Non-Spherical Clusters', fontsize=14)
plt.tight_layout()
plt.savefig('dbscan_vs_kmeans.png', dpi=150)
plt.show()</code></pre>

      <h3>4. Choosing eps: The K-Distance Graph</h3>
      <p>
        The hardest part of DBSCAN is choosing eps. The <strong>K-Distance Graph</strong> helps: plot the k-th nearest neighbor distance for each point (sorted in descending order). The point where the curve bends sharply (the "elbow") is a good eps value.
      </p>
      <pre><code>from sklearn.neighbors import NearestNeighbors

k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

k_distances = np.sort(distances[:, k-1])[::-1]

plt.figure(figsize=(8, 5))
plt.plot(k_distances, linewidth=2, color='#e74c3c')
plt.xlabel('Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph — Choosing eps', fontsize=14)
plt.axhline(y=0.3, color='blue', linestyle='--', label='eps = 0.3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_distance_graph.png', dpi=150)
plt.show()

for eps_val in [0.2, 0.3, 0.4, 0.5]:
    db = DBSCAN(eps=eps_val, min_samples=5)
    lbs = db.fit_predict(X_scaled)
    nc = len(set(lbs)) - (1 if -1 in lbs else 0)
    nn = list(lbs).count(-1)
    print(f"eps={eps_val}: clusters={nc}, noise={nn}")</code></pre>

      <h3>5. K-Means vs DBSCAN: When to Use Which?</h3>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>K-Means</th>
            <th>DBSCAN</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Need to specify K?</td>
            <td>Yes</td>
            <td>No</td>
          </tr>
          <tr>
            <td>Cluster shape</td>
            <td>Spherical/convex only</td>
            <td>Any arbitrary shape</td>
          </tr>
          <tr>
            <td>Outlier handling</td>
            <td>Forced into a cluster</td>
            <td>Labeled as noise (-1)</td>
          </tr>
          <tr>
            <td>Large datasets</td>
            <td>Fast, scalable</td>
            <td>Relatively slower</td>
          </tr>
          <tr>
            <td>Best suited for</td>
            <td>Spherical clusters, few outliers</td>
            <td>Irregular shapes, outlier detection</td>
          </tr>
        </tbody>
      </table>
      <p>
        One limitation of DBSCAN is handling clusters of varying density — HDBSCAN (Hierarchical DBSCAN) addresses this. In the next post, we cover Hierarchical Clustering.
      </p>
    `,
  },
  {
    title: "Hierarchical Clustering and Dendrogram Analysis",
    description:
      "Agglomerative clustering from the ground up, how to read a dendrogram and choose a cut point, linkage methods compared, and full code on the Iris dataset.",
    date: "২৩ মে, ২০২৬",
    category: "Clustering",
    readTime: 10,
    slug: "clustering-hierarchical",
    content: `
      <h3>1. What is Hierarchical Clustering?</h3>
      <p>
        After K-Means and DBSCAN, we now explore <strong>Hierarchical Clustering</strong>, which builds a tree-like hierarchy of clusters. Its biggest advantage: you don't need to specify K upfront. Instead, you examine a <strong>Dendrogram</strong> and decide at which level to cut it to get the desired number of clusters.
      </p>
      <p>
        There are two main variants: <br/>
        <strong>Agglomerative (Bottom-Up):</strong> Each point starts as its own cluster, then clusters are merged iteratively. <br/>
        <strong>Divisive (Top-Down):</strong> All points start in one cluster and are split iteratively. (Less common)
      </p>

      <h3>2. Agglomerative Clustering: Step by Step</h3>
      <p>
        <strong>Step 1:</strong> With n points, create n individual clusters. <br/>
        <strong>Step 2:</strong> Find the two closest clusters. <br/>
        <strong>Step 3:</strong> Merge them into one cluster. <br/>
        <strong>Step 4:</strong> Repeat steps 2–3 until only one cluster remains.
      </p>
      <p>This entire merge history is preserved in a Dendrogram.</p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data
y_true = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scipy linkage matrix (needed for dendrogram)
Z = linkage(X_scaled, method='ward')

print("Linkage Matrix (first 5 rows):")
print("Format: [cluster_i, cluster_j, distance, count]")
print(Z[:5])</code></pre>

      <h3>3. Reading the Dendrogram: Where to Cut?</h3>
      <p>
        A dendrogram shows data points on the X-axis and merge distances on the Y-axis. Lower merges = closer clusters; higher merges = more distant clusters. To choose K, find the tallest vertical line not crossed by any horizontal line, then draw a horizontal line there. The number of vertical lines it crosses is your optimal K — this is the "largest gap" method.
      </p>
      <pre><code>plt.figure(figsize=(14, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    color_threshold=6.0,
)
plt.title('Dendrogram — Iris Dataset (Ward Linkage)', fontsize=14)
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.axhline(y=6.0, color='red', linestyle='--',
            linewidth=2, label='Cut point (distance=6.0)')
plt.legend()
plt.tight_layout()
plt.savefig('iris_dendrogram.png', dpi=150)
plt.show()

for n in [2, 3, 4]:
    agg = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = agg.fit_predict(X_scaled)
    print(f"n_clusters={n}: label distribution = {np.bincount(labels)}")</code></pre>

      <h3>4. Linkage Methods: Which to Use When?</h3>
      <table>
        <thead>
          <tr>
            <th>Linkage</th>
            <th>Distance Rule</th>
            <th>Characteristics</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Single</td>
            <td>Minimum distance between any two points</td>
            <td>Elongated clusters, chaining effect</td>
          </tr>
          <tr>
            <td>Complete</td>
            <td>Maximum distance between any two points</td>
            <td>Compact, roughly equal-sized clusters</td>
          </tr>
          <tr>
            <td>Average</td>
            <td>Mean of all pairwise distances</td>
            <td>Compromise between single and complete</td>
          </tr>
          <tr>
            <td>Ward</td>
            <td>Increase in total variance after merge</td>
            <td>Most popular, compact clusters</td>
          </tr>
        </tbody>
      </table>
      <pre><code>fig, axes = plt.subplots(1, 4, figsize=(18, 5))
linkage_methods = ['single', 'complete', 'average', 'ward']

for ax, method in zip(axes, linkage_methods):
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X_scaled)

    for i, color in enumerate(['#e74c3c', '#3498db', '#2ecc71']):
        mask = labels == i
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 2],
                   c=color, alpha=0.7, s=40, label=f'Cluster {i+1}')
    ax.set_title(f'{method.capitalize()} Linkage', fontsize=11)
    ax.set_xlabel('Sepal Length (scaled)')
    if method == 'single':
        ax.set_ylabel('Petal Length (scaled)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Iris Dataset: Linkage Method Comparison (n_clusters=3)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('linkage_comparison.png', dpi=150)
plt.show()</code></pre>

      <h3>5. n_clusters vs distance_threshold and When to Use Hierarchical</h3>
      <pre><code>agg_thresh = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=6.0,
    linkage='ward'
)
labels_thresh = agg_thresh.fit_predict(X_scaled)
print(f"Clusters with distance_threshold=6.0: {agg_thresh.n_clusters_}")

from sklearn.metrics import adjusted_rand_score
agg_3 = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_3 = agg_3.fit_predict(X_scaled)
ari = adjusted_rand_score(y_true, labels_3)
print(f"Adjusted Rand Index (ward, n=3): {ari:.3f}")
print("(1.0 = perfect, 0.0 = random)")</code></pre>
      <p>
        Hierarchical Clustering is most useful when you want to understand the cluster hierarchy, don't know K, or have a small-to-medium dataset. On large datasets it can be slow due to O(n²) complexity.
      </p>
    `,
  },
  {
    title: "Clustering Evaluation: Silhouette, Davies-Bouldin and More",
    description:
      "How to evaluate unsupervised clustering without ground truth labels — Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score with full code.",
    date: "২৩ মে, ২০২৬",
    category: "Clustering",
    readTime: 9,
    slug: "clustering-evaluation",
    content: `
      <h3>1. The Challenge of Evaluating Unsupervised Learning</h3>
      <p>
        In supervised learning, evaluation is straightforward — you have ground truth labels, so you compute accuracy or F1. But clustering has no "right answer." So how do you know which clustering is good?
      </p>
      <p>
        Good clustering rests on two principles: <br/>
        <strong>Cohesion:</strong> Points in the same cluster should be close to each other. <br/>
        <strong>Separation:</strong> Points in different clusters should be far from each other.
      </p>

      <h3>2. Inertia / WCSS: The Simplest Metric</h3>
      <p>
        <strong>Inertia</strong> (WCSS) is the sum of squared distances from each point to its cluster centroid. Lower inertia means tighter clusters. The problem: inertia always decreases as K increases — even K = n gives inertia = 0. Use it only in the Elbow Method, watching the rate of decrease.
      </p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

X, y_true = make_blobs(n_samples=500, centers=4,
                       cluster_std=0.9, random_state=42)

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

      <h3>3. Silhouette Score: The Best All-Around Metric</h3>
      <p>
        The <strong>Silhouette Score</strong> measures both cohesion and separation for every point. For each point i: <br/>
        <strong>a(i)</strong> = mean distance to other points in its own cluster (cohesion, lower is better) <br/>
        <strong>b(i)</strong> = mean distance to points in the nearest other cluster (separation, higher is better) <br/>
        <strong>s(i)</strong> = (b(i) - a(i)) / max(a(i), b(i))
      </p>
      <p>
        Ranges from -1 to +1. +1 means perfectly clustered, 0 means on the boundary, -1 means likely misclassified.
      </p>
      <pre><code>from sklearn.metrics import silhouette_samples

best_k = list(K_range)[np.argmax(silhouette_scores)]
print(f"Best Silhouette Score at K={best_k}: {max(silhouette_scores):.3f}")

km_best = KMeans(n_clusters=best_k, init='k-means++',
                 n_init=10, random_state=42)
labels_best = km_best.fit_predict(X)
sample_silhouette = silhouette_samples(X, labels_best)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
axes[0].legend()

axes[1].plot(list(K_range), silhouette_scores, 'go-')
axes[1].set_title('Silhouette Score vs K', fontsize=12)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=150)
plt.show()</code></pre>

      <h3>4. Davies-Bouldin and Calinski-Harabasz Indices</h3>
      <p>
        <strong>Davies-Bouldin Index (DBI):</strong> Measures the ratio of within-cluster scatter to between-cluster separation. Lower is better. <br/>
        <strong>Calinski-Harabasz Score (CH):</strong> Ratio of between-cluster variance to within-cluster variance. Higher is better.
      </p>
      <pre><code>fig, axes = plt.subplots(2, 2, figsize=(13, 10))

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
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(list(K_range), db_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_title('Davies-Bouldin Index (lower=better)', fontsize=12)
axes[1, 0].set_xlabel('K')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(list(K_range), ch_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_title('Calinski-Harabasz Score (higher=better)', fontsize=12)
axes[1, 1].set_xlabel('K')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Clustering Evaluation Metrics Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('clustering_metrics.png', dpi=150)
plt.show()</code></pre>

      <h3>5. Metrics Summary: Which One to Use?</h3>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Range</th>
            <th>Better Direction</th>
            <th>Pro</th>
            <th>Con</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Inertia (WCSS)</td>
            <td>0 to ∞</td>
            <td>Lower is better</td>
            <td>Simple, fast</td>
            <td>Always decreases with K</td>
          </tr>
          <tr>
            <td>Silhouette Score</td>
            <td>-1 to +1</td>
            <td>Closer to +1</td>
            <td>Measures cohesion + separation</td>
            <td>Slow on large datasets</td>
          </tr>
          <tr>
            <td>Davies-Bouldin</td>
            <td>0 to ∞</td>
            <td>Lower is better</td>
            <td>Fast to compute</td>
            <td>Centroid-based only</td>
          </tr>
          <tr>
            <td>Calinski-Harabasz</td>
            <td>0 to ∞</td>
            <td>Higher is better</td>
            <td>Fast, works on large data</td>
            <td>Assumes convex clusters</td>
          </tr>
        </tbody>
      </table>
      <p>
        In practice, don't rely on a single metric — use several together. <strong>Silhouette Score</strong> is typically the most reliable. If you have ground truth labels, also use <strong>Adjusted Rand Index (ARI)</strong> or <strong>Normalized Mutual Information (NMI)</strong>.
      </p>
    `,
  },
  {
    title: "Customer Segmentation: Full Project with K-Means and DBSCAN",
    description:
      "End-to-end customer segmentation pipeline — EDA, feature scaling, finding optimal K with elbow and silhouette, K-Means clustering, cluster interpretation, DBSCAN comparison, and business insights.",
    date: "২৩ মে, ২০২৬",
    category: "Clustering",
    readTime: 13,
    slug: "clustering-project",
    content: `
      <h3>1. Project Introduction: Why Customer Segmentation?</h3>
      <p>
        A shopping mall's marketing team wants to divide their customers into meaningful groups so each group can be targeted with a tailored strategy. This is <strong>Customer Segmentation</strong>, and clustering is the ideal tool for the job.
      </p>
      <p>
        We'll use two features: <br/>
        <strong>Annual Income (k$):</strong> The customer's annual income. <br/>
        <strong>Spending Score (1-100):</strong> A mall-assigned score based on how much a customer spends.
      </p>
      <p>
        Full pipeline: generate data → EDA → feature scaling → find optimal K → K-Means → interpret clusters → DBSCAN comparison → business insights.
      </p>

      <h3>2. Data Generation and Exploratory Data Analysis</h3>
      <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# Synthetic mall customers: 5 natural groups
n = 40
income_groups = [(20, 5), (20, 5), (55, 8), (85, 8), (85, 8)]
spending_groups = [(75, 8), (25, 8), (50, 10), (80, 8), (20, 8)]

incomes, scores = [], []
for (inc_mean, inc_std), (sc_mean, sc_std) in zip(income_groups, spending_groups):
    incomes.extend(np.random.normal(inc_mean, inc_std, n).clip(10, 120))
    scores.extend(np.random.normal(sc_mean, sc_std, n).clip(1, 100))

df = pd.DataFrame({
    'CustomerID': range(1, len(incomes) + 1),
    'Annual_Income': np.round(incomes, 1),
    'Spending_Score': np.round(scores, 1)
})

print(df.describe())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(df['Annual_Income'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].set_title('Annual Income Distribution')
axes[0].set_xlabel('Annual Income (k$)')

axes[1].hist(df['Spending_Score'], bins=20, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_title('Spending Score Distribution')
axes[1].set_xlabel('Spending Score (1-100)')

axes[2].scatter(df['Annual_Income'], df['Spending_Score'], alpha=0.6, color='#9b59b6', s=50)
axes[2].set_title('Income vs Spending Score')
axes[2].set_xlabel('Annual Income (k$)')
axes[2].set_ylabel('Spending Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('customer_eda.png', dpi=150)
plt.show()</code></pre>

      <h3>3. Feature Scaling and Finding Optimal K</h3>
      <pre><code>X = df[['Annual_Income', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method', fontsize=13)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].grid(True, linestyle='--', alpha=0.7)

best_k = list(K_range)[np.argmax(silhouette_scores)]
axes[1].plot(list(K_range), silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=best_k, color='blue', linestyle='--', label=f'Best K={best_k}')
axes[1].set_title('Silhouette Score', fontsize=13)
axes[1].set_xlabel('K')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.suptitle(f'Optimal K Selection (Best = {best_k})', fontsize=14)
plt.tight_layout()
plt.savefig('optimal_k.png', dpi=150)
plt.show()
print(f"Both Elbow and Silhouette suggest K={best_k}")</code></pre>

      <h3>4. K-Means Clustering and Cluster Interpretation</h3>
      <pre><code>kmeans_final = KMeans(n_clusters=5, init='k-means++',
                      n_init=15, random_state=42)
df['KMeans_Cluster'] = kmeans_final.fit_predict(X_scaled)
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

cluster_stats = df.groupby('KMeans_Cluster').agg({
    'Annual_Income': ['mean', 'std', 'count'],
    'Spending_Score': ['mean', 'std']
}).round(2)
print(cluster_stats)

cluster_names = {}
for i, (inc, sc) in enumerate(centroids_original):
    if inc < 45 and sc > 60:
        cluster_names[i] = 'Impulsive (Low income, high spending)'
    elif inc < 45 and sc < 40:
        cluster_names[i] = 'Conservative (Low income, low spending)'
    elif inc > 65 and sc > 60:
        cluster_names[i] = 'Target (High income, high spending)'
    elif inc > 65 and sc < 40:
        cluster_names[i] = 'Careful (High income, low spending)'
    else:
        cluster_names[i] = 'Sensible (Average income and spending)'

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
plt.figure(figsize=(11, 8))

for i in range(5):
    mask = df['KMeans_Cluster'] == i
    plt.scatter(df[mask]['Annual_Income'], df[mask]['Spending_Score'],
                c=colors[i], s=60, alpha=0.8,
                label=cluster_names.get(i, f'Cluster {i}'),
                edgecolors='white', linewidths=0.5)

plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
            c='black', marker='*', s=300, label='Centroids', zorder=6)

plt.title('Customer Segmentation — K-Means (K=5)', fontsize=15)
plt.xlabel('Annual Income (k$)', fontsize=13)
plt.ylabel('Spending Score (1-100)', fontsize=13)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_kmeans.png', dpi=150)
plt.show()</code></pre>

      <h3>5. DBSCAN Comparison and Business Insights</h3>
      <pre><code>nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_dist = np.sort(distances[:, 4])[::-1]

plt.figure(figsize=(8, 4))
plt.plot(k_dist, linewidth=2, color='#e74c3c')
plt.axhline(y=0.5, color='blue', linestyle='--', label='eps=0.5')
plt.title('K-Distance Graph (k=5)', fontsize=13)
plt.xlabel('Points (sorted)')
plt.ylabel('5th Nearest Neighbor Distance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_kdistance.png', dpi=150)
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df['DBSCAN_Cluster'])) - (
    1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise_db = (df['DBSCAN_Cluster'] == -1).sum()
print(f"DBSCAN: {n_clusters_db} clusters, {n_noise_db} noise points")

# Business Insights
print("\n" + "="*55)
print("BUSINESS INSIGHT SUMMARY")
print("="*55)
insights = {
    'Target (High income, high spending)':
        'Highest value customers. Premium offers and loyalty programs.',
    'Conservative (Low income, low spending)':
        'Budget-conscious. Discounts and installment plans.',
    'Impulsive (Low income, high spending)':
        'Impulse buyers. Flash sales and limited-time offers.',
    'Careful (High income, low spending)':
        'Value-conscious affluent. Emphasize ROI of premium products.',
    'Sensible (Average)':
        'Balanced customers. Standard promotional campaigns.',
}
for segment, strategy in insights.items():
    print(f"\n{segment}:")
    print(f"  Strategy: {strategy}")</code></pre>

      <h3>6. Key Takeaways from the Project</h3>
      <table>
        <thead>
          <tr>
            <th>Aspect</th>
            <th>K-Means</th>
            <th>DBSCAN</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Result on this data</td>
            <td>5 clean spherical clusters</td>
            <td>Clusters with noise separation</td>
          </tr>
          <tr>
            <td>Business interpretation</td>
            <td>Easy, each cluster is clear</td>
            <td>Isolates outlier customers</td>
          </tr>
          <tr>
            <td>Scaling required?</td>
            <td>Yes (essential)</td>
            <td>Yes (essential)</td>
          </tr>
          <tr>
            <td>Better for this dataset?</td>
            <td>K-Means (spherical clusters)</td>
            <td>DBSCAN (for outlier detection)</td>
          </tr>
        </tbody>
      </table>
      <p>
        This project showed how clustering solves a real business problem. The main takeaways: understand your data structure before choosing an algorithm; always scale features; evaluate with multiple metrics; and interpret clusters using domain knowledge.
      </p>
    `,
  },
];
