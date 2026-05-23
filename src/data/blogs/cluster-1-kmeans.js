export const cluster_1_kmeans = {
  title: "K-Means ক্লাস্টারিং: ধাপে ধাপে শিখুন",
  description:
    "K-Means অ্যালগরিদম কীভাবে কাজ করে, K-Means++ initialization, Elbow Method, এবং sklearn দিয়ে বাস্তব উদাহরণ সহ সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 11,
  slug: "clustering-kmeans",
  content: `
    <h3>১. ক্লাস্টারিং কী এবং K-Means কেন?</h3>
    <p>
      মেশিন লার্নিং-এ <strong>ক্লাস্টারিং</strong> হলো একটি unsupervised learning পদ্ধতি, যেখানে আমরা কোনো label ছাড়াই ডেটাকে অর্থপূর্ণ গ্রুপে ভাগ করি। কল্পনা করুন আপনার কাছে হাজারো গ্রাহকের ডেটা আছে — ক্লাস্টারিং এই গ্রাহকদের আচরণ অনুযায়ী স্বয়ংক্রিয়ভাবে গ্রুপ তৈরি করতে পারে।
    </p>
    <p>
      <strong>K-Means</strong> সবচেয়ে জনপ্রিয় এবং সহজ ক্লাস্টারিং অ্যালগরিদম। এটি ডেটা পয়েন্টগুলোকে K সংখ্যক গ্রুপে ভাগ করে, যেখানে প্রতিটি পয়েন্ট তার সবচেয়ে কাছের গ্রুপের কেন্দ্রে (centroid) assign হয়। অ্যালগরিদমটি iterative — এটি বারবার চলে যতক্ষণ না clusters স্থিতিশীল হয়।
    </p>

    <h3>২. K-Means অ্যালগরিদম: ধাপে ধাপে</h3>
    <p>K-Means চারটি মূল ধাপে কাজ করে:</p>
    <p>
      <strong>ধাপ ১ — Initialization:</strong> K সংখ্যক centroid এলোমেলোভাবে বেছে নাও। <br/>
      <strong>ধাপ ২ — Assignment:</strong> প্রতিটি ডেটা পয়েন্টকে সবচেয়ে কাছের centroid-এর cluster-এ assign করো। <br/>
      <strong>ধাপ ৩ — Update:</strong> প্রতিটি cluster-এর নতুন centroid হিসাব করো (সকল পয়েন্টের গড়)। <br/>
      <strong>ধাপ ৪ — Repeat:</strong> Centroid পরিবর্তন না হওয়া পর্যন্ত ধাপ ২ ও ৩ পুনরাবৃত্তি করো।
    </p>
    <p>
      দূরত্ব পরিমাপে সাধারণত <strong>Euclidean distance</strong> ব্যবহার হয়:
      d(x, μ) = √Σ(xᵢ - μᵢ)²
    </p>

    <h3>৩. K-Means++ Initialization</h3>
    <p>
      সাধারণ K-Means-এ centroid এলোমেলোভাবে বেছে নেওয়া হয়, যা মাঝে মাঝে খারাপ ফলাফল দিতে পারে। <strong>K-Means++</strong> এই সমস্যা সমাধান করে স্মার্ট initialization দিয়ে:
    </p>
    <p>
      প্রথম centroid এলোমেলো। পরের প্রতিটি centroid বেছে নেওয়া হয় এমনভাবে যে, বিদ্যমান centroid থেকে যত দূরের পয়েন্ট, তার বেছে নেওয়ার সম্ভাবনা তত বেশি। এটি clusters-কে সুষমভাবে ছড়িয়ে দেয় এবং convergence দ্রুত হয়।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Synthetic dataset তৈরি করা
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.8, random_state=42)

# K-Means++ (sklearn-এর default init='k-means++')
kmeans_pp = KMeans(n_clusters=4, init='k-means++',
                   max_iter=300, n_init=10, random_state=42)

# সাধারণ random initialization
kmeans_rand = KMeans(n_clusters=4, init='random',
                     max_iter=300, n_init=10, random_state=42)

kmeans_pp.fit(X)
kmeans_rand.fit(X)

print(f"K-Means++ Inertia:  {kmeans_pp.inertia_:.2f}")
print(f"K-Means Random Inertia: {kmeans_rand.inertia_:.2f}")
print(f"K-Means++ Iterations:  {kmeans_pp.n_iter_}")
print(f"K-Means Random Iterations: {kmeans_rand.n_iter_}")</code></pre>

    <h3>৪. Elbow Method: সঠিক K কীভাবে বেছে নেবেন?</h3>
    <p>
      K-Means-এর সবচেয়ে বড় প্রশ্ন হলো: <em>K কত হবে?</em> <strong>Elbow Method</strong> এর উত্তর দেয়। এটি বিভিন্ন K-এর জন্য <strong>WCSS (Within-Cluster Sum of Squares)</strong> বা Inertia প্লট করে।
    </p>
    <p>
      Inertia হলো প্রতিটি পয়েন্ট থেকে তার cluster centroid-এর দূরত্বের বর্গের যোগফল। K বাড়লে Inertia কমে, কিন্তু একটি বিন্দুতে কমার হার হঠাৎ কমে যায় — সেটাই "Elbow" বা কনুই। সেই K-টিই optimal।
    </p>
    <pre><code>inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++',
                max_iter=300, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K (ক্লাস্টার সংখ্যা)', fontsize=13)
plt.ylabel('Inertia (WCSS)', fontsize=13)
plt.title('Elbow Method — সঠিক K নির্বাচন', fontsize=15)
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()

# Elbow point প্রায়ই K=4 হবে এই ডেটায়
print("Elbow point: K=4 এর কাছে inertia কমার হার উল্লেখযোগ্যভাবে কমে")</code></pre>

    <h3>৫. sklearn KMeans: সম্পূর্ণ উদাহরণ ও Visualization</h3>
    <p>এখন সম্পূর্ণ pipeline দেখা যাক — ডেটা তৈরি, model fit, এবং cluster visualization:</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# ডেটা তৈরি করুন
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=1.0, random_state=42)

# Feature scaling (K-Means দূরত্ব-ভিত্তিক, তাই scaling গুরুত্বপূর্ণ)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means model
kmeans = KMeans(
    n_clusters=4,        # cluster সংখ্যা
    init='k-means++',    # স্মার্ট initialization
    max_iter=300,        # সর্বোচ্চ iteration
    n_init=10,           # ভিন্ন initialization কতবার চেষ্টা
    random_state=42
)
kmeans.fit(X_scaled)

labels = kmeans.labels_          # প্রতিটি পয়েন্টের cluster label
centroids = kmeans.cluster_centers_  # centroid coordinates
inertia = kmeans.inertia_         # total WCSS

print(f"Cluster labels (first 10): {labels[:10]}")
print(f"Inertia: {inertia:.2f}")
print(f"Iterations to converge: {kmeans.n_iter_}")

# Cluster visualization
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
plt.figure(figsize=(10, 7))

for i in range(4):
    mask = labels == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=colors[i], label=f'Cluster {i+1}',
                alpha=0.7, s=50, edgecolors='white', linewidths=0.5)

# Centroid plot
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

    <h3>৬. K-Means-এর সীমাবদ্ধতা</h3>
    <p>K-Means শক্তিশালী হলেও কিছু গুরুত্বপূর্ণ সীমাবদ্ধতা আছে:</p>
    <table>
      <thead>
        <tr>
          <th>সীমাবদ্ধতা</th>
          <th>ব্যাখ্যা</th>
          <th>সমাধান</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>K আগে থেকে নির্দিষ্ট করতে হয়</td>
          <td>সঠিক K না জানলে ভুল ফলাফল</td>
          <td>Elbow method, Silhouette score</td>
        </tr>
        <tr>
          <td>শুধু spherical cluster</td>
          <td>অ-গোলাকার আকৃতির cluster ধরতে পারে না</td>
          <td>DBSCAN বা Spectral Clustering ব্যবহার করুন</td>
        </tr>
        <tr>
          <td>Outlier-এ সংবেদনশীল</td>
          <td>Outlier centroid টেনে নিয়ে যায়</td>
          <td>K-Medoids বা DBSCAN ব্যবহার করুন</td>
        </tr>
        <tr>
          <td>Local minimum</td>
          <td>Random initialization-এ খারাপ ফলাফল হতে পারে</td>
          <td>K-Means++ এবং n_init বাড়ান</td>
        </tr>
      </tbody>
    </table>
    <p>
      K-Means তখন সবচেয়ে ভালো কাজ করে যখন clusters গোলাকার, প্রায় সমান আকারের, এবং ডেটাতে outlier কম থাকে। পরবর্তী পোস্টে আমরা DBSCAN দেখব যা এই সীমাবদ্ধতাগুলো অনেকটা কাটিয়ে উঠতে পারে।
    </p>
  `,
};
