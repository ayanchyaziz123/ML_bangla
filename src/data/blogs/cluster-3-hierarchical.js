export const cluster_3_hierarchical = {
  title: "Hierarchical Clustering ও Dendrogram বিশ্লেষণ",
  description:
    "Agglomerative Clustering-এর নীতি, Dendrogram কীভাবে পড়তে হয়, Linkage পদ্ধতি, এবং Iris dataset-এ সম্পূর্ণ কোড উদাহরণ।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 10,
  slug: "clustering-hierarchical",
  content: `
    <h3>১. Hierarchical Clustering কী?</h3>
    <p>
      K-Means এবং DBSCAN-এর পরে আমরা এখন <strong>Hierarchical Clustering</strong> দেখব, যা ক্লাস্টারগুলোর মধ্যে একটি গাছের মতো (tree-like) সম্পর্ক তৈরি করে। এর সবচেয়ে বড় সুবিধা হলো: আপনাকে আগে থেকে K বলতে হয় না। বরং একটি <strong>Dendrogram</strong> দেখে আপনি নিজে সিদ্ধান্ত নিতে পারেন কোন level-এ কাটলে কতটি cluster পাবেন।
    </p>
    <p>
      দুটি প্রধান ধরনের Hierarchical Clustering আছে: <br/>
      <strong>Agglomerative (Bottom-Up):</strong> প্রতিটি পয়েন্ট আলাদা cluster হিসেবে শুরু করে, তারপর ধীরে ধীরে merge করতে থাকে। <br/>
      <strong>Divisive (Top-Down):</strong> সব পয়েন্ট একটি cluster থেকে শুরু করে, তারপর ভাগ করতে থাকে। (কম ব্যবহৃত)
    </p>
    <p>
      sklearn-এ সাধারণত <strong>AgglomerativeClustering</strong> ব্যবহার হয়।
    </p>

    <h3>২. Agglomerative Clustering: ধাপে ধাপে</h3>
    <p>
      <strong>ধাপ ১:</strong> n পয়েন্ট থাকলে n টি আলাদা cluster তৈরি হয়। <br/>
      <strong>ধাপ ২:</strong> সবচেয়ে কাছের দুটি cluster খুঁজে বের করো। <br/>
      <strong>ধাপ ৩:</strong> সেই দুটি cluster merge করো। <br/>
      <strong>ধাপ ৪:</strong> মাত্র ১টি cluster না হওয়া পর্যন্ত ধাপ ২-৩ পুনরাবৃত্তি করো।
    </p>
    <p>
      এই পুরো merge history-টি একটি Dendrogram-এ সংরক্ষিত থাকে।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Iris dataset লোড করুন
iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scipy linkage matrix তৈরি করুন (dendrogram-এর জন্য)
# ward linkage ব্যবহার করা হচ্ছে
Z = linkage(X_scaled, method='ward')

print("Linkage Matrix (first 5 rows):")
print("Format: [cluster_i, cluster_j, distance, count]")
print(Z[:5])</code></pre>

    <h3>৩. Dendrogram: কীভাবে পড়তে হয় এবং কোথায় Cut করবেন?</h3>
    <p>
      Dendrogram হলো hierarchical clustering-এর চাক্ষুষ উপস্থাপনা। X-axis-এ পয়েন্টগুলো, Y-axis-এ merge-এর দূরত্ব। নিচের merge মানে কাছাকাছি clusters, উপরের merge মানে দূরের clusters।
    </p>
    <p>
      <strong>কোথায় cut করবেন?</strong> যেখানে Y-axis-এ সবচেয়ে লম্বা vertical line আছে সেখানে horizontal line টানুন। সেই line কতটি vertical line কাটে — সেটাই optimal cluster সংখ্যা। এটিকে বলে "largest gap" পদ্ধতি।
    </p>
    <pre><code># Dendrogram visualization
plt.figure(figsize=(14, 7))

dendrogram(
    Z,
    truncate_mode='lastp',  # শুধু শেষের p merge দেখাবে
    p=30,                   # 30টি leaf দেখাবে
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    color_threshold=6.0,    # এই height-এ রঙ পরিবর্তন
)

plt.title('Dendrogram — Iris Dataset (Ward Linkage)', fontsize=14)
plt.xlabel('Sample Index বা (Cluster Size)', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.axhline(y=6.0, color='red', linestyle='--',
            linewidth=2, label='Cut point (distance=6.0)')
plt.legend()
plt.tight_layout()
plt.savefig('iris_dendrogram.png', dpi=150)
plt.show()

# AgglomerativeClustering দিয়ে fit
for n in [2, 3, 4]:
    agg = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = agg.fit_predict(X_scaled)
    print(f"n_clusters={n}: label distribution = {np.bincount(labels)}")</code></pre>

    <h3>৪. Linkage পদ্ধতি: কোনটি কখন ব্যবহার করবেন?</h3>
    <p>দুটি cluster-এর মধ্যে দূরত্ব কীভাবে মাপা হবে সেটা Linkage নির্ধারণ করে:</p>
    <table>
      <thead>
        <tr>
          <th>Linkage</th>
          <th>দূরত্ব মাপার নিয়ম</th>
          <th>বৈশিষ্ট্য</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Single</td>
          <td>দুই cluster-এর সবচেয়ে কাছের পয়েন্টের দূরত্ব</td>
          <td>লম্বা, চিকন cluster তৈরি করে (chaining effect)</td>
        </tr>
        <tr>
          <td>Complete</td>
          <td>দুই cluster-এর সবচেয়ে দূরের পয়েন্টের দূরত্ব</td>
          <td>গোলাকার, সমান আকৃতির cluster</td>
        </tr>
        <tr>
          <td>Average</td>
          <td>সব পয়েন্ট-জোড়ার গড় দূরত্ব</td>
          <td>Single ও Complete-এর মধ্যম</td>
        </tr>
        <tr>
          <td>Ward</td>
          <td>Merge করলে মোট variance কতটা বাড়ে</td>
          <td>সবচেয়ে জনপ্রিয়, compact cluster</td>
        </tr>
      </tbody>
    </table>
    <pre><code># বিভিন্ন linkage তুলনা
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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

plt.suptitle('Iris Dataset: বিভিন্ন Linkage-এর তুলনা (n_clusters=3)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('linkage_comparison.png', dpi=150)
plt.show()</code></pre>

    <h3>৫. n_clusters বনাম distance_threshold এবং কখন ব্যবহার করবেন?</h3>
    <p>
      sklearn AgglomerativeClustering-এ দুটি উপায়ে cluster সংখ্যা নিয়ন্ত্রণ করা যায়:
    </p>
    <p>
      <strong>n_clusters:</strong> সরাসরি cluster সংখ্যা দিন। সহজ এবং সাধারণত এটাই যথেষ্ট।
    </p>
    <p>
      <strong>distance_threshold:</strong> এই দূরত্বের বেশি হলে merge করবে না। Dendrogram দেখে এই threshold বেছে নিন। n_clusters=None রাখতে হবে।
    </p>
    <pre><code># distance_threshold ব্যবহার
agg_thresh = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=6.0,  # dendrogram থেকে বেছে নেওয়া
    linkage='ward'
)
labels_thresh = agg_thresh.fit_predict(X_scaled)
print(f"distance_threshold=6.0 দিয়ে cluster সংখ্যা: {agg_thresh.n_clusters_}")

# Accuracy পরীক্ষা (true label থাকলে)
from sklearn.metrics import adjusted_rand_score
agg_3 = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_3 = agg_3.fit_predict(X_scaled)
ari = adjusted_rand_score(y_true, labels_3)
print(f"Adjusted Rand Index (ward, n=3): {ari:.3f}")
print("(1.0 = নিখুঁত, 0.0 = random)")</code></pre>
    <p>
      Hierarchical Clustering তখন সবচেয়ে কার্যকর যখন আপনি cluster hierarchy বুঝতে চান, K জানেন না, বা ডেটাসেট small-to-medium আকারের (কয়েক হাজার পয়েন্ট পর্যন্ত)। বড় ডেটাসেটে O(n²) complexity-র কারণে ধীর হতে পারে।
    </p>
  `,
};
