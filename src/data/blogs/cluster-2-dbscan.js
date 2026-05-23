export const cluster_2_dbscan = {
  title: "DBSCAN: ঘনত্ব-ভিত্তিক ক্লাস্টারিং সম্পূর্ণ গাইড",
  description:
    "DBSCAN অ্যালগরিদমের intuition, eps ও min_samples প্যারামিটার, core/border/noise পয়েন্ট, এবং K-Means-এর সাথে তুলনা — বাস্তব কোড সহ।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 10,
  slug: "clustering-dbscan",
  content: `
    <h3>১. DBSCAN কী এবং কেন এটি বিশেষ?</h3>
    <p>
      <strong>DBSCAN</strong> (Density-Based Spatial Clustering of Applications with Noise) হলো একটি ঘনত্ব-ভিত্তিক ক্লাস্টারিং অ্যালগরিদম যা K-Means-এর তুলনায় অনেক বেশি নমনীয়। K-Means-এ আপনাকে আগে থেকে K বলতে হয় এবং এটি শুধু গোলাকার cluster চেনে। DBSCAN-এর দুটি বড় সুবিধা:
    </p>
    <p>
      <strong>১)</strong> K (cluster সংখ্যা) আগে থেকে বলতে হয় না — অ্যালগরিদম নিজেই ঠিক করে। <br/>
      <strong>২)</strong> যেকোনো আকৃতির cluster চিনতে পারে — চাঁদ, আংটি, বা spiral আকৃতিও।
    </p>
    <p>
      মূল ধারণা হলো: যদি কোনো এলাকায় পয়েন্টের ঘনত্ব বেশি হয়, সেটা একটি cluster। আর যে পয়েন্টগুলো কোনো ঘন এলাকায় নেই, সেগুলো <strong>noise</strong> বা outlier।
    </p>

    <h3>২. DBSCAN-এর মূল ধারণা: তিন ধরনের পয়েন্ট</h3>
    <p>DBSCAN দুটি hyperparameter ব্যবহার করে:</p>
    <p>
      <strong>eps (ε):</strong> কোনো পয়েন্টের "neighborhood"-এর radius। eps-এর মধ্যে যত পয়েন্ট আছে তারা সেই পয়েন্টের প্রতিবেশী। <br/>
      <strong>min_samples:</strong> একটি পয়েন্টকে "core point" হতে হলে তার eps-radius-এ কমপক্ষে এই সংখ্যক পয়েন্ট থাকতে হবে।
    </p>
    <p>এই দুটি দিয়ে তিন ধরনের পয়েন্ট সংজ্ঞায়িত হয়:</p>
    <table>
      <thead>
        <tr>
          <th>পয়েন্টের ধরন</th>
          <th>সংজ্ঞা</th>
          <th>Label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Core Point</td>
          <td>eps-radius-এ কমপক্ষে min_samples পয়েন্ট আছে</td>
          <td>যেকোনো cluster label</td>
        </tr>
        <tr>
          <td>Border Point</td>
          <td>কোনো core point-এর eps-radius-এ আছে, কিন্তু নিজে core নয়</td>
          <td>সেই core-এর cluster label</td>
        </tr>
        <tr>
          <td>Noise Point</td>
          <td>কোনো core point-এর প্রতিবেশী নয়</td>
          <td>-1 (outlier)</td>
        </tr>
      </tbody>
    </table>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# make_moons: K-Means যেখানে fail করে DBSCAN সেখানে সফল
X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN model
dbscan = DBSCAN(
    eps=0.3,           # neighborhood radius
    min_samples=5,     # core point হতে ন্যূনতম প্রতিবেশী
    metric='euclidean' # দূরত্বের মাপকাঠি
)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"ক্লাস্টার সংখ্যা: {n_clusters}")
print(f"Noise পয়েন্ট সংখ্যা: {n_noise}")
print(f"Unique labels: {set(labels)}")</code></pre>

    <h3>৩. DBSCAN অ্যালগরিদম: বিস্তারিত</h3>
    <p>DBSCAN কীভাবে cluster খোঁজে তার ধাপগুলো:</p>
    <p>
      <strong>ধাপ ১:</strong> একটি unvisited পয়েন্ট P বেছে নাও।<br/>
      <strong>ধাপ ২:</strong> P-এর eps-neighborhood হিসাব করো।<br/>
      <strong>ধাপ ৩:</strong> যদি neighborhood-এ min_samples-এর কম পয়েন্ট থাকে, P-কে noise mark করো।<br/>
      <strong>ধাপ ৪:</strong> যদি যথেষ্ট পয়েন্ট থাকে (core point), নতুন cluster শুরু করো।<br/>
      <strong>ধাপ ৫:</strong> সেই cluster-এ সব reachable পয়েন্ট add করো (recursively)।<br/>
      <strong>ধাপ ৬:</strong> সব পয়েন্ট visit না হওয়া পর্যন্ত পুনরাবৃত্তি করো।
    </p>
    <pre><code># DBSCAN visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# DBSCAN result
unique_labels = set(labels)
colors_map = {-1: '#999999', 0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}

for label in unique_labels:
    mask = labels == label
    color = colors_map.get(label, '#f39c12')
    marker = 'x' if label == -1 else 'o'
    size = 30 if label == -1 else 50
    name = 'Noise' if label == -1 else f'Cluster {label + 1}'
    axes[0].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=color, marker=marker, s=size, label=name, alpha=0.8)

axes[0].set_title('DBSCAN (make_moons)', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# K-Means comparison
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)

for i, color in enumerate(['#e74c3c', '#3498db']):
    mask = km_labels == i
    axes[1].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=color, s=50, label=f'K-Means Cluster {i+1}', alpha=0.8)
axes[1].set_title('K-Means (make_moons) — ব্যর্থ', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('DBSCAN vs K-Means: অ-গোলাকার ক্লাস্টার', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('dbscan_vs_kmeans.png', dpi=150)
plt.show()</code></pre>

    <h3>৪. সঠিক eps কীভাবে বেছে নেবেন: K-Distance Graph</h3>
    <p>
      DBSCAN-এর সবচেয়ে কঠিন কাজ হলো eps নির্বাচন। <strong>K-Distance Graph</strong> (বা Elbow in distances) এতে সাহায্য করে। প্রতিটি পয়েন্টের k-তম nearest neighbor-এর দূরত্ব sort করে plot করুন — যেখানে হঠাৎ curve বাঁকে সেটাই ভালো eps।
    </p>
    <pre><code>from sklearn.neighbors import NearestNeighbors
import numpy as np

# k = min_samples হিসেবে ব্যবহার করুন
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# k-তম neighbor-এর দূরত্ব sort করুন
k_distances = np.sort(distances[:, k-1])[::-1]

plt.figure(figsize=(8, 5))
plt.plot(k_distances, linewidth=2, color='#e74c3c')
plt.xlabel('পয়েন্ট (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-তম Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph — eps নির্বাচনের জন্য', fontsize=14)
plt.axhline(y=0.3, color='blue', linestyle='--',
            label='eps = 0.3 (elbow point)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_distance_graph.png', dpi=150)
plt.show()

# eps পরীক্ষা করুন
for eps_val in [0.2, 0.3, 0.4, 0.5]:
    db = DBSCAN(eps=eps_val, min_samples=5)
    lbs = db.fit_predict(X_scaled)
    nc = len(set(lbs)) - (1 if -1 in lbs else 0)
    nn = list(lbs).count(-1)
    print(f"eps={eps_val}: clusters={nc}, noise={nn}")</code></pre>

    <h3>৫. K-Means বনাম DBSCAN: কখন কোনটি ব্যবহার করবেন?</h3>
    <table>
      <thead>
        <tr>
          <th>বৈশিষ্ট্য</th>
          <th>K-Means</th>
          <th>DBSCAN</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>K আগে নির্দিষ্ট করতে হয়?</td>
          <td>হ্যাঁ</td>
          <td>না</td>
        </tr>
        <tr>
          <td>Cluster আকৃতি</td>
          <td>শুধু spherical/convex</td>
          <td>যেকোনো আকৃতি</td>
        </tr>
        <tr>
          <td>Outlier handling</td>
          <td>Outlier-কে cluster-এ ফেলে</td>
          <td>Outlier আলাদা করে (-1)</td>
        </tr>
        <tr>
          <td>বড় ডেটাসেট</td>
          <td>দ্রুত, scalable</td>
          <td>তুলনামূলক ধীর</td>
        </tr>
        <tr>
          <td>ব্যবহার উপযুক্ত কখন</td>
          <td>গোলাকার cluster, outlier কম</td>
          <td>অদ্ভুত আকৃতি, outlier বেশি</td>
        </tr>
      </tbody>
    </table>
    <p>
      DBSCAN-এর একটি সীমাবদ্ধতা হলো ভিন্ন ঘনত্বের cluster একসাথে ভালো করে চেনে না। সেক্ষেত্রে HDBSCAN (Hierarchical DBSCAN) ব্যবহার করা যায়। পরবর্তী পোস্টে আমরা Hierarchical Clustering দেখব।
    </p>
  `,
};
