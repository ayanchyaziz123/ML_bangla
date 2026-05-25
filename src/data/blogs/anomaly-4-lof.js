export const anomaly_4_lof = {
  slug: 'anomaly-4-lof',
  title: 'Local Outlier Factor: প্রতিবেশী-ভিত্তিক অ্যানোমালি ডিটেকশন',
  description: 'LOF অ্যালগরিদম, local reachability density, LOF score-এর ব্যাখ্যা, k-distance, contextual anomaly সনাক্তকরণ এবং sklearn LocalOutlierFactor দিয়ে সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'অ্যানোমালি ডিটেকশন',
  readTime: 14,
  content: `
    <h3>১. LOF কেন দরকার?</h3>
    <p>
      Isolation Forest এবং One-Class SVM <strong>global anomalies</strong> ভালো ধরে — যেসব পয়েন্ট সমগ্র ডেটা থেকে অনেক দূরে। কিন্তু বাস্তবে এমন অ্যানোমালি থাকে যেগুলো global scale-এ স্বাভাবিক দেখায়, কিন্তু তার <em>আশেপাশের</em> পয়েন্টের তুলনায় অস্বাভাবিক।
    </p>
    <p>
      উদাহরণ: কল্পনা করুন দুটি শহর — ঢাকায় গড় বেতন ৫০,০০০ টাকা এবং গ্রামে ১৫,০০০ টাকা। গ্রামে কেউ ২৫,০০০ টাকা আয় করলে সেটি গ্রামীণ পরিপ্রেক্ষিতে অ্যানোমালি, কিন্তু ঢাকার মানুষের সাথে তুলনায় স্বাভাবিক। Global পদ্ধতি এটি ধরতে পারে না, কিন্তু <strong>LOF পারে</strong> কারণ এটি প্রতিটি পয়েন্টের স্থানীয় পরিবেশ বিশ্লেষণ করে।
    </p>

    <h3>২. মূল ধারণাসমূহ</h3>

    <h4>২.১ k-distance</h4>
    <p>
      একটি পয়েন্ট p-এর <strong>k-distance</strong> হলো তার k-তম নিকটতম প্রতিবেশীর সাথে দূরত্ব।
    </p>
    <p>
      উদাহরণ: k=3 হলে, p-এর ৩য় নিকটতম প্রতিবেশীর দূরত্বই k-distance।
    </p>

    <h4>২.২ Reachability Distance</h4>
    <p>
      পয়েন্ট p থেকে পয়েন্ট o-এর reachability distance:
    </p>
    <p>
      <strong>reach-dist_k(p, o) = max(k-distance(o), dist(p, o))</strong>
    </p>
    <p>
      এটি সরাসরি দূরত্বের চেয়ে বেশি স্থিতিশীল — ঘন এলাকায় পয়েন্টগুলোর মধ্যে দূরত্বের noise কমায়।
    </p>

    <h4>২.৩ Local Reachability Density (LRD)</h4>
    <p>
      একটি পয়েন্টের <strong>LRD</strong> হলো তার k-nearest neighbors-এর average reachability distance-এর inverse:
    </p>
    <p>
      <strong>LRD_k(p) = 1 / (Σ reach-dist_k(p, o) / k)</strong>
    </p>
    <p>
      LRD বড় = পয়েন্টটি ঘন এলাকায় আছে (প্রতিবেশীরা কাছাকাছি)।
      LRD ছোট = পয়েন্টটি বিরল এলাকায় আছে।
    </p>

    <h4>২.৪ LOF Score</h4>
    <p>
      অবশেষে, <strong>LOF score</strong> হলো পয়েন্টের LRD এবং তার প্রতিবেশীদের LRD-এর অনুপাত:
    </p>
    <p>
      <strong>LOF_k(p) = (Σ LRD_k(o) / LRD_k(p)) / k</strong>
    </p>
    <p>
      <strong>LOF ≈ 1:</strong> পয়েন্টটি তার প্রতিবেশীদের মতো ঘনত্বে আছে → স্বাভাবিক<br/>
      <strong>LOF &gt; 1:</strong> পয়েন্টটি তার প্রতিবেশীদের চেয়ে কম ঘন এলাকায় → সম্ভাব্য অ্যানোমালি<br/>
      <strong>LOF &gt;&gt; 1:</strong> শক্তিশালী অ্যানোমালি
    </p>

    <h3>৩. ধাপে ধাপে LOF বোঝা</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# সহজ উদাহরণ: দুটি ক্লাস্টার + একটি অ্যানোমালি
np.random.seed(42)
cluster1 = np.random.normal([2, 2], 0.3, (30, 2))
cluster2 = np.random.normal([7, 7], 0.3, (30, 2))
anomaly = np.array([[5, 5]])  # দুই ক্লাস্টারের মাঝে

X = np.vstack([cluster1, cluster2, anomaly])

# LOF প্রয়োগ করা
lof = LocalOutlierFactor(n_neighbors=5, contamination=0.01)
labels = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_  # LOF scores (বড় = অধিক অস্বাভাবিক)

print("LOF Scores (top 5 highest):")
top_idx = np.argsort(scores)[-5:]
for idx in top_idx:
    print(f"  Point {idx}: score={scores[idx]:.3f}, label={labels[idx]}")

# Visualization
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1],
                      c=scores, cmap='YlOrRd',
                      s=50 * scores,  # score বড় = বড় বৃত্ত
                      alpha=0.7)
plt.colorbar(scatter, label='LOF Score')
plt.scatter(anomaly[:, 0], anomaly[:, 1],
            c='red', s=200, marker='*', label='Known Anomaly', zorder=5)
plt.title('LOF Score Visualization')
plt.legend()
plt.show()</code></pre>

    <h3>৪. k-Neighbors প্যারামিটারের প্রভাব</h3>
    <p>
      <code>n_neighbors</code> (k) প্যারামিটার LOF-এর সবচেয়ে গুরুত্বপূর্ণ hyperparameter। এটি নির্ধারণ করে কতটি প্রতিবেশী দেখে local density হিসাব হবে।
    </p>
    <ul>
      <li><strong>ছোট k (e.g., 5):</strong> খুব local — ছোট ক্লাস্টারের মধ্যে অ্যানোমালি ধরতে ভালো</li>
      <li><strong>বড় k (e.g., 50):</strong> বেশি global দৃষ্টিভঙ্গি — noise-এ কম sensitive</li>
      <li><strong>সাধারণ নিয়ম:</strong> k = 20 থেকে 30 বেশিরভাগ ক্ষেত্রে ভালো কাজ করে</li>
    </ul>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
# দুই ধরনের অ্যানোমালি: global ও local
dense_cluster = np.random.normal([0, 0], 0.5, (100, 2))
sparse_cluster = np.random.normal([5, 5], 2.0, (50, 2))
global_anomaly = np.array([[10, 0]])     # global: সব থেকে দূরে
local_anomaly = np.array([[5.5, 5.5]])   # local: sparse cluster-এর মধ্যে কিন্তু আলাদা

X = np.vstack([dense_cluster, sparse_cluster, global_anomaly, local_anomaly])
true_anomalies = len(X) - 2  # শেষ দুটো আসল অ্যানোমালি

k_values = [5, 10, 20, 50]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, k in zip(axes, k_values):
    lof = LocalOutlierFactor(n_neighbors=k, contamination=0.03)
    labels = lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_

    colors = ['red' if l == -1 else 'steelblue' for l in labels]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=30, alpha=0.6)
    ax.scatter(global_anomaly[:, 0], global_anomaly[:, 1],
               marker='*', c='black', s=200, label='True Anomaly')
    ax.scatter(local_anomaly[:, 0], local_anomaly[:, 1],
               marker='*', c='black', s=200)
    ax.set_title(f'k = {k}')

plt.suptitle('LOF: Effect of n_neighbors', fontsize=13)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. Contextual Anomaly সনাক্তকরণ</h3>
    <p>
      LOF contextual anomaly সনাক্তকরণে বিশেষভাবে কার্যকর। নিচের উদাহরণে দেখি যেখানে একটি পয়েন্ট globally স্বাভাবিক কিন্তু locally অস্বাভাবিক।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

np.random.seed(42)
# তিনটি ভিন্ন ঘনত্বের ক্লাস্টার
cluster_dense = np.random.normal([0, 0], 0.3, (80, 2))
cluster_medium = np.random.normal([5, 0], 1.0, (80, 2))
cluster_sparse = np.random.normal([10, 0], 3.0, (80, 2))

# Contextual anomaly: medium cluster-এর মধ্যে থেকেও আলাদা
contextual = np.array([[5.5, 2.5]])

X = np.vstack([cluster_dense, cluster_medium, cluster_sparse, contextual])

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005)
lof_labels = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

# Isolation Forest (comparison)
iso = IsolationForest(contamination=0.005, random_state=42)
iso_labels = iso.fit_predict(X)

print("Contextual anomaly point (শেষ পয়েন্ট):")
print(f"  LOF Score: {lof_scores[-1]:.3f} → {'ANOMALY' if lof_labels[-1] == -1 else 'Normal'}")
print(f"  IsoForest: {'ANOMALY' if iso_labels[-1] == -1 else 'Normal'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, labels, title in zip(
    axes,
    [lof_labels, iso_labels],
    ['LOF', 'Isolation Forest']
):
    colors = ['red' if l == -1 else 'steelblue' for l in labels]
    ax.scatter(X[:-1, 0], X[:-1, 1], c=colors[:-1], s=20, alpha=0.5)
    c = 'red' if labels[-1] == -1 else 'green'
    ax.scatter(contextual[:, 0], contextual[:, 1],
               c=c, s=150, marker='*', zorder=5,
               label=f"Contextual: {'ANOMALY' if labels[-1]==-1 else 'Normal'}")
    ax.set_title(title)
    ax.legend()

plt.suptitle('Contextual Anomaly: LOF vs Isolation Forest')
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. sklearn LocalOutlierFactor: বিস্তারিত</h3>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

np.random.seed(42)

# ক্রেডিট কার্ড-এর মতো ডেটা
n_normal = 1000
normal = pd.DataFrame({
    'amount': np.random.lognormal(3, 0.8, n_normal),
    'hour': np.random.randint(8, 22, n_normal).astype(float),
    'distance_from_home': np.random.exponential(5, n_normal),
    'velocity': np.random.normal(2, 1, n_normal).clip(0)
})

n_fraud = 30
fraud = pd.DataFrame({
    'amount': np.random.lognormal(6, 0.5, n_fraud),  # অনেক বড় পরিমাণ
    'hour': np.random.choice([2, 3, 4], n_fraud).astype(float),  # রাতে
    'distance_from_home': np.random.exponential(50, n_fraud),  # দূরে
    'velocity': np.random.normal(100, 20, n_fraud).clip(0)  # অস্বাভাবিক বেগ
})

y_true = np.array([1]*n_normal + [-1]*n_fraud)
X = pd.concat([normal, fraud]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LOF মডেল
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.03,  # মোটামুটি fraud rate
    algorithm='ball_tree',  # দ্রুত nearest neighbor search
    metric='minkowski'
)

y_pred = lof.fit_predict(X_scaled)
scores = -lof.negative_outlier_factor_

print("LOF Performance on Fraud Detection:")
print(classification_report(y_true, y_pred, target_names=['Fraud (-1)', 'Normal (1)']))
print(f"Top 5 highest LOF scores (সম্ভাব্য fraud):")
top5 = np.argsort(scores)[-5:]
for i in top5:
    print(f"  Index {i}: LOF={scores[i]:.2f}, True={'Fraud' if y_true[i]==-1 else 'Normal'}")</code></pre>

    <h3>৭. LOF Score Threshold নির্বাচন</h3>
    <p>
      কোনো supervised label ছাড়াই সঠিক threshold নির্বাচন চ্যালেঞ্জিং। কিছু কৌশল:
    </p>
    <pre><code>import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.vstack([
    np.random.normal(0, 1, (300, 2)),
    np.random.uniform(-8, 8, (15, 2))
])

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
lof.fit_predict(X)
scores = -lof.negative_outlier_factor_

# পদ্ধতি ১: Elbow/Knee Method
sorted_scores = np.sort(scores)[::-1]
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(sorted_scores)), sorted_scores)
plt.axhline(y=1.5, color='red', linestyle='--', label='threshold=1.5')
plt.axhline(y=2.0, color='orange', linestyle='--', label='threshold=2.0')
plt.xlabel('Rank')
plt.ylabel('LOF Score')
plt.title('Score Distribution (sorted)')
plt.legend()

# পদ্ধতি ২: Percentile-based
plt.subplot(1, 2, 2)
thresholds = [1.2, 1.5, 2.0, 2.5, 3.0]
n_detected = [(scores > t).sum() for t in thresholds]
plt.bar(range(len(thresholds)), n_detected,
        tick_label=[str(t) for t in thresholds])
plt.xlabel('LOF Threshold')
plt.ylabel('Detected Anomalies')
plt.title('Threshold vs Detected Anomalies')
plt.tight_layout()
plt.show()

print("LOF Score Statistics:")
print(f"  Mean: {scores.mean():.3f}")
print(f"  95th percentile: {np.percentile(scores, 95):.3f}")
print(f"  99th percentile: {np.percentile(scores, 99):.3f}")</code></pre>

    <h3>৮. LOF-এর সুবিধা ও সীমাবদ্ধতা</h3>

    <h4>সুবিধা:</h4>
    <ul>
      <li>Local density পার্থক্য সনাক্ত করতে পারে — অন্য পদ্ধতির চেয়ে বেশি সূক্ষ্ম</li>
      <li>কোনো distribution assumption নেই</li>
      <li>Mixed-density ডেটায় (বিভিন্ন ঘনত্বের ক্লাস্টার) বিশেষভাবে ভালো</li>
      <li>LOF score ব্যাখ্যাযোগ্য (1-এর কাছাকাছি = স্বাভাবিক)</li>
    </ul>

    <h4>সীমাবদ্ধতা:</h4>
    <ul>
      <li>Quadratic time complexity O(n²) — বড় ডেটাসেটে ধীর</li>
      <li>High-dimensional ডেটায় nearest neighbor কম কার্যকর (curse of dimensionality)</li>
      <li>নতুন ডেটা predict করতে পুরো training dataset দরকার</li>
      <li>k (n_neighbors) সঠিকভাবে বেছে না নিলে ফলাফল ভুল হতে পারে</li>
    </ul>

    <h3>৯. সারসংক্ষেপ: কোন পদ্ধতি কখন?</h3>
    <p>
      এই সিরিজে আমরা চারটি পদ্ধতি দেখেছি:
    </p>
    <ul>
      <li><strong>Statistical (Z-score, IQR):</strong> সহজ, interpretable, ছোট ডেটায় ভালো</li>
      <li><strong>Isolation Forest:</strong> দ্রুত, scalable, high-dimensional global anomaly</li>
      <li><strong>One-Class SVM:</strong> Complex boundary, ছোট-মাঝারি ডেটায় precise</li>
      <li><strong>LOF:</strong> Local density comparison, mixed-density ডেটায় সেরা</li>
    </ul>
    <p>
      পরবর্তী পর্বে আমরা এই সব পদ্ধতি একত্রে একটি বাস্তব <strong>ক্রেডিট কার্ড জালিয়াতি সনাক্তকরণ প্রজেক্টে</strong> প্রয়োগ করব।
    </p>
  `
};
