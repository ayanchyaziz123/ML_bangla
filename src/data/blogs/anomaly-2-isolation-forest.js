export const anomaly_2_isolation_forest = {
  slug: 'anomaly-2-isolation-forest',
  title: 'Isolation Forest: গাছ দিয়ে অ্যানোমালি খোঁজা',
  description: 'Isolation Forest-এর অন্তর্নিহিত ধারণা, র‍্যান্ডম পার্টিশনিং, অ্যানোমালি স্কোর ও পাথ লেন্থ, contamination প্যারামিটার এবং sklearn দিয়ে সম্পূর্ণ বাস্তবায়ন।',
  date: 'মে ২০২৬',
  category: 'অ্যানোমালি ডিটেকশন',
  readTime: 13,
  content: `
    <h3>১. Isolation Forest-এর মূল ধারণা</h3>
    <p>
      Isolation Forest ২০০৮ সালে Liu et al. প্রস্তাব করেন। এটির মূল অন্তর্দৃষ্টিটি অত্যন্ত সুন্দর এবং সরল:
    </p>
    <p>
      <em>"অ্যানোমালি পয়েন্টগুলো স্বাভাবিক পয়েন্টের চেয়ে isolate করা অনেক সহজ।"</em>
    </p>
    <p>
      ভাবুন একটি ঘন বনের মধ্যে কিছু গাছ একসাথে দলবদ্ধভাবে দাঁড়িয়ে আছে (স্বাভাবিক ডেটা), আর কিছু গাছ একা দূরে দাঁড়িয়ে আছে (অ্যানোমালি)। যদি আপনি এলোমেলোভাবে বেড়া দিয়ে এলাকা ভাগ করতে থাকেন, একা দাঁড়ানো গাছটি অনেক কম বেড়া দিয়ে আলাদা হয়ে যাবে — কারণ তার আশেপাশে কেউ নেই।
    </p>

    <h3>২. অ্যালগরিদম: কীভাবে কাজ করে?</h3>

    <h4>ধাপ ১: Random Partitioning</h4>
    <p>
      প্রতিটি Isolation Tree তৈরি হয় এইভাবে:
    </p>
    <ol>
      <li>ডেটা থেকে একটি random feature (column) বেছে নাও</li>
      <li>সেই feature-এর min ও max মানের মধ্যে একটি random split point বেছে নাও</li>
      <li>ডেটাকে দুই ভাগে ভাগ করো</li>
      <li>যতক্ষণ প্রতিটি ভাগে মাত্র একটি পয়েন্ট না থাকে বা নির্দিষ্ট গভীরতায় না পৌঁছায়, ততক্ষণ পুনরাবৃত্তি করো</li>
    </ol>

    <h4>ধাপ ২: Path Length হিসাব</h4>
    <p>
      একটি পয়েন্ট isolate হতে কতটি splits দরকার — এটিই path length। অ্যানোমালি পয়েন্টের path length ছোট (কম splits দরকার), স্বাভাবিক পয়েন্টের path length বড়।
    </p>

    <h4>ধাপ ৩: Anomaly Score</h4>
    <p>
      অনেক tree-এর গড় path length থেকে anomaly score তৈরি হয়:
    </p>
    <p>
      <strong>s(x, n) = 2^(-E(h(x)) / c(n))</strong>
    </p>
    <p>
      যেখানে E(h(x)) = গড় path length, c(n) = normalization factor। Score ≈ 1 মানে অ্যানোমালি, Score ≈ 0.5 মানে স্বাভাবিক।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# ডেটা তৈরি করা
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=300, centers=1,
                          cluster_std=0.5, random_state=42)
X_anomaly = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_anomaly])

# Isolation Forest প্রয়োগ
iso_forest = IsolationForest(
    n_estimators=100,      # গাছের সংখ্যা
    contamination=0.06,    # অ্যানোমালির অনুমানিত অনুপাত
    random_state=42,
    max_samples='auto'
)

labels = iso_forest.fit_predict(X)
scores = iso_forest.decision_function(X)

# ফলাফল বিশ্লেষণ
n_anomalies = (labels == -1).sum()
print(f"মোট পয়েন্ট: {len(X)}")
print(f"সনাক্ত অ্যানোমালি: {n_anomalies}")
print(f"Score Range: [{scores.min():.3f}, {scores.max():.3f}]")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red' if l == -1 else 'blue' for l in labels]
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=30)
plt.title('Isolation Forest: Detected Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(handles=[
    plt.scatter([], [], c='blue', label='Normal'),
    plt.scatter([], [], c='red', label='Anomaly')
])

plt.subplot(1, 2, 2)
plt.hist(scores, bins=30, edgecolor='black', color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
plt.title('Anomaly Scores Distribution')
plt.xlabel('Decision Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()</code></pre>

    <h3>৩. Contamination প্যারামিটার</h3>
    <p>
      <code>contamination</code> হলো Isolation Forest-এর সবচেয়ে গুরুত্বপূর্ণ প্যারামিটার। এটি ডেটাসেটে অ্যানোমালির প্রত্যাশিত অনুপাত নির্ধারণ করে।
    </p>
    <p>
      উদাহরণ: contamination=0.05 মানে মোট ডেটার ৫% অ্যানোমালি বলে আমরা ধরে নিচ্ছি।
    </p>
    <pre><code>import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

np.random.seed(42)
X_normal = np.random.normal(0, 1, (200, 2))
X_anomaly = np.random.uniform(-5, 5, (10, 2))
X = np.vstack([X_normal, X_anomaly])

# বিভিন্ন contamination মান পরীক্ষা করা
contamination_values = [0.02, 0.05, 0.10, 0.15]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, cont in zip(axes, contamination_values):
    iso = IsolationForest(contamination=cont, random_state=42)
    preds = iso.fit_predict(X)
    n_out = (preds == -1).sum()

    colors = ['red' if p == -1 else 'steelblue' for p in preds]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=20)
    ax.set_title(f'contamination={cont}\\n({n_out} anomalies)')
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')

plt.suptitle('Effect of Contamination Parameter', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. n_estimators: কতটি গাছ দরকার?</h3>
    <p>
      Isolation Forest একাধিক decision tree দিয়ে তৈরি (ensemble method)। <code>n_estimators</code> প্যারামিটার গাছের সংখ্যা নির্ধারণ করে।
    </p>
    <ul>
      <li>বেশি গাছ = বেশি stable score, কিন্তু বেশি computational cost</li>
      <li>সাধারণত <strong>100 গাছ</strong> যথেষ্ট; 500-এর বেশি সাধারণত প্রয়োজন নেই</li>
    </ul>
    <pre><code>from sklearn.ensemble import IsolationForest
import numpy as np
import time

np.random.seed(42)
X = np.random.normal(0, 1, (1000, 10))

for n_est in [10, 50, 100, 200, 500]:
    start = time.time()
    iso = IsolationForest(n_estimators=n_est, random_state=42)
    iso.fit(X)
    elapsed = time.time() - start
    scores = iso.decision_function(X)
    print(f"n_estimators={n_est:4d}: time={elapsed:.3f}s, score_std={scores.std():.4f}")</code></pre>

    <h3>৫. Decision Boundary Visualization</h3>
    <p>
      Isolation Forest-এর decision boundary 2D ডেটায় দেখলে আমরা বুঝতে পারি কীভাবে এটি স্বাভাবিক এলাকা থেকে অ্যানোমালি আলাদা করে।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 0.5, (150, 2)),
    np.random.normal(3, 0.5, (150, 2)),
    np.random.uniform(-4, 6, (15, 2))
])

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X)

# Decision boundary grid তৈরি করা
xx, yy = np.meshgrid(np.linspace(-5, 7, 200), np.linspace(-5, 7, 200))
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Anomaly Score')

preds = iso.predict(X)
plt.scatter(X[preds == 1, 0], X[preds == 1, 1],
            c='blue', s=30, alpha=0.7, label='Normal')
plt.scatter(X[preds == -1, 0], X[preds == -1, 1],
            c='red', s=60, alpha=0.9, label='Anomaly', marker='x', linewidths=2)

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.title('Isolation Forest Decision Boundary')
plt.legend()
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. Real-World উদাহরণ: সার্ভার মেট্রিক্স</h3>
    <p>
      একটি বাস্তব ব্যবহারের ক্ষেত্রে, ধরুন আমাদের কাছে সার্ভারের CPU, memory এবং response time ডেটা আছে। Isolation Forest দিয়ে অস্বাভাবিক সার্ভার অবস্থা সনাক্ত করা যায়।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_normal = 500

# স্বাভাবিক সার্ভার মেট্রিক্স
normal_data = pd.DataFrame({
    'cpu_usage': np.random.normal(40, 10, n_normal),
    'memory_usage': np.random.normal(60, 8, n_normal),
    'response_time_ms': np.random.normal(200, 30, n_normal),
    'error_rate': np.random.normal(0.01, 0.005, n_normal)
})

# অস্বাভাবিক মেট্রিক্স (অ্যানোমালি)
anomaly_data = pd.DataFrame({
    'cpu_usage': np.random.normal(90, 5, 20),
    'memory_usage': np.random.normal(95, 3, 20),
    'response_time_ms': np.random.normal(2000, 200, 20),
    'error_rate': np.random.normal(0.3, 0.1, 20)
})

df = pd.concat([normal_data, anomaly_data], ignore_index=True)

# স্কেলিং ও মডেল
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

iso = IsolationForest(n_estimators=100, contamination=0.04, random_state=42)
df['anomaly'] = iso.fit_predict(X_scaled)
df['score'] = iso.decision_function(X_scaled)

# ফলাফল
print("অ্যানোমালি পয়েন্টের গড় মেট্রিক্স:")
print(df[df['anomaly'] == -1][['cpu_usage', 'memory_usage', 'response_time_ms']].mean())
print("\\nস্বাভাবিক পয়েন্টের গড় মেট্রিক্স:")
print(df[df['anomaly'] == 1][['cpu_usage', 'memory_usage', 'response_time_ms']].mean())</code></pre>

    <h3>৭. Isolation Forest-এর সুবিধা ও সীমাবদ্ধতা</h3>

    <h4>সুবিধা:</h4>
    <ul>
      <li>উচ্চ-মাত্রার ডেটায় ভালো কাজ করে (curse of dimensionality কম প্রভাবিত করে)</li>
      <li>Linear time complexity — বড় ডেটাসেটে দ্রুত</li>
      <li>কোনো distribution assumption নেই</li>
      <li>Training-এর সময় label দরকার নেই (unsupervised)</li>
    </ul>

    <h4>সীমাবদ্ধতা:</h4>
    <ul>
      <li>Categorical feature সরাসরি সামলাতে পারে না</li>
      <li>খুব উচ্চ-মাত্রার sparse ডেটায় কম কার্যকর</li>
      <li>সঠিক contamination মান জানা না থাকলে ফলাফল পরিবর্তিত হয়</li>
    </ul>

    <h3>৮. পরবর্তী পদক্ষেপ</h3>
    <p>
      Isolation Forest দারুণভাবে global anomalies সনাক্ত করতে পারে, কিন্তু local anomalies-এর ক্ষেত্রে (যেমন ঘন ক্লাস্টারের কাছাকাছি অ্যানোমালি) এটি কম কার্যকর। এর জন্য পরবর্তী পর্বে আমরা <strong>Local Outlier Factor (LOF)</strong> আলোচনা করব, যা প্রতিটি পয়েন্টের স্থানীয় ঘনত্ব বিশ্লেষণ করে।
    </p>
  `
};
