export const knn_1_math = {
  title: "KNN: গণিত ও স্বজ্ঞাত বোঝাপড়া",
  description: "K-নিকটতম প্রতিবেশী অ্যালগরিদমের মূল গণিত, দূরত্ব মেট্রিক্স, K-এর প্রভাব এবং মাত্রার অভিশাপ সম্পর্কে বিস্তারিত আলোচনা।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 10,
  slug: "knn-math-intuition",
  content: `
    <h3>১. KNN-এর মূল স্বজ্ঞা</h3>
    <p>
      K-নিকটতম প্রতিবেশী (KNN) অ্যালগরিদম মেশিন লার্নিংয়ের সবচেয়ে সরল কিন্তু শক্তিশালী অ্যালগরিদমগুলোর একটি।
      এর মূল ধারণা অত্যন্ত সহজ: একটি নতুন data point-কে classify করতে হলে, তার K সংখ্যক নিকটতম প্রতিবেশীর
      class দেখো এবং majority vote অনুযায়ী সিদ্ধান্ত নাও।
    </p>
    <p>
      উদাহরণ হিসেবে চিন্তা করো: তুমি একটি নতুন শহরে গিয়েছ এবং জানতে চাইছ কোনো রেস্তোরাঁ ভালো কিনা।
      তুমি সেই রেস্তোরাঁর ৫ জন নিকটতম প্রতিবেশীকে জিজ্ঞেস করলে। যদি ৪ জন বলে ভালো এবং ১ জন বলে
      খারাপ, তাহলে তুমি সিদ্ধান্ত নেবে রেস্তোরাঁটি ভালো। এটাই KNN-এর মূল নীতি।
    </p>
    <p>
      KNN একটি <strong>lazy learner</strong> — এটি training-এর সময় কোনো model তৈরি করে না।
      বরং সমস্ত training data মেমোরিতে রেখে দেয় এবং prediction-এর সময় সেগুলো ব্যবহার করে।
    </p>

    <h3>২. দূরত্ব মেট্রিক্স: কাছাকাছি কে?</h3>
    <p>
      KNN-এ "নিকটতম" নির্ধারণ করতে দূরত্ব মেট্রিক্স ব্যবহার করা হয়। তিনটি প্রধান মেট্রিক্স হলো:
    </p>

    <p><strong>Euclidean Distance (সরলরেখা দূরত্ব):</strong></p>
    <p>
      দুটি point x এবং y এর মধ্যে Euclidean দূরত্ব হলো:
      d(x, y) = sqrt(sum((xi - yi)^2)) যেখানে i = 1 থেকে n পর্যন্ত।
      এটি সবচেয়ে পরিচিত দূরত্ব — সরলরেখায় দুটি বিন্দুর মধ্যে দূরত্ব।
    </p>

    <p><strong>Manhattan Distance (শহুরে দূরত্ব):</strong></p>
    <p>
      d(x, y) = sum(|xi - yi|) যেখানে i = 1 থেকে n পর্যন্ত।
      এটি grid-এ চলার মতো — উত্তর-দক্ষিণ এবং পূর্ব-পশ্চিম দিকে মোট কতটুকু যেতে হবে।
      Manhattan distance outlier-দের প্রতি কম সংবেদনশীল।
    </p>

    <p><strong>Minkowski Distance (সাধারণ রূপ):</strong></p>
    <p>
      d(x, y) = (sum(|xi - yi|^p))^(1/p) যেখানে p একটি প্যারামিটার।
      p=1 হলে Manhattan, p=2 হলে Euclidean, p=∞ হলে Chebyshev distance পাওয়া যায়।
    </p>

    <table>
      <thead>
        <tr>
          <th>মেট্রিক্স</th>
          <th>সূত্র</th>
          <th>sklearn প্যারামিটার</th>
          <th>সুবিধা</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Euclidean</td>
          <td>sqrt(Σ(xi-yi)²)</td>
          <td>metric='euclidean'</td>
          <td>সাধারণ ব্যবহারের জন্য উপযুক্ত</td>
        </tr>
        <tr>
          <td>Manhattan</td>
          <td>Σ|xi-yi|</td>
          <td>metric='manhattan'</td>
          <td>Outlier-এ robust</td>
        </tr>
        <tr>
          <td>Minkowski</td>
          <td>(Σ|xi-yi|^p)^(1/p)</td>
          <td>metric='minkowski'</td>
          <td>সাধারণ রূপ, p adjust করা যায়</td>
        </tr>
        <tr>
          <td>Chebyshev</td>
          <td>max(|xi-yi|)</td>
          <td>metric='chebyshev'</td>
          <td>Chess-board দূরত্ব</td>
        </tr>
      </tbody>
    </table>

    <h3>৩. Python দিয়ে দূরত্ব হিসাব</h3>
    <p>চলো numpy ব্যবহার করে manually দূরত্ব হিসাব করি এবং বুঝি কিভাবে KNN কাজ করে:</p>
    <pre><code>import numpy as np
import pandas as pd
from collections import Counter

# Training data তৈরি করা
X_train = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])
y_train = np.array([0, 0, 1, 1, 0, 1])

# নতুন point যেটাকে classify করতে হবে
new_point = np.array([3, 4])

# Euclidean distance হিসাব
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Manhattan distance হিসাব
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

# Minkowski distance হিসাব
def minkowski_distance(p1, p2, p=3):
    return np.sum(np.abs(p1 - p2) ** p) ** (1/p)

# সব training point-এর দূরত্ব বের করা
distances = []
for i, point in enumerate(X_train):
    euclid = euclidean_distance(new_point, point)
    manhattan = manhattan_distance(new_point, point)
    distances.append({
        'index': i,
        'label': y_train[i],
        'euclidean': round(euclid, 3),
        'manhattan': round(manhattan, 3)
    })

df = pd.DataFrame(distances)
print("সব point-এর দূরত্ব:")
print(df.sort_values('euclidean'))
print()

# Manual KNN prediction (K=3)
K = 3
sorted_distances = sorted(distances, key=lambda x: x['euclidean'])
k_nearest = sorted_distances[:K]

print(f"K={K} নিকটতম প্রতিবেশী:")
for neighbor in k_nearest:
    print(f"  Index {neighbor['index']}: Label={neighbor['label']}, Distance={neighbor['euclidean']}")

# Majority vote
labels = [n['label'] for n in k_nearest]
vote_counts = Counter(labels)
predicted_class = vote_counts.most_common(1)[0][0]
print(f"\nPredicted Class: {predicted_class}")
print(f"Vote breakdown: {dict(vote_counts)}")</code></pre>

    <h3>৪. K-এর প্রভাব: Overfitting vs Underfitting</h3>
    <p>
      K-এর মান নির্বাচন KNN-এ সবচেয়ে গুরুত্বপূর্ণ সিদ্ধান্ত। এটি সরাসরি model-এর
      complexity নিয়ন্ত্রণ করে:
    </p>
    <p>
      <strong>K=1 (Overfit):</strong> প্রতিটি training point নিজেই একটি boundary তৈরি করে।
      Training accuracy 100% কিন্তু test data-তে ভালো করে না। Noise-এর প্রতি অত্যন্ত সংবেদনশীল।
    </p>
    <p>
      <strong>K বড় (Underfit):</strong> K যখন n (সব training data) এর কাছাকাছি হয়, তখন
      model সবসময় majority class predict করে। সরল হয়ে যায়, জটিল pattern ধরতে পারে না।
    </p>
    <p>
      <strong>Optimal K:</strong> সাধারণত K = sqrt(n) থেকে শুরু করা হয়। n = training data-র
      সংখ্যা। তবে cross-validation দিয়ে সঠিক K বের করা সবচেয়ে ভালো পদ্ধতি।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset তৈরি
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sqrt(n) rule
n = len(X_train)
suggested_k = int(np.sqrt(n))
print(f"Training data: {n} samples")
print(f"Suggested K (sqrt rule): {suggested_k}")

# বিভিন্ন K-এর জন্য accuracy
train_scores = []
test_scores = []
k_values = range(1, 50)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(k_values, test_scores, label='Test Accuracy', color='red', linewidth=2)
plt.axvline(x=suggested_k, color='green', linestyle='--', label=f'sqrt(n)={suggested_k}')
plt.xlabel('K মান')
plt.ylabel('Accuracy')
plt.title('K-এর মান vs Accuracy (Bias-Variance Tradeoff)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_k = k_values[np.argmax(test_scores)]
print(f"\nBest K: {best_k}, Test Accuracy: {max(test_scores):.4f}")</code></pre>

    <h3>৫. মাত্রার অভিশাপ (Curse of Dimensionality)</h3>
    <p>
      KNN-এর সবচেয়ে বড় সমস্যা হলো <strong>curse of dimensionality</strong>। যখন features-এর
      সংখ্যা (dimensions) বাড়তে থাকে, তখন দূরত্ব হিসাবের ধারণাটি অর্থহীন হয়ে পড়ে।
    </p>
    <p>
      High-dimensional space-এ সব data point একে অপর থেকে প্রায় সমান দূরে থাকে।
      ফলে "নিকটতম প্রতিবেশী" খোঁজার ধারণাটি কার্যকর থাকে না।
    </p>
    <p>
      উদাহরণ: ১০০ টি data point একটি unit square-এ সমানভাবে ছড়িয়ে আছে। ১০% data cover করতে
      ১D-তে 0.1 দৈর্ঘ্যের segment দরকার, ২D-তে sqrt(0.1) ≈ 0.316 দৈর্ঘ্যের square দরকার,
      কিন্তু ১০D-তে 0.1^(1/10) ≈ 0.794 দৈর্ঘ্যের hypercube দরকার।
      অর্থাৎ high dimension-এ কাছের প্রতিবেশী খুঁজতে অনেক বড় এলাকা scan করতে হয়।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# Curse of dimensionality দেখানো
np.random.seed(42)
n_samples = 1000
dimensions = [1, 2, 5, 10, 20, 50, 100]
avg_distances = []

for d in dimensions:
    # d-dimensional unit hypercube-এ random points
    points = np.random.uniform(0, 1, (n_samples, d))
    origin = np.zeros(d)

    # সব points-এর origin থেকে দূরত্ব
    dists = np.sqrt(np.sum(points**2, axis=1))
    avg_distances.append(np.mean(dists))

# Nearest neighbor distance ratio
ratios = []
for d in dimensions:
    points = np.random.uniform(0, 1, (200, d))

    dist_matrix = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :])**2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    min_dists = dist_matrix.min(axis=1)
    max_dists = dist_matrix.max(axis=1)
    ratio = np.mean(min_dists / max_dists)  # কাছের vs দূরের অনুপাত
    ratios.append(ratio)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(dimensions, avg_distances, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Dimension সংখ্যা')
axes[0].set_ylabel('গড় দূরত্ব (origin থেকে)')
axes[0].set_title('Dimension বাড়লে দূরত্ব বাড়ে')
axes[0].grid(True, alpha=0.3)

axes[1].plot(dimensions, ratios, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Dimension সংখ্যা')
axes[1].set_ylabel('min/max দূরত্বের অনুপাত')
axes[1].set_title('High Dimension-এ সব দূরত্ব সমান হয়')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nDimension | গড় দূরত্ব | min/max অনুপাত")
print("-" * 40)
for d, avg, ratio in zip(dimensions, avg_distances, ratios):
    print(f"{d:9d} | {avg:.4f}    | {ratio:.4f}")</code></pre>

    <h3>৬. Time ও Space Complexity</h3>
    <p>
      KNN-এর computational cost বোঝা খুব জরুরি:
    </p>
    <p>
      <strong>Training complexity:</strong> O(1) — KNN কোনো training করে না, শুধু data store করে।
    </p>
    <p>
      <strong>Prediction complexity:</strong> O(n × d) — প্রতিটি prediction-এর জন্য সব n টি
      training point-এর সাথে d-dimensional দূরত্ব হিসাব করতে হয়।
    </p>
    <p>
      <strong>Space complexity:</strong> O(n × d) — সমস্ত training data মেমোরিতে রাখতে হয়।
    </p>
    <p>
      এই কারণে KNN বড় dataset-এ ধীর। Ball Tree এবং KD Tree ব্যবহার করে prediction complexity
      O(d × log n) পর্যন্ত কমানো সম্ভব (low-dimensional data-তে)।
    </p>

    <table>
      <thead>
        <tr>
          <th>অপারেশন</th>
          <th>Brute Force</th>
          <th>KD Tree</th>
          <th>Ball Tree</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Training</td>
          <td>O(n·d)</td>
          <td>O(n·d·log n)</td>
          <td>O(n·d·log n)</td>
        </tr>
        <tr>
          <td>Prediction</td>
          <td>O(n·d)</td>
          <td>O(d·log n)</td>
          <td>O(d·log n)</td>
        </tr>
        <tr>
          <td>Space</td>
          <td>O(n·d)</td>
          <td>O(n·d)</td>
          <td>O(n·d)</td>
        </tr>
        <tr>
          <td>High Dim</td>
          <td>ঠিক আছে</td>
          <td>d&gt;20 হলে খারাপ</td>
          <td>High dim-এ ভালো</td>
        </tr>
      </tbody>
    </table>
  `,
};
