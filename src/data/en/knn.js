export const knnEn = [
  {
    title: "KNN: Math & Intuition",
    description: "Core mathematics of K-Nearest Neighbors — distance metrics, the effect of K on overfitting and underfitting, curse of dimensionality, and time/space complexity.",
    date: "২৩ মে, ২০২৬",
    category: "K-Nearest Neighbors",
    readTime: 10,
    slug: "knn-math-intuition",
    content: `
      <h3>1. The Core Intuition of KNN</h3>
      <p>
        K-Nearest Neighbors (KNN) is one of the simplest yet most powerful algorithms in machine learning.
        The core idea is straightforward: to classify a new data point, look at its K nearest neighbors
        in the training set and assign the class by majority vote.
      </p>
      <p>
        Think of it this way: you move to a new city and want to know if a restaurant is good.
        You ask the 5 nearest neighbors. If 4 say it's good and 1 says it's bad, you conclude it's good.
        That is exactly KNN's principle.
      </p>
      <p>
        KNN is a <strong>lazy learner</strong> — it builds no model during training.
        It simply stores all training data in memory and uses it at prediction time.
      </p>

      <h3>2. Distance Metrics: Who Is Nearest?</h3>
      <p>
        KNN uses distance metrics to define "nearest." The three main metrics are:
      </p>
      <p><strong>Euclidean Distance (straight-line):</strong></p>
      <p>
        d(x, y) = sqrt(sum((xi - yi)^2)) for i = 1 to n.
        This is the most familiar distance — the straight-line distance between two points.
      </p>
      <p><strong>Manhattan Distance (city-block):</strong></p>
      <p>
        d(x, y) = sum(|xi - yi|) for i = 1 to n.
        Like navigating a city grid — total blocks traveled north-south plus east-west.
        Manhattan distance is less sensitive to outliers.
      </p>
      <p><strong>Minkowski Distance (general form):</strong></p>
      <p>
        d(x, y) = (sum(|xi - yi|^p))^(1/p) where p is a parameter.
        p=1 gives Manhattan, p=2 gives Euclidean, p=infinity gives Chebyshev.
      </p>

      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Formula</th>
            <th>sklearn Parameter</th>
            <th>Advantage</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Euclidean</td>
            <td>sqrt(Σ(xi-yi)²)</td>
            <td>metric='euclidean'</td>
            <td>Good for general use</td>
          </tr>
          <tr>
            <td>Manhattan</td>
            <td>Σ|xi-yi|</td>
            <td>metric='manhattan'</td>
            <td>Robust to outliers</td>
          </tr>
          <tr>
            <td>Minkowski</td>
            <td>(Σ|xi-yi|^p)^(1/p)</td>
            <td>metric='minkowski'</td>
            <td>General form, adjustable p</td>
          </tr>
          <tr>
            <td>Chebyshev</td>
            <td>max(|xi-yi|)</td>
            <td>metric='chebyshev'</td>
            <td>Chess-board distance</td>
          </tr>
        </tbody>
      </table>

      <h3>3. Computing Distance in Python</h3>
      <p>Let's manually compute distances with numpy to understand how KNN works:</p>
      <pre><code>import numpy as np
import pandas as pd
from collections import Counter

# Training data
X_train = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])
y_train = np.array([0, 0, 1, 1, 0, 1])

new_point = np.array([3, 4])

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def minkowski_distance(p1, p2, p=3):
    return np.sum(np.abs(p1 - p2) ** p) ** (1/p)

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
print("Distances to all points:")
print(df.sort_values('euclidean'))

# Manual KNN prediction (K=3)
K = 3
sorted_distances = sorted(distances, key=lambda x: x['euclidean'])
k_nearest = sorted_distances[:K]

print(f"\nK={K} Nearest Neighbors:")
for neighbor in k_nearest:
    print(f"  Index {neighbor['index']}: Label={neighbor['label']}, Distance={neighbor['euclidean']}")

labels = [n['label'] for n in k_nearest]
vote_counts = Counter(labels)
predicted_class = vote_counts.most_common(1)[0][0]
print(f"\nPredicted Class: {predicted_class}")
print(f"Vote breakdown: {dict(vote_counts)}")</code></pre>

      <h3>4. Effect of K: Overfitting vs Underfitting</h3>
      <p>
        Choosing K is the most important decision in KNN. It directly controls model complexity:
      </p>
      <p>
        <strong>K=1 (Overfit):</strong> Every training point forms its own boundary.
        Training accuracy is 100% but the model fails on unseen data. Extremely sensitive to noise.
      </p>
      <p>
        <strong>Large K (Underfit):</strong> When K approaches n (all training data), the model
        always predicts the majority class. Too simple to capture complex patterns.
      </p>
      <p>
        <strong>Optimal K:</strong> A common starting point is K = sqrt(n), where n is the
        number of training samples. Cross-validation is the most reliable way to find the best K.
      </p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n = len(X_train)
suggested_k = int(np.sqrt(n))
print(f"Training samples: {n}, Suggested K (sqrt rule): {suggested_k}")

train_scores = []
test_scores = []
k_values = range(1, 50)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Train Accuracy', color='blue', linewidth=2)
plt.plot(k_values, test_scores, label='Test Accuracy', color='red', linewidth=2)
plt.axvline(x=suggested_k, color='green', linestyle='--', label=f'sqrt(n)={suggested_k}')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.title('K vs Accuracy (Bias-Variance Tradeoff)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_k = list(k_values)[np.argmax(test_scores)]
print(f"\nBest K: {best_k}, Test Accuracy: {max(test_scores):.4f}")</code></pre>

      <h3>5. Curse of Dimensionality</h3>
      <p>
        KNN's biggest challenge is the <strong>curse of dimensionality</strong>. As the number
        of features grows, all points become approximately equidistant from each other.
        The concept of "nearest neighbor" loses meaning in high-dimensional space.
      </p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
dimensions = [1, 2, 5, 10, 20, 50, 100]

ratios = []
for d in dimensions:
    points = np.random.uniform(0, 1, (200, d))
    dist_matrix = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :])**2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists = dist_matrix.min(axis=1)
    max_dists = dist_matrix.max(axis=1)
    ratio = np.mean(min_dists / max_dists)
    ratios.append(ratio)

plt.figure(figsize=(8, 5))
plt.plot(dimensions, ratios, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Dimensions')
plt.ylabel('min/max distance ratio')
plt.title('Curse of Dimensionality: All Distances Become Equal')
plt.grid(True, alpha=0.3)
plt.show()

print("Dimension | min/max ratio")
print("-" * 25)
for d, ratio in zip(dimensions, ratios):
    print(f"{d:9d} | {ratio:.4f}")</code></pre>

      <h3>6. Time and Space Complexity</h3>
      <p>
        <strong>Training complexity:</strong> O(1) — KNN does no training, just stores data.
      </p>
      <p>
        <strong>Prediction complexity:</strong> O(n × d) — each prediction computes distance
        to all n training points across d dimensions.
      </p>
      <p>
        <strong>Space complexity:</strong> O(n × d) — all training data must be kept in memory.
      </p>

      <table>
        <thead>
          <tr>
            <th>Operation</th>
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
            <td>OK</td>
            <td>Poor above d=20</td>
            <td>Better for high dim</td>
          </tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "KNN Classification with Python: Complete Implementation",
    description: "Full KNN classification walkthrough using sklearn — data normalization with StandardScaler, Iris dataset, accuracy metrics, decision boundary visualization, and K selection.",
    date: "২৩ মে, ২০২৬",
    category: "K-Nearest Neighbors",
    readTime: 11,
    slug: "knn-classification-python",
    content: `
      <h3>1. Why Normalization Is Essential</h3>
      <p>
        KNN relies entirely on distance calculations. If features have different scales,
        large-scale features will dominate the distance computation and bias the results.
      </p>
      <p>
        Example: age (0–100) vs. salary (10,000–100,000). A salary difference of 20,000
        completely overwhelms an age difference of 5. StandardScaler transforms all features
        to mean=0 and std=1, giving every feature equal influence.
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'age': [25, 30, 35, 28],
    'salary': [30000, 50000, 80000, 45000],
    'experience': [2, 5, 10, 4]
}
df = pd.DataFrame(data)

p1 = df.iloc[0].values
p2 = df.iloc[1].values
dist_unscaled = np.sqrt(np.sum((p1 - p2)**2))
print(f"Distance without scaling: {dist_unscaled:.2f}")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
p1_s = df_scaled[0]
p2_s = df_scaled[1]
dist_scaled = np.sqrt(np.sum((p1_s - p2_s)**2))
print(f"Distance after scaling: {dist_scaled:.2f}")

print("\nScaled data:")
print(pd.DataFrame(df_scaled, columns=df.columns).round(3))</code></pre>

      <h3>2. Complete Pipeline with the Iris Dataset</h3>
      <p>
        The Iris dataset contains 3 flower classes (Setosa, Versicolor, Virginica) and
        4 features. Let's build a full KNN classifier:
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset info:")
print(f"  Features: {feature_names}")
print(f"  Classes: {target_names}")
print(f"  Shape: {X.shape}")
print(f"  Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# IMPORTANT: fit only on training data, transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('KNN Confusion Matrix (Iris Dataset)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()</code></pre>

      <h3>3. Uniform vs Distance Weights</h3>
      <p>
        KNN supports two weighting strategies:
      </p>
      <p>
        <strong>weights='uniform':</strong> All K neighbors cast equal votes regardless of distance.
      </p>
      <p>
        <strong>weights='distance':</strong> Closer neighbors vote with more weight (weight = 1/distance).
        This is generally more accurate because nearby neighbors are more relevant.
      </p>
      <pre><code>from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

results = {}
for weights in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors=5, weights=weights)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[weights] = scores
    print(f"weights='{weights}':")
    print(f"  CV Scores: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    print()

best = max(results.items(), key=lambda x: x[1].mean())
print(f"Better strategy: weights='{best[0]}' (mean={best[1].mean():.4f})")</code></pre>

      <h3>4. Decision Boundary Visualization</h3>
      <p>
        Using 2 features, we can visualize how K affects the decision boundary — small K produces
        jagged boundaries (overfit), large K produces smooth but less detailed boundaries:
      </p>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data[:, :2]  # sepal length and sepal width only
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
k_values = [1, 5, 15]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cmap_light = ListedColormap(['#FFB3B3', '#B3F0EB', '#B3DCF0'])

for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)

    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[idx].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    for class_idx, color in enumerate(colors):
        mask = y == class_idx
        axes[idx].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                         c=color, label=iris.target_names[class_idx],
                         edgecolors='black', linewidth=0.5, s=50)

    axes[idx].set_title(f'K={k} Decision Boundary')
    axes[idx].set_xlabel('Sepal Length (scaled)')
    axes[idx].set_ylabel('Sepal Width (scaled)')
    axes[idx].legend(fontsize=8)

plt.suptitle('KNN Decision Boundary: Effect of K', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()</code></pre>

      <h3>5. K vs Accuracy Curve</h3>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 31)
cv_scores = []
cv_stds = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_stds.append(scores.std())

cv_scores = np.array(cv_scores)
cv_stds = np.array(cv_stds)

plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, 'b-o', linewidth=2, markersize=6, label='Mean CV Accuracy')
plt.fill_between(k_range, cv_scores - cv_stds, cv_scores + cv_stds,
                 alpha=0.2, color='blue', label='±1 STD')

best_k = list(k_range)[np.argmax(cv_scores)]
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k}')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K vs Cross-Validation Accuracy (10-fold CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
final_knn.fit(X_train_scaled, y_train)
test_accuracy = accuracy_score(y_test, final_knn.predict(X_test_scaled))
print(f"Best K: {best_k}, Test Accuracy: {test_accuracy:.4f}")</code></pre>

      <h3>6. Model Performance Summary</h3>
      <table>
        <thead>
          <tr>
            <th>Configuration</th>
            <th>K</th>
            <th>Weights</th>
            <th>CV Accuracy</th>
            <th>Best Used When</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Default</td>
            <td>5</td>
            <td>uniform</td>
            <td>~96%</td>
            <td>Starting point</td>
          </tr>
          <tr>
            <td>Distance Weighted</td>
            <td>5</td>
            <td>distance</td>
            <td>~97%</td>
            <td>Dense clusters present</td>
          </tr>
          <tr>
            <td>Large K</td>
            <td>15</td>
            <td>uniform</td>
            <td>~95%</td>
            <td>Noisy data</td>
          </tr>
          <tr>
            <td>Tuned</td>
            <td>CV-optimized</td>
            <td>distance</td>
            <td>~98%</td>
            <td>Best performance needed</td>
          </tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "KNN Regression: Predicting Continuous Outputs",
    description: "KNeighborsRegressor for continuous value prediction — California housing dataset, MSE, R² evaluation, weighted averaging, and comparison with Linear Regression.",
    date: "২৩ মে, ২০২৬",
    category: "K-Nearest Neighbors",
    readTime: 9,
    slug: "knn-regression",
    content: `
      <h3>1. How KNN Regression Works</h3>
      <p>
        KNN Regression uses the same algorithm as KNN Classification, but with a different output.
        Instead of taking a majority class vote from K neighbors, regression computes the
        <strong>mean (or weighted mean)</strong> of the K nearest neighbors' target values.
      </p>
      <p>
        Prediction formula: y-hat = (1/K) × sum(yi), where yi are the target values of the K nearest neighbors.
      </p>
      <p>
        Example: predict a house price. KNN finds the 5 nearest neighbor houses with prices
        [200k, 250k, 220k, 280k, 230k]. Prediction = mean = 236k.
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# House size (sqft) vs price (thousands)
X_train = np.array([[500], [750], [1000], [1200], [1500], [1800], [2000]])
y_train = np.array([150, 200, 280, 330, 420, 550, 650])

new_house = np.array([[1100]])

knn_reg = KNeighborsRegressor(n_neighbors=3, weights='uniform')
knn_reg.fit(X_train, y_train)

pred_price = knn_reg.predict(new_house)
print(f"House size: {new_house[0][0]} sqft")
print(f"Predicted price: \${pred_price[0]:.2f}k")

distances, indices = knn_reg.kneighbors(new_house)
print(f"\nK=3 Nearest Neighbors:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. size={X_train[idx][0]} sqft, price=\${y_train[idx]}k, dist={dist:.1f}")
print(f"\nMean price: \${np.mean(y_train[indices[0]]):.2f}k")</code></pre>

      <h3>2. California Housing Dataset Example</h3>
      <p>
        The California housing dataset contains housing features and predicts the median house value.
        It is a classic real-world regression benchmark:
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X = housing.data
y = housing.target

print("California Housing Dataset:")
print(f"  Features: {housing.feature_names}")
print(f"  Samples: {X.shape[0]}")
print(f"  Target range: \${y.min()*100000:.0f} - \${y.max()*100000:.0f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_reg = KNeighborsRegressor(
    n_neighbors=10,
    weights='distance',
    metric='euclidean',
    n_jobs=-1
)
knn_reg.fit(X_train_scaled, y_train)

y_pred = knn_reg.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance (K=10, weights='distance'):")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f} (\${rmse*100000:.0f})")
print(f"  MAE:  {mae:.4f} (\${mae*100000:.0f})")
print(f"  R2:   {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual House Value ($100k)')
plt.ylabel('Predicted House Value ($100k)')
plt.title(f'KNN Regression: Actual vs Predicted (R2={r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

      <h3>3. Weighted KNN Regression</h3>
      <p>
        Using <strong>weights='distance'</strong> makes closer neighbors contribute more to the prediction.
        This generally improves regression accuracy because nearby points are more relevant:
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

results = {}
for weights in ['uniform', 'distance']:
    for k in [5, 10, 15, 20]:
        knn = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[(k, weights)] = {'MSE': mse, 'R2': r2}

rows = []
for (k, w), metrics in results.items():
    rows.append({'K': k, 'Weights': w, 'MSE': round(metrics['MSE'], 4),
                 'R2': round(metrics['R2'], 4)})

results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
print("KNN Regression Results:")
print(results_df.to_string(index=False))</code></pre>

      <h3>4. Finding Optimal K for Regression</h3>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

k_range = range(1, 31)
mse_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    scores = cross_val_score(knn, X_train_scaled, y_train,
                             cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(-scores.mean())

best_k = list(k_range)[np.argmin(mse_scores)]

plt.figure(figsize=(10, 6))
plt.plot(k_range, mse_scores, 'b-o', linewidth=2, markersize=6)
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k}')
plt.xlabel('K value')
plt.ylabel('Mean Squared Error (CV)')
plt.title('K vs MSE: Finding Optimal K for KNN Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print(f"Best K: {best_k}, Best CV MSE: {mse_scores[best_k-1]:.4f}")</code></pre>

      <h3>5. KNN Regression vs Linear Regression</h3>
      <table>
        <thead>
          <tr>
            <th>Property</th>
            <th>KNN Regression</th>
            <th>Linear Regression</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Model Type</td>
            <td>Non-parametric</td>
            <td>Parametric</td>
          </tr>
          <tr>
            <td>Assumptions</td>
            <td>None</td>
            <td>Assumes linear relationship</td>
          </tr>
          <tr>
            <td>Training Speed</td>
            <td>O(1) — instant</td>
            <td>O(n·d²) — usually fast</td>
          </tr>
          <tr>
            <td>Prediction Speed</td>
            <td>O(n·d) — slow</td>
            <td>O(d) — instant</td>
          </tr>
          <tr>
            <td>Memory</td>
            <td>O(n·d) — stores all data</td>
            <td>O(d) — just coefficients</td>
          </tr>
          <tr>
            <td>Non-linearity</td>
            <td>Handles naturally</td>
            <td>Cannot (without polynomial features)</td>
          </tr>
          <tr>
            <td>Interpretability</td>
            <td>Low</td>
            <td>High — clear coefficients</td>
          </tr>
          <tr>
            <td>Outlier Sensitivity</td>
            <td>Low with distance weights</td>
            <td>High sensitivity</td>
          </tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "KNN Distance Metrics: Which One to Use and When",
    description: "Deep dive into Euclidean, Manhattan, Minkowski, Chebyshev, Hamming, and Cosine distance metrics — how each works, when to use them, and performance comparison in sklearn.",
    date: "২৩ মে, ২০২৬",
    category: "K-Nearest Neighbors",
    readTime: 9,
    slug: "knn-distance-metrics",
    content: `
      <h3>1. Why Distance Metrics Matter</h3>
      <p>
        KNN's performance depends entirely on how we measure "distance." Using the wrong metric
        causes the model to find completely wrong neighbors and produce unreliable predictions.
      </p>
      <p>
        The right metric depends on data type:
        continuous numerical data, categorical data, text data, and binary data
        each call for different metrics.
      </p>

      <h3>2. The Main Distance Metrics</h3>
      <p><strong>Euclidean Distance:</strong></p>
      <p>
        Straight-line distance between two points.
        Formula: d = sqrt(sum((xi - yi)^2)).
        Most common choice for continuous numerical data. Sensitive to outliers because squaring amplifies large differences.
      </p>
      <p><strong>Manhattan Distance:</strong></p>
      <p>
        Grid-based distance (like navigating city streets).
        Formula: d = sum(|xi - yi|).
        Robust to outliers because no squaring. Often works better than Euclidean for high-dimensional data.
      </p>
      <p><strong>Minkowski Distance:</strong></p>
      <p>
        General form of Euclidean and Manhattan.
        Formula: d = (sum(|xi - yi|^p))^(1/p).
        p=1 is Manhattan, p=2 is Euclidean, p→infinity is Chebyshev.
        Control via the p parameter in sklearn.
      </p>
      <p><strong>Chebyshev Distance:</strong></p>
      <p>
        Maximum difference across any single dimension.
        Formula: d = max(|xi - yi|).
        Analogous to a king's move in chess. Use when uniform movement across all dimensions is needed.
      </p>
      <p><strong>Hamming Distance:</strong></p>
      <p>
        For categorical or binary data.
        Formula: fraction of positions where two strings differ.
        Used in text comparison, DNA sequence matching.
      </p>
      <p><strong>Cosine Distance:</strong></p>
      <p>
        Measures the angle between two vectors, not their magnitude.
        Cosine similarity = (A dot B) / (||A|| times ||B||).
        Cosine distance = 1 - Cosine similarity.
        Most popular for text/NLP, recommendation systems, and sparse data.
      </p>

      <h3>3. Computing All Metrics in Python</h3>
      <pre><code>import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev
from scipy.spatial.distance import hamming, cosine
import pandas as pd

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([2, 4, 1, 3, 5])

print("Numerical Distance Metrics:")
print(f"  Euclidean:        {euclidean(x1, x2):.4f}")
print(f"  Manhattan:        {cityblock(x1, x2):.4f}")
print(f"  Minkowski (p=3):  {minkowski(x1, x2, p=3):.4f}")
print(f"  Chebyshev:        {chebyshev(x1, x2):.4f}")
print(f"  Cosine:           {cosine(x1, x2):.4f}")

str1 = np.array([1, 0, 1, 1, 0, 1])
str2 = np.array([1, 1, 1, 0, 0, 1])
print(f"\nHamming Distance (binary): {hamming(str1, str2):.4f}")
print(f"  (Differing positions: {np.sum(str1 != str2)}/{len(str1)})")

doc1 = np.array([2, 1, 0, 3, 1])
doc2 = np.array([1, 0, 1, 2, 2])
cos_sim = np.dot(doc1, doc2) / (np.linalg.norm(doc1) * np.linalg.norm(doc2))
print(f"\nCosine Similarity (text): {cos_sim:.4f}")
print(f"Cosine Distance: {1-cos_sim:.4f}")

print("\nMinkowski: effect of p:")
for p in [1, 1.5, 2, 3, 5, 10]:
    dist = minkowski(x1, x2, p=p)
    label = f"p={p}" + (" (Manhattan)" if p==1 else " (Euclidean)" if p==2 else "")
    print(f"  {label}: {dist:.4f}")</code></pre>

      <h3>4. Comparing Metrics on the Same Dataset</h3>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

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
    knn.fit(X_train_s, y_train)
    test_acc = knn.score(X_test_s, y_test)
    results.append({
        'Metric': metric,
        'CV Mean': round(cv_scores.mean(), 4),
        'CV Std': round(cv_scores.std(), 4),
        'Test Accuracy': round(test_acc, 4)
    })

df_results = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print("Distance Metrics Comparison on Wine Dataset (K=5):")
print(df_results.to_string(index=False))</code></pre>

      <h3>5. Cosine Distance for Text Classification</h3>
      <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

texts = [
    "machine learning python algorithm",
    "deep learning neural network",
    "python programming software",
    "neural network deep learning model",
    "algorithm data structure computer",
    "machine learning data science",
]
labels = [0, 1, 0, 1, 0, 0]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('knn', KNeighborsClassifier(n_neighbors=3, metric='cosine'))
])

pipeline.fit(texts, labels)

test_texts = [
    "python deep learning algorithm",
    "neural network machine",
]
predictions = pipeline.predict(test_texts)
for text, pred in zip(test_texts, predictions):
    label = "Deep Learning" if pred == 1 else "Programming"
    print(f"Text: '{text}' -> Class: {label}")</code></pre>

      <h3>6. Distance Metrics Comparison Table</h3>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Formula</th>
            <th>Best Use Case</th>
            <th>sklearn Parameter</th>
            <th>Limitation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Euclidean</td>
            <td>sqrt(sum(xi-yi)^2)</td>
            <td>General continuous data</td>
            <td>metric='euclidean'</td>
            <td>Outlier-sensitive, scale-dependent</td>
          </tr>
          <tr>
            <td>Manhattan</td>
            <td>sum(|xi-yi|)</td>
            <td>High-dim, with outliers</td>
            <td>metric='manhattan'</td>
            <td>Ignores diagonal direction</td>
          </tr>
          <tr>
            <td>Minkowski</td>
            <td>(sum(|xi-yi|^p))^(1/p)</td>
            <td>General, tunable</td>
            <td>metric='minkowski', p=?</td>
            <td>p requires tuning</td>
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
            <td>Ignores magnitude</td>
          </tr>
          <tr>
            <td>Cosine</td>
            <td>1 - (A·B)/(||A||·||B||)</td>
            <td>Text, NLP, sparse data</td>
            <td>metric='cosine'</td>
            <td>Ignores magnitude entirely</td>
          </tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "KNN Tuning and Cross-Validation: Building an Optimal Model",
    description: "Complete guide to KNN hyperparameter tuning — GridSearchCV, Ball Tree vs KD Tree vs Brute Force, Pipeline with StandardScaler, learning curves, and saving production-ready models.",
    date: "২৩ মে, ২০২৬",
    category: "K-Nearest Neighbors",
    readTime: 11,
    slug: "knn-tuning-cross-validation",
    content: `
      <h3>1. Finding the Optimal K with Cross-Validation</h3>
      <p>
        K is the most important hyperparameter in KNN. Cross-validation evaluates model
        performance across multiple data splits to find the K that generalizes best and avoids overfitting.
      </p>
      <p>
        In K-fold cross-validation, the training data is divided into K folds. Each fold serves
        as the validation set once while the rest are used for training.
        The average performance across all folds is used for evaluation.
      </p>
      <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
k_range = range(1, 51)
cv_means = []
cv_stds = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_s, y_train, cv=kf, scoring='accuracy')
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

cv_means = np.array(cv_means)
cv_stds = np.array(cv_stds)
best_k = list(k_range)[np.argmax(cv_means)]

plt.figure(figsize=(12, 6))
plt.plot(k_range, cv_means, 'b-o', linewidth=2, markersize=5, label='Mean CV Accuracy')
plt.fill_between(k_range, cv_means - 2*cv_stds, cv_means + 2*cv_stds,
                 alpha=0.15, color='blue', label='95% CI')
plt.axvline(x=best_k, color='red', linestyle='--', lw=2, label=f'Best K={best_k}')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('10-Fold CV Accuracy')
plt.title('Cross-Validation for Optimal K Selection (Breast Cancer)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Best K: {best_k}, Best CV Accuracy: {cv_means[best_k-1]:.4f}")</code></pre>

      <h3>2. GridSearchCV for Full Hyperparameter Tuning</h3>
      <p>
        GridSearchCV exhaustively tests all combinations of specified hyperparameter values
        and selects the best configuration. For KNN, n_neighbors, weights, and metric are
        the most impactful parameters:
      </p>
      <pre><code>import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import time

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [2, 3]
}

knn = KNeighborsClassifier(n_jobs=-1)
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

start = time.time()
grid_search.fit(X_train_s, y_train)
elapsed = time.time() - start

print(f"GridSearch time: {elapsed:.2f} seconds")
print(f"Total combinations tested: {len(grid_search.cv_results_['params'])}")
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test_s, y_test)
print(f"Test Accuracy: {test_score:.4f}")

results_df = pd.DataFrame(grid_search.cv_results_)
top_configs = results_df.nlargest(10, 'mean_test_score')[
    ['param_n_neighbors', 'param_weights', 'param_metric',
     'mean_test_score', 'std_test_score']
]
top_configs.columns = ['K', 'Weights', 'Metric', 'Mean CV', 'Std CV']
print("\nTop 10 Configurations:")
print(top_configs.to_string(index=False))</code></pre>

      <h3>3. Ball Tree vs KD Tree vs Brute Force</h3>
      <p>
        The <strong>algorithm</strong> parameter in sklearn controls how nearest neighbors are found.
        Choosing the right algorithm can dramatically improve prediction speed:
      </p>
      <p>
        <strong>Brute Force:</strong> Computes distance to every training point. Best for small datasets or high dimensions.
      </p>
      <p>
        <strong>KD Tree:</strong> Organizes data into a k-dimensional tree. Fast for low-dimensional data (d ≤ 20) with large n.
      </p>
      <p>
        <strong>Ball Tree:</strong> Partitions data into hierarchical balls. Better than KD Tree for high dimensions and non-Euclidean metrics.
      </p>
      <pre><code>import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Algorithm Comparison: Training and Prediction Time")
print("=" * 60)

for n_samples in [1000, 5000, 10000]:
    X, y = make_classification(n_samples=n_samples, n_features=20,
                               n_redundant=5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\nn_samples={n_samples}:")
    for algo in ['brute', 'kd_tree', 'ball_tree']:
        knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)

        t0 = time.time()
        knn.fit(X_tr_s, y_tr)
        train_time = time.time() - t0

        t0 = time.time()
        acc = knn.score(X_te_s, y_te)
        pred_time = time.time() - t0

        print(f"  {algo:10s}: train={train_time*1000:.2f}ms, "
              f"predict={pred_time*1000:.2f}ms, acc={acc:.4f}")</code></pre>

      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Build Time</th>
            <th>Query Time</th>
            <th>Best Use Case</th>
            <th>Limitation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Brute Force</td>
            <td>O(n·d)</td>
            <td>O(n·d)</td>
            <td>Small dataset, high dim</td>
            <td>Slow for large n</td>
          </tr>
          <tr>
            <td>KD Tree</td>
            <td>O(n·d·log n)</td>
            <td>O(d·log n)</td>
            <td>d ≤ 20, large n</td>
            <td>Degrades in high dim</td>
          </tr>
          <tr>
            <td>Ball Tree</td>
            <td>O(n·d·log n)</td>
            <td>O(d·log n)</td>
            <td>High dim, non-Euclidean</td>
            <td>Higher build time</td>
          </tr>
        </tbody>
      </table>

      <h3>4. Pipeline: StandardScaler + KNN</h3>
      <p>
        A Pipeline chains preprocessing and modeling steps, preventing data leakage
        and keeping code clean and reproducible:
      </p>
      <pre><code>import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Pipeline parameter naming: step_name__param_name
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
    'knn__algorithm': ['auto']
}

grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pass raw (unscaled) data — Pipeline handles scaling with no leakage
grid_search.fit(X_train, y_train)

print("Pipeline GridSearch Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

best_pipeline = grid_search.best_estimator_
joblib.dump(best_pipeline, 'knn_pipeline.pkl')
print("\nModel saved to knn_pipeline.pkl")

loaded_model = joblib.load('knn_pipeline.pkl')
predictions = loaded_model.predict(X_test[:5])
print(f"Sample predictions: {predictions}")</code></pre>

      <h3>5. Learning Curve Analysis</h3>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
])

train_sizes, train_scores, val_scores = learning_curve(
    best_pipeline, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'b-o', label='Training Accuracy', lw=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, val_mean, 'r-o', label='Validation Accuracy', lw=2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.15, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('KNN Learning Curve (Breast Cancer Dataset)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

      <h3>6. KNN Tuning Best Practices</h3>
      <table>
        <thead>
          <tr>
            <th>Hyperparameter</th>
            <th>Default</th>
            <th>Tuning Range</th>
            <th>Recommendation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>n_neighbors</td>
            <td>5</td>
            <td>1-50 (or sqrt(n))</td>
            <td>Use odd numbers for binary classification</td>
          </tr>
          <tr>
            <td>weights</td>
            <td>uniform</td>
            <td>uniform, distance</td>
            <td>Distance usually performs better</td>
          </tr>
          <tr>
            <td>metric</td>
            <td>minkowski</td>
            <td>euclidean, manhattan, cosine</td>
            <td>Choose based on data type</td>
          </tr>
          <tr>
            <td>algorithm</td>
            <td>auto</td>
            <td>auto, ball_tree, kd_tree, brute</td>
            <td>auto is fine in most cases</td>
          </tr>
          <tr>
            <td>n_jobs</td>
            <td>1</td>
            <td>-1 (all cores)</td>
            <td>Set -1 for large datasets</td>
          </tr>
          <tr>
            <td>leaf_size</td>
            <td>30</td>
            <td>10-50</td>
            <td>Balance tree build vs query speed</td>
          </tr>
        </tbody>
      </table>
    `,
  },
];
