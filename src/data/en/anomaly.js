export const anomalyEn = [
  {
    slug: 'anomaly-1-statistics',
    title: 'Anomaly Detection: Statistical Methods',
    description: 'Z-score, IQR, Grubbs test, and statistical thresholds for outlier detection',
    category: 'Anomaly Detection',
    content: `
<h3>Types of Anomalies</h3>
<ul>
<li><strong>Point anomaly:</strong> Single data point is unusual (e.g., a single high transaction)</li>
<li><strong>Contextual anomaly:</strong> Normal globally but unusual in context (warm day in winter)</li>
<li><strong>Collective anomaly:</strong> A sequence of points is anomalous together</li>
</ul>

<pre><code class="language-python">import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 990), [5, -5, 6, -6, 8, -8, 7, 9, -7, 10]])

# Method 1: Z-score
z_scores = np.abs(stats.zscore(data))
z_outliers = np.where(z_scores > 3)[0]
print(f"Z-score outliers: {len(z_outliers)} at indices {z_outliers[:5]}")

# Method 2: IQR
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
iqr_outliers = np.where((data < lower) | (data > upper))[0]
print(f"IQR outliers: {len(iqr_outliers)}")

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(data, 'b.', alpha=0.3, label='Normal')
plt.plot(z_outliers, data[z_outliers], 'r*', markersize=12, label='Z-score outliers')
plt.axhline(3, color='r', linestyle='--', alpha=0.5)
plt.axhline(-3, color='r', linestyle='--', alpha=0.5)
plt.legend(); plt.title('Statistical Anomaly Detection')
plt.show()
</code></pre>
`
  },
  {
    slug: 'anomaly-2-isolation-forest',
    title: 'Isolation Forest: Tree-Based Anomaly Detection',
    description: 'How Isolation Forest isolates anomalies through random partitioning and path length',
    category: 'Anomaly Detection',
    content: `
<h3>Isolation Forest Intuition</h3>
<p>Anomalies are <em>few</em> and <em>different</em> — they require fewer random splits to isolate. The anomaly score is based on the average path length across all trees: shorter path = more anomalous.</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X_normal = np.random.normal(0, 0.5, (300, 2))
X_anomaly = np.random.uniform(-4, 4, (15, 2))
X = np.vstack([X_normal, X_anomaly])
y_true = np.array([0]*300 + [1]*15)

clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
clf.fit(X)
y_pred = (clf.predict(X) == -1).astype(int)
scores = -clf.score_samples(X)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.4)
plt.scatter(*X_normal.T, c='blue', s=10, label='Normal', alpha=0.5)
plt.scatter(*X_anomaly.T, c='red', s=80, marker='*', label='Anomaly', zorder=5)
plt.colorbar(label='Anomaly Score')
plt.legend(); plt.title('Isolation Forest Decision Boundary')
plt.show()
</code></pre>
`
  },
  {
    slug: 'anomaly-3-one-class-svm',
    title: 'One-Class SVM: Boundary of Normalcy',
    description: 'Learning a tight decision boundary around normal data with One-Class SVM',
    category: 'Anomaly Detection',
    content: `
<h3>One-Class SVM</h3>
<p>One-Class SVM learns a hypersphere (or decision boundary) that encloses the normal training data. Points outside the boundary are classified as anomalies. The <code>nu</code> parameter controls the expected fraction of outliers.</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_train = np.random.normal(0, 1, (200, 2))
X_normal_test = np.random.normal(0, 1, (50, 2))
X_anomaly = np.random.uniform(-5, 5, (20, 2))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_normal_s = scaler.transform(X_normal_test)
X_anomaly_s = scaler.transform(X_anomaly)

oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
oc_svm.fit(X_train_s)

pred_normal = oc_svm.predict(X_normal_s)
pred_anomaly = oc_svm.predict(X_anomaly_s)

print(f"Normal correctly classified: {(pred_normal == 1).mean():.2%}")
print(f"Anomalies detected: {(pred_anomaly == -1).mean():.2%}")

# Compare nu values
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, nu in zip(axes, [0.05, 0.1, 0.2]):
    clf = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    clf.fit(X_train_s)
    xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.5)
    ax.scatter(*X_train_s.T, c='b', s=5, alpha=0.3)
    ax.set_title(f'nu={nu}')
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'anomaly-4-lof',
    title: 'Local Outlier Factor (LOF)',
    description: 'Density-based anomaly detection comparing local reachability densities',
    category: 'Anomaly Detection',
    content: `
<h3>LOF Algorithm</h3>
<p>LOF compares the local density of a point to the density of its neighbors. Points in low-density regions relative to neighbors have high LOF scores (>1) and are considered anomalies.</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
X1 = np.random.normal([0, 0], 0.3, (200, 2))
X2 = np.random.normal([3, 3], 0.5, (100, 2))
X_outliers = np.array([[5, 0], [0, 5], [-3, -3], [7, 7]])
X = np.vstack([X1, X2, X_outliers])

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.04)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

# Visualize with score-scaled markers
plt.figure(figsize=(10, 7))
sizes = (lof_scores / lof_scores.max()) * 200
colors = np.where(y_pred == -1, 'red', 'blue')
plt.scatter(X[:, 0], X[:, 1], s=sizes, c=colors, alpha=0.6)
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
            s=200, facecolors='none', edgecolors='red', linewidths=2, label='Anomaly')
plt.legend(); plt.title('LOF: Larger markers = Higher anomaly score')
plt.show()

# LOF is better than Isolation Forest for local anomalies
print(f"Detected {(y_pred == -1).sum()} anomalies")
</code></pre>
`
  },
  {
    slug: 'anomaly-5-project',
    title: 'Anomaly Detection Project: Credit Card Fraud',
    description: 'Comparing Isolation Forest, LOF, and One-Class SVM on the credit card fraud dataset',
    category: 'Anomaly Detection',
    content: `
<h3>Credit Card Fraud Detection</h3>
<pre><code class="language-python">import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('creditcard.csv')
print(f"Fraud rate: {df['Class'].mean():.4%}")

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
X, y = df[features].values, df['Class'].values

contamination = y.mean()

# Model comparison
models = {
    'Isolation Forest': IsolationForest(contamination=contamination, n_estimators=200, random_state=42),
    'LOF': LocalOutlierFactor(contamination=contamination, n_neighbors=20),
}

for name, clf in models.items():
    if hasattr(clf, 'fit_predict'):
        y_pred = (clf.fit_predict(X) == -1).astype(int)
    else:
        clf.fit(X)
        y_pred = (clf.predict(X) == -1).astype(int)
    print(f"\\n{name}")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fraud']))

# Threshold optimization for Isolation Forest
iforest = IsolationForest(contamination=contamination, random_state=42)
iforest.fit(X)
scores = -iforest.score_samples(X)
auc = roc_auc_score(y, scores)
print(f"\\nIsolation Forest ROC-AUC: {auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y, scores)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_t = thresholds[f1.argmax()]
y_best = (scores >= best_t).astype(int)
print(f"Best threshold F1: {f1.max():.4f}")
print(classification_report(y, y_best, target_names=['Normal', 'Fraud']))
</code></pre>
`
  },
];
