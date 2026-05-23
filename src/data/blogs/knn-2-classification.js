export const knn_2_classification = {
  title: "KNN Classification: Python দিয়ে সম্পূর্ণ বাস্তবায়ন",
  description: "Sklearn দিয়ে KNN Classification-এর সম্পূর্ণ উদাহরণ — data preprocessing, normalization, model training, evaluation এবং decision boundary visualization।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 11,
  slug: "knn-classification-python",
  content: `
    <h3>১. কেন Normalization জরুরি?</h3>
    <p>
      KNN সম্পূর্ণরূপে দূরত্বের উপর নির্ভরশীল। যদি features-এর scale ভিন্ন হয়,
      তাহলে বড় scale-এর feature দূরত্ব হিসাবে অস্বাভাবিক প্রভাব ফেলে।
    </p>
    <p>
      উদাহরণ: বয়স (0-100) এবং বেতন (10000-100000) এর মধ্যে দূরত্ব হিসাব করলে বেতনের
      পার্থক্য বয়সের পার্থক্যকে সম্পূর্ণ ছাপিয়ে যাবে। StandardScaler দিয়ে সব features-কে
      mean=0, std=1 করে নিলে এই সমস্যা দূর হয়।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Scale difference সমস্যার উদাহরণ
data = {
    'বয়স': [25, 30, 35, 28],
    'বেতন': [30000, 50000, 80000, 45000],
    'অভিজ্ঞতা': [2, 5, 10, 4]
}
df = pd.DataFrame(data)

# Scaling ছাড়া দূরত্ব
p1 = df.iloc[0].values  # [25, 30000, 2]
p2 = df.iloc[1].values  # [30, 50000, 5]
dist_unscaled = np.sqrt(np.sum((p1 - p2)**2))
print(f"Scaling ছাড়া দূরত্ব: {dist_unscaled:.2f}")
# বেতনের পার্থক্য (20000) সব কিছু dominate করে

# Scaling-এর পরে দূরত্ব
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
p1_s = df_scaled[0]
p2_s = df_scaled[1]
dist_scaled = np.sqrt(np.sum((p1_s - p2_s)**2))
print(f"Scaling-এর পরে দূরত্ব: {dist_scaled:.2f}")
# এখন সব features সমান গুরুত্ব পায়

print("\nScaled data:")
print(pd.DataFrame(df_scaled, columns=df.columns).round(3))</code></pre>

    <h3>২. Iris Dataset দিয়ে সম্পূর্ণ Pipeline</h3>
    <p>
      Iris dataset-এ ৩ ধরনের ফুল (Setosa, Versicolor, Virginica) এবং ৪টি feature আছে।
      চলো KNN দিয়ে এটি classify করি:
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

# ১. Data Load করা
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

# ২. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ৩. Normalization (IMPORTANT: fit শুধু train data-তে, transform উভয়তে)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # শুধু transform, fit নয়

# ৪. Model Training
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train_scaled, y_train)

# ৫. Prediction
y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)

# ৬. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ৭. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('KNN Confusion Matrix (Iris Dataset)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()</code></pre>

    <h3>৩. Uniform vs Distance Weights</h3>
    <p>
      KNN-এ দুটি weighting strategy আছে:
    </p>
    <p>
      <strong>weights='uniform':</strong> সব K প্রতিবেশীর vote সমান। দূরে থাকা প্রতিবেশী
      এবং কাছে থাকা প্রতিবেশী একই গুরুত্ব পায়।
    </p>
    <p>
      <strong>weights='distance':</strong> কাছের প্রতিবেশীর vote বেশি গুরুত্বপূর্ণ।
      weight = 1/distance। এটি সাধারণত বেশি accurate কারণ কাছের প্রতিবেশী বেশি relevant।
    </p>
    <pre><code>from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# দুটি weighting strategy তুলনা
results = {}

for weights in ['uniform', 'distance']:
    knn = KNeighborsClassifier(n_neighbors=5, weights=weights)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[weights] = scores
    print(f"weights='{weights}':")
    print(f"  CV Scores: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    print()

# কোন strategy ভালো?
best = max(results.items(), key=lambda x: x[1].mean())
print(f"ভালো strategy: weights='{best[0]}' (mean={best[1].mean():.4f})")</code></pre>

    <h3>৪. Decision Boundary Visualization</h3>
    <p>
      Decision boundary দেখতে ২টি feature ব্যবহার করে plot করা যায়।
      এতে বোঝা যায় K-এর মান boundary-কে কতটা smooth বা jagged করে:
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# শুধু ২টি feature নেওয়া (visualization-এর জন্য)
iris = load_iris()
X = iris.data[:, :2]  # sepal length ও sepal width
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ৩টি K-এর জন্য decision boundary
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
k_values = [1, 5, 15]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cmap_light = ListedColormap(['#FFB3B3', '#B3F0EB', '#B3DCF0'])

for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)

    # Mesh grid তৈরি
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

plt.suptitle('KNN Decision Boundary: K-এর প্রভাব', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
# K=1: জটিল, jagged boundary (overfit)
# K=5: মসৃণ boundary (ভালো balance)
# K=15: অনেক smooth, কিছু detail হারিয়ে যায়</code></pre>

    <h3>৫. K vs Accuracy Curve এবং সেরা K নির্বাচন</h3>
    <p>
      বিভিন্ন K-এর জন্য cross-validation accuracy plot করে optimal K বের করা যায়:
    </p>
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

# Plot with error bands
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, 'b-o', linewidth=2, markersize=6, label='Mean CV Accuracy')
plt.fill_between(k_range,
                 cv_scores - cv_stds,
                 cv_scores + cv_stds,
                 alpha=0.2, color='blue', label='±1 STD')

best_k = list(k_range)[np.argmax(cv_scores)]
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
            label=f'Best K={best_k}')
plt.xlabel('K (প্রতিবেশী সংখ্যা)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K vs Cross-Validation Accuracy (10-fold CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.show()

print(f"Best K: {best_k}")
print(f"Best CV Accuracy: {cv_scores[best_k-1]:.4f}")

# Final model with best K
final_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
final_knn.fit(X_train_scaled, y_train)
test_accuracy = accuracy_score(y_test, final_knn.predict(X_test_scaled))
print(f"Final Test Accuracy: {test_accuracy:.4f}")</code></pre>

    <h3>৬. Model Performance Summary</h3>
    <table>
      <thead>
        <tr>
          <th>Configuration</th>
          <th>K</th>
          <th>Weights</th>
          <th>CV Accuracy</th>
          <th>উপযুক্ত কখন</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Default</td>
          <td>5</td>
          <td>uniform</td>
          <td>~96%</td>
          <td>সাধারণ শুরুর জন্য</td>
        </tr>
        <tr>
          <td>Distance Weighted</td>
          <td>5</td>
          <td>distance</td>
          <td>~97%</td>
          <td>Dense clusters থাকলে</td>
        </tr>
        <tr>
          <td>Large K</td>
          <td>15</td>
          <td>uniform</td>
          <td>~95%</td>
          <td>Noisy data-তে</td>
        </tr>
        <tr>
          <td>Tuned</td>
          <td>CV-optimized</td>
          <td>distance</td>
          <td>~98%</td>
          <td>Best performance চাইলে</td>
        </tr>
      </tbody>
    </table>
  `,
};
