export const knn_5_tuning = {
  title: "KNN Tuning ও Cross-Validation: Optimal Model তৈরি",
  description: "GridSearchCV দিয়ে KNN hyperparameter tuning, Ball Tree vs KD Tree, Pipeline তৈরি এবং production-ready KNN model-এর সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 11,
  slug: "knn-tuning-cross-validation",
  content: `
    <h3>১. Cross-Validation দিয়ে Optimal K খোঁজা</h3>
    <p>
      KNN-এর সবচেয়ে গুরুত্বপূর্ণ hyperparameter হলো K। Cross-validation ব্যবহার করে
      বিভিন্ন K-এর জন্য model-এর performance মূল্যায়ন করা যায় এবং overfitting এড়ানো যায়।
    </p>
    <p>
      K-fold cross-validation-এ training data কে K ভাগে ভাগ করা হয়। প্রতিটি ভাগ একবার
      validation set হিসেবে ব্যবহার হয় এবং বাকি ভাগগুলো training-এ ব্যবহার হয়।
      সব fold-এর average performance দিয়ে final evaluation করা হয়।
    </p>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Breast Cancer Dataset (binary classification)
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

# K-fold CV দিয়ে optimal K খোঁজা
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

# Plot with confidence interval
plt.figure(figsize=(12, 6))
plt.plot(k_range, cv_means, 'b-o', linewidth=2, markersize=5, label='Mean CV Accuracy')
plt.fill_between(k_range, cv_means - 2*cv_stds, cv_means + 2*cv_stds,
                 alpha=0.15, color='blue', label='95% CI')
plt.axvline(x=best_k, color='red', linestyle='--', lw=2, label=f'Best K={best_k}')
plt.axhline(y=cv_means[best_k-1], color='green', linestyle=':', lw=2,
            label=f'Best Accuracy={cv_means[best_k-1]:.4f}')
plt.xlabel('K (প্রতিবেশী সংখ্যা)')
plt.ylabel('10-Fold CV Accuracy')
plt.title('Cross-Validation দিয়ে Optimal K নির্বাচন (Breast Cancer)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 51, 5))
plt.tight_layout()
plt.show()

print(f"\nBest K: {best_k}")
print(f"Best CV Accuracy: {cv_means[best_k-1]:.4f}")</code></pre>

    <h3>২. GridSearchCV দিয়ে সম্পূর্ণ Hyperparameter Tuning</h3>
    <p>
      GridSearchCV একসাথে সব hyperparameter-এর সব সম্ভাব্য combination test করে
      সেরাটি বের করে। KNN-এর জন্য n_neighbors, weights এবং metric tune করা সবচেয়ে গুরুত্বপূর্ণ:
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time

# Parameter grid সংজ্ঞায়ন
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [2, 3]  # শুধু minkowski-এর জন্য প্রযোজ্য
}

# GridSearchCV setup
knn = KNeighborsClassifier(n_jobs=-1)  # parallel computation
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # সব CPU core ব্যবহার
    verbose=1,
    return_train_score=True
)

# Training (সময় পরিমাপ)
start = time.time()
grid_search.fit(X_train_s, y_train)
elapsed = time.time() - start

print(f"GridSearch সময়: {elapsed:.2f} seconds")
print(f"মোট combinations test: {len(grid_search.cv_results_['params'])}")
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Best model দিয়ে test
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test_s, y_test)
print(f"Test Accuracy: {test_score:.4f}")

# Top 10 configurations
results_df = pd.DataFrame(grid_search.cv_results_)
top_configs = results_df.nlargest(10, 'mean_test_score')[
    ['param_n_neighbors', 'param_weights', 'param_metric',
     'mean_test_score', 'std_test_score']
]
top_configs.columns = ['K', 'Weights', 'Metric', 'Mean CV', 'Std CV']
print("\nTop 10 Configurations:")
print(top_configs.to_string(index=False))</code></pre>

    <h3>৩. Ball Tree vs KD Tree vs Brute Force</h3>
    <p>
      sklearn-এ KNN-এর <strong>algorithm</strong> parameter দিয়ে nearest neighbor খোঁজার
      পদ্ধতি নির্বাচন করা যায়। সঠিক algorithm নির্বাচন prediction speed বহুগুণ বাড়াতে পারে:
    </p>
    <p>
      <strong>Brute Force:</strong> সব training point-এর সাথে দূরত্ব হিসাব করে।
      ছোট dataset বা high dimension-এ ব্যবহার করা উচিত।
    </p>
    <p>
      <strong>KD Tree:</strong> Data-কে k-dimensional tree structure-এ সাজায়।
      Low-dimensional data (d ≤ 20) এবং large dataset-এ দ্রুত কাজ করে।
    </p>
    <p>
      <strong>Ball Tree:</strong> Data-কে hierarchical balls-এ বিভক্ত করে।
      High-dimensional data এবং non-Euclidean metrics-এ KD Tree-এর চেয়ে ভালো।
    </p>
    <pre><code>import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# বিভিন্ন dataset size-এ algorithm তুলনা
print("Algorithm Comparison: Training ও Prediction Time")
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
          <th>সেরা ব্যবহার</th>
          <th>Limitation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Brute Force</td>
          <td>O(n·d)</td>
          <td>O(n·d)</td>
          <td>Small dataset, high dim</td>
          <td>Large n-এ ধীর</td>
        </tr>
        <tr>
          <td>KD Tree</td>
          <td>O(n·d·log n)</td>
          <td>O(d·log n)</td>
          <td>d ≤ 20, large n</td>
          <td>High dim-এ brute-এর সমান</td>
        </tr>
        <tr>
          <td>Ball Tree</td>
          <td>O(n·d·log n)</td>
          <td>O(d·log n)</td>
          <td>High dim, non-Euclidean</td>
          <td>Build time বেশি</td>
        </tr>
      </tbody>
    </table>

    <h3>৪. Pipeline: StandardScaler + KNN</h3>
    <p>
      Pipeline ব্যবহার করলে preprocessing এবং model একসাথে পরিচালনা করা যায়।
      এটি data leakage এড়ায় এবং code পরিষ্কার রাখে:
    </p>
    <pre><code>import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib

# Pipeline তৈরি
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Pipeline-এর parameter naming: step_name__param_name
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
    'knn__algorithm': ['auto']
}

# GridSearchCV with Pipeline (data leakage নেই!)
grid_search = GridSearchCV(
    knn_pipeline,
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

# Raw (unscaled) data দিতে পারি - Pipeline নিজেই scale করবে
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search.fit(X_train, y_train)

print("Pipeline GridSearch Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# Model সংরক্ষণ
best_pipeline = grid_search.best_estimator_
joblib.dump(best_pipeline, 'knn_pipeline.pkl')
print("\nModel saved to knn_pipeline.pkl")

# Load ও predict
loaded_model = joblib.load('knn_pipeline.pkl')
predictions = loaded_model.predict(X_test[:5])
print(f"Sample predictions: {predictions}")</code></pre>

    <h3>৫. Learning Curve দিয়ে Data Efficiency বিশ্লেষণ</h3>
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

# Learning curve compute
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
plt.show()

# Gap বিশ্লেষণ
gap = train_mean - val_mean
print("\nTraining vs Validation Gap:")
for size, t, v, g in zip(train_sizes, train_mean, val_mean, gap):
    print(f"  n={int(size):4d}: Train={t:.4f}, Val={v:.4f}, Gap={g:.4f}")</code></pre>

    <h3>৬. KNN Tuning Best Practices সারসংক্ষেপ</h3>
    <table>
      <thead>
        <tr>
          <th>Hyperparameter</th>
          <th>Default</th>
          <th>Tuning Range</th>
          <th>পরামর্শ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>n_neighbors</td>
          <td>5</td>
          <td>1-50 (বা sqrt(n))</td>
          <td>Odd number ব্যবহার করো binary-তে</td>
        </tr>
        <tr>
          <td>weights</td>
          <td>uniform</td>
          <td>uniform, distance</td>
          <td>Distance প্রায়ই ভালো</td>
        </tr>
        <tr>
          <td>metric</td>
          <td>minkowski</td>
          <td>euclidean, manhattan, cosine</td>
          <td>Data type অনুযায়ী নির্বাচন</td>
        </tr>
        <tr>
          <td>algorithm</td>
          <td>auto</td>
          <td>auto, ball_tree, kd_tree, brute</td>
          <td>auto সাধারণত ঠিক আছে</td>
        </tr>
        <tr>
          <td>n_jobs</td>
          <td>1</td>
          <td>-1 (সব core)</td>
          <td>Large dataset-এ -1 দাও</td>
        </tr>
        <tr>
          <td>leaf_size</td>
          <td>30</td>
          <td>10-50</td>
          <td>Tree build/query speed balance</td>
        </tr>
      </tbody>
    </table>
  `,
};
