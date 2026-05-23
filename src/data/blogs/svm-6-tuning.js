export const svm_6_tuning = {
  title: "SVM Hyperparameter Tuning — C, gamma ও Kernel নির্বাচন",
  description: "C ও gamma-র প্রভাব, GridSearchCV ও RandomizedSearchCV, learning curve দিয়ে overfit/underfit diagnosis — বাংলায় সম্পূর্ণ SVM tuning গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 11,
  slug: "svm-hyperparameter-tuning",
  content: `
    <h3>১. তিনটি মূল Hyperparameter</h3>
    <table>
      <thead><tr><th>Parameter</th><th>Default</th><th>বড় করলে</th><th>ছোট করলে</th></tr></thead>
      <tbody>
        <tr><td><strong>C</strong></td><td>1.0</td><td>Less regularization, narrow margin, overfit-এর ঝুঁকি</td><td>More regularization, wide margin, underfit-এর ঝুঁকি</td></tr>
        <tr><td><strong>gamma</strong> (rbf)</td><td>'scale'</td><td>Complex boundary, local decision, overfit</td><td>Smooth boundary, global decision, underfit</td></tr>
        <tr><td><strong>epsilon</strong> (SVR)</td><td>0.1</td><td>Wider tube, fewer SVs, simpler model</td><td>Narrower tube, more SVs, complex model</td></tr>
      </tbody>
    </table>

    <h3>২. C ও gamma-র যৌথ প্রভাব</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
X_s = StandardScaler().fit_transform(X)

C_range     = np.logspace(-2, 3, 10)
gamma_range = np.logspace(-4, 1, 10)

scores = np.zeros((len(C_range), len(gamma_range)))
for i, C in enumerate(C_range):
    for j, g in enumerate(gamma_range):
        clf = SVC(kernel='rbf', C=C, gamma=g)
        scores[i,j] = cross_val_score(clf, X_s, y, cv=5, scoring='accuracy').mean()

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(scores, interpolation='nearest', cmap='viridis', vmin=0.5, vmax=1.0)
plt.colorbar(label='CV Accuracy')
plt.xticks(range(len(gamma_range)), [f'{g:.1e}' for g in gamma_range], rotation=45, ha='right')
plt.yticks(range(len(C_range)),     [f'{c:.1e}' for c in C_range])
plt.xlabel('gamma'); plt.ylabel('C')
plt.title('SVM RBF: C vs gamma Heatmap')
plt.tight_layout(); plt.show()</code></pre>

    <h3>৩. GridSearchCV দিয়ে Tuning</h3>
    <pre><code>from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(probability=True, random_state=42)),
])

param_grid = [
    {
        'svm__kernel': ['rbf'],
        'svm__C':      [0.1, 1, 10, 100],
        'svm__gamma':  ['scale', 'auto', 0.001, 0.01, 0.1],
    },
    {
        'svm__kernel': ['linear'],
        'svm__C':      [0.001, 0.01, 0.1, 1, 10],
    },
    {
        'svm__kernel': ['poly'],
        'svm__C':      [0.1, 1, 10],
        'svm__degree': [2, 3, 4],
        'svm__coef0':  [0, 1],
    },
]

grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"সেরা params: {grid.best_params_}")
print(f"CV AUC:      {grid.best_score_:.4f}")
print(f"Test AUC:    {grid.score(X_test, y_test):.4f}")</code></pre>

    <h3>৪. RandomizedSearchCV — দ্রুত অনুসন্ধান</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
import numpy as np

# Continuous parameter space sampling
param_dist = {
    'svm__C':     loguniform(1e-3, 1e3),   # log-uniform over [0.001, 1000]
    'svm__gamma': loguniform(1e-5, 1e1),   # log-uniform over [0.00001, 10]
    'svm__kernel': ['rbf', 'linear'],
}

search = RandomizedSearchCV(
    pipe, param_dist,
    n_iter=50, cv=5,
    scoring='roc_auc',
    random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"সেরা params: {search.best_params_}")
print(f"Test AUC:    {search.score(X_test, y_test):.4f}")</code></pre>

    <h3>৫. Learning Curve দিয়ে Diagnosis</h3>
    <pre><code">from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='roc_auc', n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.plot(train_sizes, train_mean, 'o-', label='Train AUC')
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
    plt.plot(train_sizes, val_mean, 's-', label='CV AUC')
    plt.fill_between(train_sizes, val_mean-val_std,   val_mean+val_std,   alpha=0.15)
    plt.xlabel('Training samples'); plt.ylabel('AUC')
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Underfit: C খুব ছোট
plt.sca(axes[0])
plot_learning_curve(
    Pipeline([('s',StandardScaler()),('m',SVC(C=0.001,kernel='rbf',gamma='scale'))]),
    X, y, 'SVM C=0.001 (Underfit)')

# Good fit
plt.sca(axes[1])
plot_learning_curve(
    Pipeline([('s',StandardScaler()),('m',SVC(C=10,kernel='rbf',gamma='scale'))]),
    X, y, 'SVM C=10 (Good Fit)')

plt.tight_layout(); plt.show()</code></pre>

    <h3>৬. Kernel নির্বাচন গাইড</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>প্রস্তাবিত Kernel</th><th>কারণ</th></tr></thead>
      <tbody>
        <tr><td>n_features &gt;&gt; n_samples (text)</td><td>linear</td><td>High-dim-এ linear প্রায়ই যথেষ্ট</td></tr>
        <tr><td>n_samples &lt; 10,000, non-linear</td><td>rbf</td><td>General-purpose, usually best</td></tr>
        <tr><td>Image data, polynomial relationships</td><td>poly</td><td>Polynomial interaction capture করে</td></tr>
        <tr><td>n_samples &gt; 100,000</td><td>LinearSVC</td><td>liblinear অনেক দ্রুত</td></tr>
        <tr><td>Unknown, tuning করছো</td><td>rbf থেকে শুরু করো</td><td>Default best choice</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>C ও gamma</td><td>সবচেয়ে গুরুত্বপূর্ণ — log scale-এ tune করো</td></tr>
        <tr><td>GridSearchCV</td><td>ছোট grid-এ exhaustive search</td></tr>
        <tr><td>RandomizedSearchCV</td><td>বড় parameter space-এ দ্রুত search</td></tr>
        <tr><td>Learning curve</td><td>overfit/underfit diagnosis-এর সেরা tool</td></tr>
        <tr><td>শুরু করো</td><td>RBF, C=1, gamma='scale' — তারপর tune</td></tr>
      </tbody>
    </table>
  `,
};
