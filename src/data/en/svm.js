export const svmEn = [
  {
    slug: "svm-math-hyperplane-margin",
    title: "SVM Mathematics — Hyperplane, Margin, and Support Vectors",
    description: "How SVM finds the maximum-margin hyperplane — complete math, primal formulation, soft margin, and building from scratch in Python.",
    category: "Support Vector Machine",
    readTime: 13,
    content: `
    <h3>1. What is SVM and Why?</h3>
    <p>Logistic Regression and Decision Trees can draw many valid decision boundaries. SVM finds the unique boundary that <strong>maximises the margin</strong> — the gap between the two classes. This maximum-margin principle is the core strength of SVM.</p>
    <pre><code># Many valid hyperplanes exist between two classes.
# SVM selects the one with the largest margin:
#   ════════════════  ← maximum margin hyperplane

# Intuition: a wider gap means the model is more confident
# and generalises better to unseen data.</code></pre>

    <h3>2. Hyperplane Mathematics</h3>
    <pre><code>import numpy as np

# In n-dimensional space, a hyperplane is: w·x + b = 0
# w = normal vector, b = bias

w = np.array([2, -1])
b = -3

# Decision value for a point:
x1 = np.array([1, 2])   # 2(1) - (2) - 3 = -3  → negative side
x3 = np.array([4, 1])   # 2(4) - (1) - 3 = +4  → positive side

for label, x in [("x1", x1), ("x3", x3)]:
    val  = w @ x + b
    side = "positive" if val > 0 else "negative"
    print(f"{label}: w·x + b = {val:+.0f}  → {side} side")

# Geometric distance from hyperplane:
# d = |w·x + b| / ||w||
def dist(x, w, b):
    return abs(w @ x + b) / np.linalg.norm(w)

print(f"Distance of x3: {dist(x3, w, b):.4f}")</code></pre>

    <h3>3. The Margin</h3>
    <pre><code># Margin = 2 / ||w||   (for unit normal vector)
#
# SVM Objective:
#   Maximize margin  ←→  Minimize ||w||² / 2
#
# Constraints (all training samples correctly classified):
#   yᵢ(w·xᵢ + b) ≥ 1    for all i,  yᵢ ∈ {+1, -1}
#
# Support Vectors: points that sit exactly on the margin boundary:
#   yᵢ(w·xᵢ + b) = 1
# These are the ONLY points that define the model.</code></pre>

    <h3>4. Hard Margin vs Soft Margin</h3>
    <pre><code">import numpy as np

# Hard Margin: all points must be correctly classified.
#   Problem: no solution if data is not linearly separable.
#   Problem: a single outlier can destroy the margin.

# Soft Margin (C-SVM):
#   Allow some violations (slack variables ξᵢ ≥ 0).
#   Minimize: (1/2)||w||² + C × Σξᵢ
#   Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ

# C = regularisation parameter
C_values = [0.01, 0.1, 1, 10, 100]
for C in C_values:
    effect = ("Wide margin" if C < 0.1 else
              "Balanced (default)" if C == 1 else
              "Narrow margin" if C < 50 else
              "Hard margin-like")
    print(f"C={C:>6.2f}  {effect}")</code></pre>

    <h3>5. Hinge Loss Formulation</h3>
    <pre><code">import numpy as np

# Equivalent to minimising:
# C × Σ max(0, 1 - yᵢ(w·xᵢ + b))  +  (1/2)||w||²
#      └──── Hinge Loss ────┘          └── Regularisation ──┘

def hinge_loss(y_true, decision_scores):
    """y_true ∈ {-1, +1}"""
    return np.maximum(0, 1 - y_true * decision_scores)

y      = np.array([1, 1, -1, -1])
scores = np.array([2.5, 0.3, -1.8, 0.5])
losses = hinge_loss(y, scores)
print("Per-sample hinge losses:", losses.round(3))
print("Mean hinge loss:", losses.mean().round(4))</code></pre>

    <h3>6. SVM from Scratch (Gradient Descent)</h3>
    <pre><code">import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C, self.lr, self.epochs = C, lr, epochs

    def fit(self, X, y):
        n, d   = X.shape
        y_     = np.where(y <= 0, -1, 1)
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            for i in range(n):
                if y_[i] * (X[i] @ self.w + self.b) >= 1:
                    self.w -= self.lr * self.w           # no violation
                else:
                    self.w += self.lr * (self.C * y_[i] * X[i] - self.w)
                    self.b += self.lr * self.C * y_[i]  # violation

    def predict(self, X):
        return np.where(X @ self.w + self.b >= 0, 1, -1)

    def score(self, X, y):
        return (self.predict(X) == np.where(y <= 0, -1, 1)).mean()

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            random_state=42, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_train)
X_ts_s   = scaler.transform(X_test)

svm = SimpleSVM(C=1.0)
svm.fit(X_tr_s, y_train)
print(f"Scratch SVM Accuracy: {svm.score(X_ts_s, y_test):.4f}")</code></pre>

    <h3>Summary</h3>
    <table>
      <thead><tr><th>Concept</th><th>Key Point</th></tr></thead>
      <tbody>
        <tr><td>SVM Objective</td><td>Find the maximum-margin hyperplane</td></tr>
        <tr><td>Hyperplane</td><td>w·x + b = 0, margin = 2/‖w‖</td></tr>
        <tr><td>Support Vectors</td><td>Points on the margin boundary — they alone define the model</td></tr>
        <tr><td>Hard Margin</td><td>Zero violations allowed — only for linearly separable data</td></tr>
        <tr><td>C (Soft Margin)</td><td>Large C → overfit; small C → underfit</td></tr>
        <tr><td>Hinge Loss</td><td>max(0, 1 − yf(x)) — zero inside margin, linear outside</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "svm-kernel-trick",
    title: "The Kernel Trick — Non-linear SVM",
    description: "Why linear SVM fails on curved data, how kernel functions implicitly map to higher dimensions, and comparing RBF, Polynomial, and Sigmoid kernels.",
    category: "Support Vector Machine",
    readTime: 11,
    content: `
    <h3>1. When Linear SVM Fails</h3>
    <p>If data is not linearly separable — e.g. XOR patterns or circular clusters — no straight hyperplane can separate the classes. The kernel trick solves this by mapping data to a higher-dimensional space where a linear separator exists.</p>
    <pre><code>from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt

X_c, y_c = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
X_m, y_m = make_moons(  n_samples=200, noise=0.1, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, X, y, t in zip(axes, [X_c,X_m],[y_c,y_m],['Circles','Moons']):
    ax.scatter(X[y==0,0],X[y==0,1],c='steelblue',s=20,label='Class 0')
    ax.scatter(X[y==1,0],X[y==1,1],c='coral',    s=20,label='Class 1')
    ax.set_title(t); ax.legend()
plt.tight_layout(); plt.show()</code></pre>

    <h3>2. The Kernel Trick</h3>
    <pre><code">import numpy as np

# Direct approach: compute φ(x) explicitly, then φ(xᵢ)·φ(xⱼ)  ← expensive
# Kernel trick:   K(xᵢ,xⱼ) = φ(xᵢ)·φ(xⱼ)  ← compute implicitly, no explicit mapping!

# Polynomial kernel example:
# K(x,z) = (x·z + 1)²  corresponds to φ(x) = (x₁², √2·x₁x₂, x₂², √2·x₁, √2·x₂, 1)

x = np.array([2, 3])
z = np.array([1, 4])

k_poly = (np.dot(x, z) + 1) ** 2
print(f"Polynomial K(x,z) = {k_poly}")

gamma    = 0.5
k_rbf    = np.exp(-gamma * np.sum((x - z)**2))
print(f"RBF K(x,z)        = {k_rbf:.6f}")</code></pre>

    <h3>3. Four Main Kernels</h3>
    <pre><code">import numpy as np

x = np.array([1.0, 2.0, 3.0])
z = np.array([4.0, 5.0, 6.0])

kernels = {
    'Linear':     lambda x,z: np.dot(x,z),
    'Polynomial': lambda x,z: (np.dot(x,z) + 1)**3,
    'RBF':        lambda x,z: np.exp(-0.1 * np.sum((x-z)**2)),
    'Sigmoid':    lambda x,z: np.tanh(0.01 * np.dot(x,z) + 0),
}
for name, k in kernels.items():
    print(f"{name:12s}: {k(x,z):.6f}")</code></pre>

    <h3>4. Kernels in sklearn</h3>
    <pre><code">from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_ts_s = scaler.transform(X_ts)

kernels = {
    'linear':    SVC(kernel='linear', C=1.0),
    'poly(d=3)': SVC(kernel='poly',   C=1.0, degree=3, coef0=1),
    'rbf':       SVC(kernel='rbf',    C=1.0, gamma='scale'),
}
for name, clf in kernels.items():
    clf.fit(X_tr_s, y_tr)
    print(f"{name:12s}  Acc={clf.score(X_ts_s,y_ts):.4f}  SVs={clf.n_support_.sum()}")</code></pre>

    <h3>5. gamma Effect on RBF</h3>
    <pre><code">from sklearn.model_selection import cross_val_score

# gamma controls the "width" of the RBF bell curve
# Large gamma → narrow bell → each training point only influences its neighbours
#             → complex boundary → overfit
# Small gamma → wide bell → smooth boundary → underfit

for g in [0.001, 0.01, 0.1, 1, 10, 100]:
    cv = cross_val_score(SVC(kernel='rbf', C=1.0, gamma=g),
                         X_tr_s, y_tr, cv=5).mean()
    print(f"gamma={g:>6.3f}  CV Acc={cv:.4f}")

# Use gamma='scale' (default) = 1/(n_features × X.var())</code></pre>

    <h3>Kernel Comparison</h3>
    <table>
      <thead><tr><th>Kernel</th><th>Formula</th><th>Parameters</th><th>Best For</th></tr></thead>
      <tbody>
        <tr><td>Linear</td><td>x·z</td><td>C only</td><td>High-dimensional, linearly separable</td></tr>
        <tr><td>Polynomial</td><td>(x·z + c)^d</td><td>degree, coef0, C</td><td>Image features, NLP</td></tr>
        <tr><td>RBF</td><td>exp(−γ‖x−z‖²)</td><td>gamma, C</td><td>General default — best in most cases</td></tr>
        <tr><td>Sigmoid</td><td>tanh(γx·z + c)</td><td>gamma, coef0, C</td><td>Neural-network-like behaviour</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "svm-classification-python",
    title: "SVM Classification — Complete sklearn Guide",
    description: "sklearn SVC and LinearSVC for binary and multiclass classification, decision boundary visualisation, and probability calibration.",
    category: "Support Vector Machine",
    readTime: 11,
    content: `
    <h3>1. SVC vs LinearSVC</h3>
    <table>
      <thead><tr><th>Topic</th><th>SVC</th><th>LinearSVC</th></tr></thead>
      <tbody>
        <tr><td>Kernels</td><td>linear, rbf, poly, sigmoid</td><td>linear only</td></tr>
        <tr><td>Algorithm</td><td>libsvm (SMO)</td><td>liblinear</td></tr>
        <tr><td>Large datasets</td><td>Slow O(n²–n³)</td><td>Fast O(n)</td></tr>
        <tr><td>Probabilities</td><td>probability=True (Platt scaling)</td><td>Not directly (needs calibration)</td></tr>
        <tr><td>When to use</td><td>n &lt; 10,000, non-linear data</td><td>n &gt; 10,000, text classification</td></tr>
      </tbody>
    </table>

    <h3>2. SVC Pipeline</h3>
    <pre><code>from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)

# Scaling is REQUIRED — SVM is scale-sensitive
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale',
                   probability=True, random_state=42)),
])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_ts)
y_prob = pipe.predict_proba(X_ts)[:, 1]

print(classification_report(y_ts, y_pred, target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_ts, y_prob):.4f}")
cv = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC:  {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>3. Decision Boundary Visualisation</h3>
    <pre><code">import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
X_s = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (kernel, kw) in zip(axes, [
        ('linear',{}), ('poly',{'degree':3}), ('rbf',{'gamma':'scale'})]):
    clf = SVC(kernel=kernel, C=1.0, **kw)
    clf.fit(X_s, y)

    xx, yy = np.meshgrid(np.linspace(-3,3,200), np.linspace(-3,3,200))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_s[y==0,0], X_s[y==0,1], c='steelblue', s=20)
    ax.scatter(X_s[y==1,0], X_s[y==1,1], c='coral',     s=20)
    sv = clf.support_vectors_
    ax.scatter(sv[:,0], sv[:,1], s=80, facecolors='none',
               edgecolors='black', lw=1.5, label='SVs')
    ax.set_title(f'{kernel}  Acc={clf.score(X_s,y):.3f}')
plt.tight_layout(); plt.show()</code></pre>

    <h3>4. LinearSVC for Text Classification</h3>
    <pre><code">from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.med','sci.space','rec.sport.hockey','comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories,
                            remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=categories,
                            remove=('headers','footers','quotes'))

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True)),
    ('svm',   LinearSVC(C=0.1, max_iter=2000)),
])
pipe.fit(train.data, train.target)
from sklearn.metrics import classification_report
print(classification_report(test.target, pipe.predict(test.data),
      target_names=categories))</code></pre>

    <h3>5. Inspect Support Vectors</h3>
    <pre><code">clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(StandardScaler().fit_transform(X_tr), y_tr)

print(f"Total support vectors:  {clf.n_support_.sum()}")
print(f"Per class:              {clf.n_support_}")
print(f"Fraction of train set:  {100*clf.n_support_.sum()/len(X_tr):.1f}%")
# Fewer support vectors → simpler model → better generalisation</code></pre>
    `,
  },
  {
    slug: "svm-regression-svr",
    title: "Support Vector Regression (SVR) — Continuous Value Prediction",
    description: "How SVR uses the epsilon-insensitive tube, sklearn SVR and NuSVR, and comparison with linear regression.",
    category: "Support Vector Machine",
    readTime: 10,
    content: `
    <h3>1. How SVR Works</h3>
    <p>Linear Regression minimises every point's error. SVR creates an <strong>ε-tube</strong> around the regression line — points inside the tube incur zero loss. Only points outside the tube are penalised.</p>
    <pre><code># SVR Objective:
# Minimize: (1/2)||w||² + C Σ(ξᵢ + ξᵢ*)
# Subject to:
#   yᵢ - f(xᵢ) ≤ ε + ξᵢ   (upper boundary)
#   f(xᵢ) - yᵢ ≤ ε + ξᵢ*  (lower boundary)
#   ξᵢ, ξᵢ* ≥ 0
#
# ε = tube half-width (zero loss inside)
# C = penalty for points outside the tube</code></pre>

    <h3>2. sklearn SVR</h3>
    <pre><code">from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()
X, y = data.data[:2000], data.target[:2000]  # SVR is slow on large n
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')),
])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_ts)

print(f"R²:   {r2_score(y_ts, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_ts, y_pred)):.4f}")
cv = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"CV R²: {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>3. Effect of epsilon</h3>
    <pre><code">from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

scaler = StandardScaler()
X_s  = scaler.fit_transform(X_tr)
X_ts_s = scaler.transform(X_ts)

print(f"{'epsilon':>9}  {'R²':>8}  {'Support Vectors':>16}")
for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
    svr = SVR(kernel='rbf', C=10, epsilon=eps, gamma='scale')
    svr.fit(X_s, y_tr)
    r2 = r2_score(y_ts, svr.predict(X_ts_s))
    print(f"{eps:>9.2f}  {r2:>8.4f}  {svr.n_support_[0]:>16}")
# Small epsilon → more SVs → complex model
# Large epsilon → fewer SVs → simpler model</code></pre>

    <h3>4. NuSVR — Control Support Vector Fraction</h3>
    <pre><code">from sklearn.svm import NuSVR

# ν ∈ (0,1]: upper bound on fraction of training points as SVs
for nu in [0.1, 0.3, 0.5, 0.7, 0.9]:
    svr = NuSVR(nu=nu, C=10, kernel='rbf', gamma='scale')
    svr.fit(X_s, y_tr)
    r2   = r2_score(y_ts, svr.predict(X_ts_s))
    frac = svr.n_support_[0] / len(X_tr)
    print(f"ν={nu:.1f}  SVs={svr.n_support_[0]} ({frac:.1%})  R²={r2:.4f}")</code></pre>

    <h3>Summary</h3>
    <table>
      <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
      <tbody>
        <tr><td>ε-tube</td><td>Zero loss inside; linear penalty outside</td></tr>
        <tr><td>C</td><td>Penalty for tube violations — like SVC's C</td></tr>
        <tr><td>epsilon</td><td>Tube width — larger = fewer SVs = simpler model</td></tr>
        <tr><td>NuSVR</td><td>Control the fraction of support vectors via ν</td></tr>
        <tr><td>SVR vs LR</td><td>SVR ignores small errors and is less sensitive to outliers</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "svm-multiclass",
    title: "SVM Multiclass Classification — OvR and OvO Strategies",
    description: "SVM is inherently binary — how One-vs-Rest and One-vs-One extend it to multiple classes, with full sklearn implementation.",
    category: "Support Vector Machine",
    readTime: 10,
    content: `
    <h3>1. Two Multiclass Strategies</h3>
    <pre><code># For K classes:
# One-vs-Rest (OvR): K binary classifiers — each class vs all others
#   Class A vs {B,C,D}
#   Class B vs {A,C,D}  ...
#   → highest confidence wins

# One-vs-One (OvO): K(K-1)/2 classifiers — every pair
#   A vs B,  A vs C,  A vs D,  B vs C,  B vs D,  C vs D
#   → majority vote wins

# For K=4: OvR=4 classifiers, OvO=6 classifiers</code></pre>

    <h3>2. sklearn Strategies</h3>
    <pre><code">from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = load_iris()
X, y = data.data, data.target
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)

strategies = {
    'SVC (OvO default)':   Pipeline([('s',StandardScaler()),('m',SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo'))]),
    'SVC (OvR)':           Pipeline([('s',StandardScaler()),('m',SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr'))]),
    'LinearSVC':           Pipeline([('s',StandardScaler()),('m',LinearSVC(C=1.0, max_iter=5000))]),
}

for name, pipe in strategies.items():
    pipe.fit(X_tr, y_tr)
    test_acc = pipe.score(X_ts, y_ts)
    cv_acc   = cross_val_score(pipe, X, y, cv=5).mean()
    print(f"{name:22s}  Test={test_acc:.4f}  CV={cv_acc:.4f}")</code></pre>

    <h3>3. Multiclass Evaluation</h3>
    <pre><code">from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

pipe = Pipeline([
    ('s', StandardScaler()),
    ('m', SVC(kernel='rbf', C=5.0, gamma='scale', probability=True)),
])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_ts)
y_prob = pipe.predict_proba(X_ts)

print(classification_report(y_ts, y_pred, target_names=data.target_names))

y_bin = label_binarize(y_ts, classes=[0,1,2])
auc   = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
print(f"Macro ROC-AUC: {auc:.4f}")</code></pre>

    <h3>OvR vs OvO Comparison</h3>
    <table>
      <thead><tr><th>Topic</th><th>One-vs-Rest (OvR)</th><th>One-vs-One (OvO)</th></tr></thead>
      <tbody>
        <tr><td>Number of classifiers</td><td>K</td><td>K(K-1)/2</td></tr>
        <tr><td>Training time</td><td>Less (K classifiers)</td><td>More (K² classifiers)</td></tr>
        <tr><td>Training data per clf</td><td>Full (imbalanced)</td><td>Smaller (balanced pairs)</td></tr>
        <tr><td>sklearn SVC default</td><td>decision_function_shape='ovr'</td><td>Internally uses OvO</td></tr>
        <tr><td>Best for</td><td>LinearSVC, many classes</td><td>SVC with kernels</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "svm-hyperparameter-tuning",
    title: "SVM Hyperparameter Tuning — C, gamma, and Kernel Selection",
    description: "Effect of C and gamma, GridSearchCV and RandomizedSearchCV, learning curves for overfit/underfit diagnosis.",
    category: "Support Vector Machine",
    readTime: 11,
    content: `
    <h3>1. Three Core Hyperparameters</h3>
    <table>
      <thead><tr><th>Parameter</th><th>Default</th><th>Larger value</th><th>Smaller value</th></tr></thead>
      <tbody>
        <tr><td><strong>C</strong></td><td>1.0</td><td>Less regularisation, narrow margin, risk of overfit</td><td>More regularisation, wide margin, risk of underfit</td></tr>
        <tr><td><strong>gamma</strong> (rbf)</td><td>'scale'</td><td>Complex boundary, local, overfit</td><td>Smooth boundary, global, underfit</td></tr>
        <tr><td><strong>epsilon</strong> (SVR)</td><td>0.1</td><td>Wider tube, fewer SVs, simpler model</td><td>Narrower tube, more SVs, complex model</td></tr>
      </tbody>
    </table>

    <h3>2. C vs gamma Heatmap</h3>
    <pre><code">import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            random_state=42, n_clusters_per_class=1)
X_s = StandardScaler().fit_transform(X)

C_range     = np.logspace(-2, 3, 8)
gamma_range = np.logspace(-4, 1, 8)
scores      = np.zeros((len(C_range), len(gamma_range)))

for i, C in enumerate(C_range):
    for j, g in enumerate(gamma_range):
        scores[i,j] = cross_val_score(SVC(kernel='rbf',C=C,gamma=g),
                                       X_s, y, cv=5).mean()

plt.imshow(scores, cmap='viridis', vmin=0.5, vmax=1.0)
plt.colorbar(label='CV Accuracy')
plt.xticks(range(len(gamma_range)), [f'{g:.0e}' for g in gamma_range], rotation=45)
plt.yticks(range(len(C_range)),     [f'{c:.0e}' for c in C_range])
plt.xlabel('gamma'); plt.ylabel('C')
plt.title('C vs gamma — CV Accuracy Heatmap'); plt.show()</code></pre>

    <h3>3. GridSearchCV</h3>
    <pre><code">from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler',StandardScaler()), ('svm',SVC(probability=True,random_state=42))])

param_grid = [
    {'svm__kernel':['rbf'],    'svm__C':[0.1,1,10,100], 'svm__gamma':['scale','auto',0.001,0.01,0.1]},
    {'svm__kernel':['linear'], 'svm__C':[0.001,0.01,0.1,1,10]},
    {'svm__kernel':['poly'],   'svm__C':[0.1,1,10], 'svm__degree':[2,3,4], 'svm__coef0':[0,1]},
]
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_tr, y_tr)
print(f"Best params: {grid.best_params_}")
print(f"CV AUC:      {grid.best_score_:.4f}")</code></pre>

    <h3>4. RandomizedSearchCV</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'svm__C':     loguniform(1e-3, 1e3),
    'svm__gamma': loguniform(1e-5, 10),
    'svm__kernel':['rbf','linear'],
}
search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                             scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_tr, y_tr)
print(f"Best params: {search.best_params_}")</code></pre>

    <h3>5. Kernel Selection Guide</h3>
    <table>
      <thead><tr><th>Scenario</th><th>Recommended Kernel</th><th>Reason</th></tr></thead>
      <tbody>
        <tr><td>n_features &gt;&gt; n_samples (text)</td><td>linear</td><td>High-dim is already rich enough</td></tr>
        <tr><td>n_samples &lt; 10,000, non-linear</td><td>rbf</td><td>General-purpose, usually best</td></tr>
        <tr><td>Image or polynomial relationships</td><td>poly</td><td>Captures feature interactions</td></tr>
        <tr><td>n_samples &gt; 100,000</td><td>LinearSVC</td><td>liblinear is much faster</td></tr>
        <tr><td>Unknown — starting out</td><td>rbf, C=1, gamma='scale'</td><td>Best default starting point</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "svm-project-cancer-detection",
    title: "Project: Cancer Detection — SVM vs Other Classifiers",
    description: "End-to-end ML pipeline on the Breast Cancer dataset: EDA, model comparison, hyperparameter tuning, and threshold analysis.",
    category: "Support Vector Machine",
    readTime: 14,
    content: `
    <h3>1. Dataset Overview</h3>
    <table>
      <thead><tr><th>Info</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Samples</td><td>569</td></tr>
        <tr><td>Features</td><td>30 (geometric measurements of cell nuclei)</td></tr>
        <tr><td>Malignant</td><td>212 (37.3%)</td></tr>
        <tr><td>Benign</td><td>357 (62.7%)</td></tr>
      </tbody>
    </table>

    <h3>2. Load Data and EDA</h3>
    <pre><code">import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.describe())
print(df['target'].value_counts())

# Feature distributions by class
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for i, col in enumerate(data.feature_names[:15]):
    ax = axes[i//5, i%5]
    df[df.target==0][col].hist(ax=ax,bins=25,alpha=0.6,color='crimson', label='Malignant',density=True)
    df[df.target==1][col].hist(ax=ax,bins=25,alpha=0.6,color='steelblue',label='Benign',  density=True)
    ax.set_title(col[:20],fontsize=8); ax.legend(fontsize=6)
plt.tight_layout(); plt.show()</code></pre>

    <h3>3. Model Comparison</h3>
    <pre><code">from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

X, y = data.data, data.target
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'SVM (RBF)':           SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'SVM (Linear)':        LinearSVC(C=1.0, max_iter=5000),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Gaussian NB':         GaussianNB(),
}

import pandas as pd
results = {}
for name, model in models.items():
    pipe = Pipeline([('s', StandardScaler()), ('m', model)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_ts)
    cv_acc = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
    try:    auc = roc_auc_score(y_ts, pipe.predict_proba(X_ts)[:,1])
    except: auc = roc_auc_score(y_ts, pipe.decision_function(X_ts))
    results[name] = {'CV Acc': cv_acc, 'Test Acc': accuracy_score(y_ts,y_pred),
                     'F1 (Malignant)': f1_score(y_ts,y_pred,pos_label=0), 'AUC': auc}

print(pd.DataFrame(results).T.sort_values('AUC',ascending=False).round(4))</code></pre>

    <h3>4. Tune the Best SVM</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

pipe_svm = Pipeline([('s',StandardScaler()), ('svm',SVC(probability=True,random_state=42))])
param_dist = {'svm__kernel':['rbf','linear'], 'svm__C':loguniform(0.1,1000), 'svm__gamma':loguniform(1e-5,10)}
search = RandomizedSearchCV(pipe_svm, param_dist, n_iter=50, cv=5,
                             scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_tr, y_tr)
print(f"Best params: {search.best_params_}")
print(f"Test AUC:    {roc_auc_score(y_ts, search.predict_proba(X_ts)[:,1]):.4f}")</code></pre>

    <h3>5. Threshold Analysis for Medical Context</h3>
    <pre><code">import numpy as np

best    = search.best_estimator_
y_prob  = best.predict_proba(X_ts)[:,1]

# In cancer detection, missing a malignant tumour (FN) is far more costly
# Lower threshold → catch more malignant cases → higher recall
print(f"{'Threshold':>10}  {'FN (missed cancer)':>20}  {'FP (false alarm)':>18}  {'F1':>6}")
for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
    y_pred_t = (y_prob < thr).astype(int)
    fn = ((y_ts==0) & (y_pred_t==1)).sum()
    fp = ((y_ts==1) & (y_pred_t==0)).sum()
    f1 = f1_score(y_ts, y_pred_t, pos_label=0)
    print(f"{thr:>10.1f}  {fn:>20}  {fp:>18}  {f1:>6.4f}")</code></pre>

    <h3>Model Performance Summary</h3>
    <table>
      <thead><tr><th>Model</th><th>Strengths</th><th>Weaknesses</th><th>Verdict</th></tr></thead>
      <tbody>
        <tr><td>SVM (RBF)</td><td>High accuracy, non-linear</td><td>Slow on large n</td><td>Excellent for this dataset</td></tr>
        <tr><td>SVM (Linear)</td><td>Fast, interpretable</td><td>Linear boundary only</td><td>Good baseline</td></tr>
        <tr><td>Logistic Regression</td><td>Probabilities, fast</td><td>Linear boundary</td><td>Strong baseline</td></tr>
        <tr><td>Random Forest</td><td>Robust, feature importance</td><td>Slower prediction</td><td>Good if explainability needed</td></tr>
        <tr><td>Gradient Boosting</td><td>Usually most accurate</td><td>Many hyperparameters</td><td>Best for production</td></tr>
      </tbody>
    </table>
    `,
  },
];
