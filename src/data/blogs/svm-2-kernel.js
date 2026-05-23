export const svm_2_kernel = {
  title: "Kernel Trick — Non-linear SVM-এর রহস্য",
  description: "Linear SVM কখন কাজ করে না, kernel function কীভাবে data কে higher dimension-এ নিয়ে যায় — RBF, Polynomial, Sigmoid kernel বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 11,
  slug: "svm-kernel-trick",
  content: `
    <h3>১. Linear SVM কখন কাজ করে না?</h3>
    <p>যদি data linearly separable না হয় — যেমন XOR problem বা circular clusters — তাহলে কোনো straight hyperplane দুটো class আলাদা করতে পারে না। Kernel Trick এই সমস্যার সমাধান।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons

# Linearly non-separable data
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
X_moons,   y_moons   = make_moons(n_samples=200,   noise=0.1, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, X, y, title in zip(axes,
    [X_circles, X_moons], [y_circles, y_moons],
    ['Circles (Non-separable)', 'Moons (Non-separable)']):
    ax.scatter(X[y==0,0], X[y==0,1], c='steelblue', s=20, label='Class 0')
    ax.scatter(X[y==1,0], X[y==1,1], c='coral',     s=20, label='Class 1')
    ax.set_title(title); ax.legend()
plt.tight_layout(); plt.show()</code></pre>

    <h3>২. Kernel Trick-এর ধারণা</h3>
    <p>Data-কে higher-dimensional space-এ map করলে সেখানে linearly separable হতে পারে। কিন্তু directly map করা computationally expensive। Kernel trick এই mapping <strong>implicitly</strong> করে — শুধু dot product হিসাব করে।</p>
    <pre><code># Feature Mapping φ: ℝⁿ → ℝᵐ  (m >> n)
# φ(x₁, x₂) = (x₁², √2·x₁x₂, x₂²)  [উদাহরণ]

# Direct approach: compute φ(x) explicitly, then φ(xᵢ)·φ(xⱼ)
# Kernel Trick:    K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)  ← directly হিসাব করো!

# Polynomial Kernel উদাহরণ:
# K(x, z) = (x·z + 1)²
# = (x₁z₁ + x₂z₂ + 1)²
# = x₁²z₁² + 2x₁z₁x₂z₂ + x₂²z₂² + 2x₁z₁ + 2x₂z₂ + 1
# = φ(x)·φ(z)  যেখানে φ(x) = (x₁², √2·x₁x₂, x₂², √2·x₁, √2·x₂, 1)

import numpy as np

x = np.array([2, 3])
z = np.array([1, 4])

# Direct polynomial kernel (degree=2, coef0=1):
k_poly = (np.dot(x, z) + 1) ** 2
print(f"Polynomial K(x,z) = {k_poly}")

# RBF/Gaussian kernel:
# K(x, z) = exp(-γ||x - z||²)
gamma = 0.5
k_rbf = np.exp(-gamma * np.sum((x - z)**2))
print(f"RBF K(x,z)        = {k_rbf:.6f}")</code></pre>

    <h3>৩. চারটি প্রধান Kernel</h3>
    <pre><code>import numpy as np

def kernel_linear(x, z):
    """K(x,z) = x·z"""
    return np.dot(x, z)

def kernel_polynomial(x, z, degree=3, coef0=1):
    """K(x,z) = (x·z + coef0)^degree"""
    return (np.dot(x, z) + coef0) ** degree

def kernel_rbf(x, z, gamma=0.5):
    """K(x,z) = exp(-γ‖x-z‖²) — সবচেয়ে জনপ্রিয়"""
    return np.exp(-gamma * np.sum((x - z)**2))

def kernel_sigmoid(x, z, coef0=0, gamma=0.01):
    """K(x,z) = tanh(γ·x·z + coef0)"""
    return np.tanh(gamma * np.dot(x, z) + coef0)

x = np.array([1.0, 2.0, 3.0])
z = np.array([4.0, 5.0, 6.0])

print(f"Linear:     {kernel_linear(x, z):.4f}")
print(f"Polynomial: {kernel_polynomial(x, z, degree=3):.4f}")
print(f"RBF:        {kernel_rbf(x, z, gamma=0.1):.6f}")
print(f"Sigmoid:    {kernel_sigmoid(x, z):.6f}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Kernel Trick — Low-dim → High-dim Mapping</text>
        <rect x="10" y="28" width="130" height="75" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="75" y="52" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">2D Input Space</text>
        <circle cx="55" cy="75" r="5" fill="#3b82f6"/>
        <circle cx="95" cy="68" r="5" fill="#ef4444"/>
        <circle cx="75" cy="80" r="5" fill="#3b82f6"/>
        <circle cx="85" cy="60" r="5" fill="#ef4444"/>
        <text x="75" y="97" text-anchor="middle" font-size="8" fill="#6b7280">non-separable</text>
        <text x="148" y="68" font-size="20" fill="#94a3b8">→</text>
        <text x="150" y="84" text-anchor="middle" font-size="8" fill="#16a34a">φ(x)</text>
        <rect x="170" y="28" width="140" height="75" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="240" y="52" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Higher-dim Space</text>
        <circle cx="195" cy="72" r="5" fill="#3b82f6"/>
        <circle cx="265" cy="60" r="5" fill="#ef4444"/>
        <circle cx="200" cy="85" r="5" fill="#3b82f6"/>
        <circle cx="270" cy="75" r="5" fill="#ef4444"/>
        <line x1="230" y1="55" x2="230" y2="95" stroke="#111827" stroke-width="1.5" stroke-dasharray="4,3"/>
        <text x="240" y="97" text-anchor="middle" font-size="8" fill="#6b7280">linearly separable!</text>
        <text x="318" y="68" font-size="20" fill="#94a3b8">→</text>
        <rect x="340" y="28" width="130" height="75" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="405" y="48" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Kernel K(x,z)</text>
        <text x="405" y="64" text-anchor="middle" font-size="8" fill="#16a34a">φ(x)·φ(z)</text>
        <text x="405" y="78" text-anchor="middle" font-size="8" fill="#16a34a">computed</text>
        <text x="405" y="91" text-anchor="middle" font-size="8" fill="#16a34a">implicitly!</text>
      </svg>
    </div>

    <h3>৪. sklearn-এ Kernel SVM</h3>
    <pre><code>from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

kernels = {
    'linear':     SVC(kernel='linear',     C=1.0),
    'poly(d=3)':  SVC(kernel='poly',       C=1.0, degree=3, coef0=1),
    'rbf':        SVC(kernel='rbf',        C=1.0, gamma='scale'),
    'sigmoid':    SVC(kernel='sigmoid',    C=1.0, gamma='scale', coef0=0),
}

for name, clf in kernels.items():
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_test_s, y_test)
    n_sv = clf.n_support_.sum()
    print(f"{name:12s}  Acc={acc:.4f}  Support Vectors={n_sv}")</code></pre>

    <h3>৫. RBF Kernel-এ gamma-র প্রভাব</h3>
    <pre><code"># gamma = 1/(2σ²) — RBF kernel-এর width নিয়ন্ত্রণ করে
# gamma বড়  → narrow bell curve → প্রতিটি training point শুধু নিজের কাছাকাছি influence করে
#           → complex decision boundary → overfit
# gamma ছোট → wide bell curve → smooth decision boundary → underfit

from sklearn.model_selection import cross_val_score
import numpy as np

gammas = [0.001, 0.01, 0.1, 1, 10, 100]
print(f"{'gamma':>8}  {'CV Accuracy':>12}")
for g in gammas:
    clf = SVC(kernel='rbf', C=1.0, gamma=g)
    cv  = cross_val_score(clf, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"{g:>8.3f}  {cv.mean():>12.4f}")

# sklearn shortcuts:
# gamma='scale' → 1 / (n_features × X.var())  ← default, recommended
# gamma='auto'  → 1 / n_features</code></pre>

    <h3>৬. Kernel তুলনা</h3>
    <table>
      <thead><tr><th>Kernel</th><th>Formula</th><th>Parameters</th><th>সেরা ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>Linear</td><td>x·z</td><td>শুধু C</td><td>High-dimensional, linearly separable</td></tr>
        <tr><td>Polynomial</td><td>(x·z + c)^d</td><td>degree, coef0, C</td><td>Image features, NLP</td></tr>
        <tr><td>RBF</td><td>exp(-γ‖x-z‖²)</td><td>gamma, C</td><td>সাধারণ default — বেশিরভাগ ক্ষেত্রে সেরা</td></tr>
        <tr><td>Sigmoid</td><td>tanh(γx·z + c)</td><td>gamma, coef0, C</td><td>Neural network-এর মতো behavior</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Kernel Trick</td><td>High-dimensional mapping implicitly করে — computationally cheap</td></tr>
        <tr><td>RBF Kernel</td><td>সবচেয়ে জনপ্রিয় — gamma দিয়ে smoothness control</td></tr>
        <tr><td>gamma বড়</td><td>Complex boundary → overfit</td></tr>
        <tr><td>gamma ছোট</td><td>Smooth boundary → underfit</td></tr>
        <tr><td>Default choice</td><td>RBF kernel, gamma='scale', C=1.0 দিয়ে শুরু করো</td></tr>
      </tbody>
    </table>
  `,
};
