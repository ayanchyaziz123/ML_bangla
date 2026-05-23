export const svm_1_math = {
  title: "SVM-এর গণিত — Hyperplane, Margin ও Support Vectors",
  description: "Support Vector Machine কীভাবে সর্বোচ্চ margin-এর hyperplane খোঁজে — সম্পূর্ণ গণিত, প্রাইমাল ফর্মুলেশন ও Python থেকে স্ক্র্যাচ বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 13,
  slug: "svm-math-hyperplane-margin",
  content: `
    <h3>১. SVM কী এবং কেন?</h3>
    <p>Logistic Regression বা Decision Tree অনেক সম্ভাব্য decision boundary আঁকতে পারে। SVM শুধু সেই boundary খোঁজে যার দুই class থেকে <strong>দূরত্ব (margin) সর্বোচ্চ</strong>। এই margin maximization-ই SVM-এর মূল শক্তি।</p>
    <pre><code>দুটো class: ● (positive) এবং ○ (negative)

Multiple valid hyperplanes:
  ────────────────   (কাছে কাছে)
  ── ── ── ── ──    (margin কম)
  ════════════════   ← SVM এটা বেছে নেয় (maximum margin!)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">SVM — Maximum Margin Hyperplane</text>
        <!-- Negative class -->
        <circle cx="80"  cy="80" r="7" fill="#3b82f6" opacity="0.8"/>
        <circle cx="60"  cy="55" r="7" fill="#3b82f6" opacity="0.8"/>
        <circle cx="100" cy="60" r="7" fill="#3b82f6" opacity="0.8"/>
        <circle cx="70"  cy="100" r="7" fill="#3b82f6" opacity="0.8"/>
        <!-- Positive class -->
        <circle cx="360" cy="55" r="7" fill="#ef4444" opacity="0.8"/>
        <circle cx="380" cy="80" r="7" fill="#ef4444" opacity="0.8"/>
        <circle cx="350" cy="90" r="7" fill="#ef4444" opacity="0.8"/>
        <circle cx="400" cy="60" r="7" fill="#ef4444" opacity="0.8"/>
        <!-- Support vectors (marked) -->
        <circle cx="140" cy="70" r="7" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
        <circle cx="140" cy="70" r="4" fill="#3b82f6"/>
        <circle cx="310" cy="75" r="7" fill="none" stroke="#ef4444" stroke-width="2.5"/>
        <circle cx="310" cy="75" r="4" fill="#ef4444"/>
        <!-- Hyperplane -->
        <line x1="225" y1="20" x2="225" y2="118" stroke="#111827" stroke-width="2.5" stroke-dasharray="0"/>
        <!-- Margin lines -->
        <line x1="170" y1="20" x2="170" y2="118" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5,4"/>
        <line x1="280" y1="20" x2="280" y2="118" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5,4"/>
        <!-- Margin brace -->
        <text x="225" y="112" text-anchor="middle" font-size="9" fill="#16a34a" font-weight="600">← margin →</text>
        <text x="120" y="40" font-size="9" fill="#3b82f6" font-weight="600">negative</text>
        <text x="340" y="40" font-size="9" fill="#ef4444" font-weight="600">positive</text>
        <text x="225" y="34" text-anchor="middle" font-size="9" fill="#111827" font-weight="700">hyperplane</text>
      </svg>
    </div>

    <h3>২. Hyperplane-এর গণিত</h3>
    <p>n-dimensional space-এ একটি hyperplane হলো: <strong>w·x + b = 0</strong> — যেখানে w হলো normal vector এবং b হলো bias।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# 2D উদাহরণ: w = [2, -1], b = -3
# Hyperplane: 2x₁ - x₂ - 3 = 0  → x₂ = 2x₁ - 3

w = np.array([2, -1])
b = -3

# দুটো point-এর জন্য decision value হিসাব
x1 = np.array([1, 2])   # 2(1) - (2) - 3 = -3 → negative side
x2 = np.array([3, 4])   # 2(3) - (4) - 3 = -1 → still negative
x3 = np.array([4, 1])   # 2(4) - (1) - 3 = +4 → positive side

for label, x in [("x1", x1), ("x2", x2), ("x3", x3)]:
    val = w @ x + b
    side = "positive" if val > 0 else "negative"
    print(f"{label}: w·x + b = {val:+.0f}  → {side} side")

# Distance from hyperplane:
# d = |w·x + b| / ||w||
def distance_from_hyperplane(x, w, b):
    return abs(w @ x + b) / np.linalg.norm(w)

print(f"\nDistance of x3 from hyperplane: {distance_from_hyperplane(x3, w, b):.4f}")</code></pre>

    <h3>৩. Margin কী?</h3>
    <pre><code># Margin = দুই class-এর closest points থেকে hyperplane-এর দূরত্বের যোগফল
# = 2 / ||w||  (unit vector-এর ক্ষেত্রে)

# SVM-এর লক্ষ্য:
# Maximize margin = Maximize 2/||w|| = Minimize ||w||²/2

# Constraints (সব training sample correctly classified):
# yᵢ(w·xᵢ + b) ≥ 1  for all i
# যেখানে yᵢ ∈ {+1, -1}

# Support Vectors:
# যে points-গুলো ঠিক margin-এর উপর বসে আছে:
# yᵢ(w·xᵢ + b) = 1  (এরাই model define করে!)</code></pre>

    <h3>৪. Hard Margin vs Soft Margin</h3>
    <pre><code># Hard Margin SVM:
# সব training point correctly classified হতেই হবে
# সমস্যা: linearly separable না হলে কোনো solution নেই
# সমস্যা: outlier থাকলে margin খুব ছোট হয়ে যায়

# Soft Margin SVM (C-SVM):
# কিছু violation (ξᵢ) allow করা হয়
# Minimize: (1/2)||w||² + C × Σξᵢ
# Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

# C = regularization parameter
# C বড় → কম violation allow → narrow margin → overfit-এর ঝুঁকি
# C ছোট → বেশি violation allow → wide margin → underfit-এর ঝুঁকি

import numpy as np

# C-এর প্রভাব visualize করা যাক
C_values = [0.01, 0.1, 1, 10, 100]
print("C       Effect")
print("-" * 40)
for C in C_values:
    if C < 0.1:
        effect = "Wide margin, many violations allowed"
    elif C < 1:
        effect = "Moderate margin"
    elif C == 1:
        effect = "Balanced (default)"
    elif C < 50:
        effect = "Narrow margin, few violations"
    else:
        effect = "Very narrow margin, hard margin-like"
    print(f"{C:6.2f}  {effect}")</code></pre>

    <h3>৫. Primal Optimization Problem</h3>
    <pre><code"># SVM Primal Form (Soft Margin):
#
# Minimize:   (1/2)||w||² + C Σᵢ ξᵢ
#
# Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ   for all i
#             ξᵢ ≥ 0                     for all i
#
# এটি একটি Quadratic Programming (QP) problem।
# sklearn এটি libsvm/liblinear দিয়ে solve করে।

# Hinge Loss দিয়ে equivalent formulation:
# Minimize: C Σᵢ max(0, 1 - yᵢ(w·xᵢ + b)) + (1/2)||w||²
#                └── Hinge Loss ──┘          └── Regularization ──┘

import numpy as np

def hinge_loss(y_true, decision_scores):
    """y_true ∈ {-1, +1}"""
    return np.maximum(0, 1 - y_true * decision_scores)

# উদাহরণ:
y      = np.array([1, 1, -1, -1])
scores = np.array([2.5, 0.3, -1.8, 0.5])

losses = hinge_loss(y, scores)
print("Per-sample hinge losses:", losses.round(3))
print("Mean hinge loss:", losses.mean().round(4))</code></pre>

    <h3>৬. Python থেকে Scratch-এ SVM</h3>
    <pre><code>import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Simple Gradient Descent SVM (Pegasos algorithm)
class SimpleSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d  = X.shape
        # y ∈ {-1, +1}
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(d)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(n):
                condition = y_[i] * (X[i] @ self.w + self.b) >= 1
                if condition:
                    # Correct margin — only regularize
                    self.w -= self.lr * self.w
                else:
                    # Violation — update w and b
                    self.w += self.lr * (self.C * y_[i] * X[i] - self.w)
                    self.b += self.lr * self.C * y_[i]

    def predict(self, X):
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)

    def score(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        return (self.predict(X) == y_).mean()

# Test
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            random_state=42, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm = SimpleSVM(C=1.0, lr=0.001, epochs=1000)
svm.fit(X_train_s, y_train)
print(f"Scratch SVM Accuracy: {svm.score(X_test_s, y_test):.4f}")</code></pre>

    <h3>৭. Geometric Intuition</h3>
    <table>
      <thead><tr><th>উপাদান</th><th>গণিত</th><th>ভূমিকা</th></tr></thead>
      <tbody>
        <tr><td>Hyperplane</td><td>w·x + b = 0</td><td>Decision boundary</td></tr>
        <tr><td>Margin</td><td>2 / ‖w‖</td><td>দুই class-এর মধ্যে gap — maximize করতে হবে</td></tr>
        <tr><td>Support Vectors</td><td>yᵢ(w·x + b) = 1</td><td>Margin boundary-তে থাকা points — model define করে</td></tr>
        <tr><td>C</td><td>Regularization</td><td>Margin width vs violation tradeoff</td></tr>
        <tr><td>Hinge Loss</td><td>max(0, 1−yf(x))</td><td>Margin violation-এর cost</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>SVM-এর লক্ষ্য</td><td>Maximum margin hyperplane খোঁজা</td></tr>
        <tr><td>Hyperplane</td><td>w·x + b = 0, margin = 2/‖w‖</td></tr>
        <tr><td>Support Vectors</td><td>Margin boundary-র points — শুধু এরাই model নির্ধারণ করে</td></tr>
        <tr><td>Hard Margin</td><td>সব violation নিষিদ্ধ — linearly separable data-র জন্য</td></tr>
        <tr><td>Soft Margin (C)</td><td>C বড় → overfit, C ছোট → underfit</td></tr>
      </tbody>
    </table>
  `,
};
