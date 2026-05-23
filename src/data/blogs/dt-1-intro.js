export const dt_1_intro = {
  title: "Decision Tree-র গণিত — Gini, Entropy ও Information Gain",
  description: "Decision Tree কীভাবে split বেছে নেয়, Gini Impurity ও Entropy-র পার্থক্য, Information Gain হিসাব — বাংলায় গণিত ও Python কোড সহ।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 12,
  slug: "dt-gini-entropy-information-gain",
  content: `
    <h3>১. Decision Tree কীভাবে সিদ্ধান্ত নেয়?</h3>
    <p>Decision Tree হলো একটি গাছের মতো কাঠামো — প্রতিটি node-এ একটি প্রশ্ন জিজ্ঞেস করা হয়, উত্তর অনুযায়ী বাম বা ডান ডালে যাওয়া হয়, এবং leaf node-এ পৌঁছালে prediction দেওয়া হয়।</p>
    <p>কিন্তু tree কোন feature-এ এবং কোন threshold-এ split করবে তা কীভাবে ঠিক করে? উত্তর: <strong>Impurity measure</strong> — Gini বা Entropy।</p>

    <h3>২. Impurity কী?</h3>
    <p>একটি node-এ যদি সব sample একই class-এর হয় → impurity = 0 (খাঁটি)। যদি সব class সমান ভাগে থাকে → impurity সর্বোচ্চ।</p>
    <pre><code>import numpy as np

# উদাহরণ node:
# সব class-1 → impurity = 0 (খাঁটি)
# 50% class-0, 50% class-1 → impurity সর্বোচ্চ

def gini(p):
    """p = class-1 এর অনুপাত, class-0 এর অনুপাত = 1-p"""
    return 1 - (p**2 + (1-p)**2)

def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"p={p:.2f} | Gini={gini(p):.3f} | Entropy={entropy(p):.3f}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Gini vs Entropy — Impurity তুলনা</text>
        <line x1="40" y1="110" x2="440" y2="110" stroke="#e5e7eb" stroke-width="1"/>
        <line x1="40" y1="20"  x2="40"  y2="110" stroke="#e5e7eb" stroke-width="1"/>
        <!-- Gini curve (max 0.5 at p=0.5) -->
        <path d="M 40 110 Q 140 30 240 30 Q 340 30 440 110" stroke="#1e40af" stroke-width="2" fill="none"/>
        <!-- Entropy curve (max 1.0 at p=0.5) -->
        <path d="M 40 110 Q 100 10 240 10 Q 380 10 440 110" stroke="#dc2626" stroke-width="2" fill="none" stroke-dasharray="5,3"/>
        <circle cx="240" cy="30" r="3" fill="#1e40af"/>
        <circle cx="240" cy="10" r="3" fill="#dc2626"/>
        <text x="260" y="34" font-size="8" fill="#1e40af">Gini max=0.5</text>
        <text x="260" y="14" font-size="8" fill="#dc2626">Entropy max=1.0</text>
        <text x="40"  y="120" font-size="8" fill="#6b7280">p=0</text>
        <text x="230" y="120" font-size="8" fill="#6b7280">p=0.5</text>
        <text x="425" y="120" font-size="8" fill="#6b7280">p=1</text>
      </svg>
    </div>

    <h3>৩. Gini Impurity</h3>
    <pre><code># Gini Impurity:
# Gini(t) = 1 − Σ pᵢ²
# যেখানে pᵢ = class i-এর অনুপাত

# বহু class-এর জন্য:
def gini_multiclass(class_counts):
    total = sum(class_counts)
    return 1 - sum((c/total)**2 for c in class_counts)

# উদাহরণ:
node_A = [10, 0]      # সব class-1 → pure
node_B = [5, 5]       # সমান → সর্বোচ্চ impure
node_C = [7, 3]       # মাঝামাঝি

print(f"Node A Gini: {gini_multiclass(node_A):.3f}")  # 0.000
print(f"Node B Gini: {gini_multiclass(node_B):.3f}")  # 0.500
print(f"Node C Gini: {gini_multiclass(node_C):.3f}")  # 0.420</code></pre>

    <h3>৪. Entropy ও Information Gain</h3>
    <pre><code># Entropy:
# H(t) = −Σ pᵢ × log₂(pᵢ)

def entropy_multiclass(class_counts):
    total = sum(class_counts)
    result = 0
    for c in class_counts:
        if c > 0:
            p = c / total
            result -= p * np.log2(p)
    return result

# Information Gain:
# IG = H(parent) − [weighted avg of children entropy]

def information_gain(parent, left_child, right_child):
    n = sum(parent)
    n_l = sum(left_child)
    n_r = sum(right_child)

    H_parent = entropy_multiclass(parent)
    H_left   = entropy_multiclass(left_child)
    H_right  = entropy_multiclass(right_child)

    return H_parent - (n_l/n * H_left + n_r/n * H_right)

# উদাহরণ split:
parent = [10, 10]       # 10 class-0, 10 class-1
left   = [8, 2]         # বাম child
right  = [2, 8]         # ডান child

ig = information_gain(parent, left, right)
print(f"Information Gain: {ig:.4f}")   # উচ্চ IG = ভালো split</code></pre>

    <h3>৫. Tree কীভাবে সেরা Split খোঁজে?</h3>
    <pre><code">import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

# manually: একটি feature-এর সেরা threshold খোঁজা
feature_idx = 2   # petal length
feature_vals = X[:, feature_idx]

best_ig  = -1
best_thr = None

thresholds = np.unique(feature_vals)
for thr in thresholds:
    left_mask  = feature_vals <= thr
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        continue

    parent      = [np.sum(y == c) for c in np.unique(y)]
    left_child  = [np.sum(y[left_mask]  == c) for c in np.unique(y)]
    right_child = [np.sum(y[right_mask] == c) for c in np.unique(y)]

    ig = information_gain(parent, left_child, right_child)
    if ig > best_ig:
        best_ig  = ig
        best_thr = thr

print(f"সেরা threshold: {best_thr:.2f}")
print(f"সেরা Information Gain: {best_ig:.4f}")</code></pre>

    <h3>৬. Gini vs Entropy — কোনটি ব্যবহার করবো?</h3>
    <table>
      <thead><tr><th></th><th>Gini Impurity</th><th>Entropy</th></tr></thead>
      <tbody>
        <tr><td><strong>গণনা</strong></td><td>দ্রুত (log নেই)</td><td>ধীর (log₂ আছে)</td></tr>
        <tr><td><strong>Range</strong></td><td>0 থেকে 0.5 (binary)</td><td>0 থেকে 1.0 (binary)</td></tr>
        <tr><td><strong>Sensitivity</strong></td><td>minority class-এ কম সংবেদনশীল</td><td>সব class সমানভাবে বিবেচনা</td></tr>
        <tr><td><strong>sklearn default</strong></td><td>হ্যাঁ (criterion='gini')</td><td>criterion='entropy'</td></tr>
        <tr><td><strong>ব্যবহারিক পার্থক্য</strong></td><td colspan="2">বেশিরভাগ ক্ষেত্রে একই ফলাফল — Gini ব্যবহার করো</td></tr>
      </tbody>
    </table>
    <pre><code">from sklearn.tree import DecisionTreeClassifier

# Gini (default)
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
# Entropy
dt_ent  = DecisionTreeClassifier(criterion='entropy', random_state=42)
# Log Loss (sklearn 1.1+)
dt_log  = DecisionTreeClassifier(criterion='log_loss', random_state=42)</code></pre>

    <h3>৭. Regression-এ MSE Reduction</h3>
    <pre><code"># Regression tree-তে Gini/Entropy নয়, MSE কমানো হয়:
# Split score = MSE(parent) − [weighted avg of children MSE]

# sklearn:
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='squared_error')  # MSE
dt_abs = DecisionTreeRegressor(criterion='absolute_error') # MAE</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>ধারণা</th><th>সূত্র</th><th>অর্থ</th></tr></thead>
      <tbody>
        <tr><td>Gini Impurity</td><td>1 − Σpᵢ²</td><td>node কতটা মিশ্রিত, 0=খাঁটি</td></tr>
        <tr><td>Entropy</td><td>−Σpᵢ log₂pᵢ</td><td>অনিশ্চয়তার পরিমাপ</td></tr>
        <tr><td>Information Gain</td><td>H(parent) − weighted H(children)</td><td>split কতটা ভালো</td></tr>
        <tr><td>সেরা split</td><td>সর্বোচ্চ IG / সর্বনিম্ন Gini</td><td>প্রতিটি node-এ এটি খোঁজা হয়</td></tr>
        <tr><td>Regression</td><td>MSE reduction</td><td>Gini নয়, MSE কমায়</td></tr>
      </tbody>
    </table>
  `,
};
