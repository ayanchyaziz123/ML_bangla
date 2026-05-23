export const dt_2_classification = {
  title: "Decision Tree Classification — সম্পূর্ণ Python গাইড",
  description: "Decision Tree দিয়ে classification, tree visualization, hyperparameter-এর প্রভাব, এবং সঠিক মডেল তৈরির সম্পূর্ণ workflow বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 11,
  slug: "dt-classification-python",
  content: `
    <h3>১. প্রথম Decision Tree</h3>
    <pre><code>from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
class_names   = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# মডেল তৈরি
dt = DecisionTreeClassifier(
    max_depth=3,           # tree-র গভীরতা সীমিত করো
    criterion='gini',      # split measure
    min_samples_split=5,   # split করতে minimum sample
    min_samples_leaf=2,    # leaf-এ minimum sample
    random_state=42,
)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=class_names))</code></pre>

    <h3>২. Tree Visualization</h3>
    <pre><code"># matplotlib দিয়ে tree আঁকো
plt.figure(figsize=(14, 6))
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,          # class অনুযায়ী রঙ
    rounded=True,
    fontsize=9,
    impurity=True,        # Gini দেখাবে
    proportion=False,     # sample count দেখাবে
)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.show()

# Text format-এ দেখো (সহজে পড়া যায়)
rules = export_text(dt, feature_names=list(feature_names))
print(rules)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 145" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Iris Dataset Decision Tree (depth=2)</text>
        <!-- Root -->
        <rect x="165" y="24" width="150" height="30" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="240" y="37" text-anchor="middle" font-size="8" fill="#1e40af">petal length ≤ 2.45?</text>
        <text x="240" y="50" text-anchor="middle" font-size="8" fill="#6b7280">gini=0.667 | n=120</text>
        <!-- Level 2 left (pure setosa) -->
        <rect x="30" y="80" width="130" height="30" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="95" y="93" text-anchor="middle" font-size="8" fill="#166534">✓ Setosa (pure)</text>
        <text x="95" y="106" text-anchor="middle" font-size="8" fill="#6b7280">gini=0.0 | n=40</text>
        <!-- Level 2 right -->
        <rect x="320" y="80" width="150" height="30" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="395" y="93" text-anchor="middle" font-size="8" fill="#92400e">petal width ≤ 1.75?</text>
        <text x="395" y="106" text-anchor="middle" font-size="8" fill="#6b7280">gini=0.495 | n=80</text>
        <!-- Leaves -->
        <rect x="250" y="132" width="110" height="10" rx="3" fill="#dcfce7" stroke="#86efac"/>
        <text x="305" y="141" text-anchor="middle" font-size="7" fill="#166534">Versicolor (n=39)</text>
        <rect x="380" y="132" width="110" height="10" rx="3" fill="#fce7f3" stroke="#f9a8d4"/>
        <text x="435" y="141" text-anchor="middle" font-size="7" fill="#9d174d">Virginica (n=41)</text>
        <!-- Lines -->
        <line x1="215" y1="54" x2="95"  y2="80" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="265" y1="54" x2="395" y2="80" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="370" y1="110" x2="305" y2="132" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="420" y1="110" x2="435" y2="132" stroke="#94a3b8" stroke-width="1.5"/>
        <text x="148" y="70" font-size="8" fill="#16a34a">হ্যাঁ</text>
        <text x="270" y="70" font-size="8" fill="#dc2626">না</text>
      </svg>
    </div>

    <h3>৩. Prediction-এর পথ বোঝা</h3>
    <pre><code>"# একটি sample কোন leaf-এ পড়লো?
sample = X_test[[0]]   # প্রথম test sample

# Node পথ দেখো
node_indicator = dt.decision_path(sample)
leaf_id = dt.apply(sample)

print(f"Sample: {sample[0]}")
print(f"Predicted class: {class_names[dt.predict(sample)[0]]}")
print(f"Probabilities: {dt.predict_proba(sample)[0]}")
print(f"Leaf node ID: {leaf_id[0]}")

# Path দেখো
feature  = dt.tree_.feature
threshold = dt.tree_.threshold
node_ids = node_indicator.indices

for node_id in node_ids:
    if feature[node_id] != -2:   # leaf নয়
        f = feature_names[feature[node_id]]
        t = threshold[node_id]
        val = sample[0, feature[node_id]]
        direction = "≤" if val <= t else ">"
        print(f"  Node {node_id}: {f} ({val:.2f}) {direction} {t:.2f}")</code></pre>

    <h3>৪. Hyperparameter-এর প্রভাব</h3>
    <pre><code">from sklearn.model_selection import cross_val_score

params_to_test = {
    'max_depth':         [1, 2, 3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf':  [1, 2, 5, 10, 20],
}

print("max_depth প্রভাব:")
print(f"{'depth':>8} | {'Train Acc':>10} | {'CV Acc':>8}")
print("-" * 35)

for d in [1, 2, 3, 5, 10, None]:
    dt_d = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_d.fit(X_train, y_train)
    train_acc = dt_d.score(X_train, y_train)
    cv_acc    = cross_val_score(dt_d, X, y, cv=5).mean()
    label = str(d) if d else "None"
    print(f"{label:>8} | {train_acc:>10.3f} | {cv_acc:>8.3f}")
# max_depth=None → Train=1.0, CV কম → Overfitting!</code></pre>

    <h3>৫. Feature Importance</h3>
    <pre><code">import pandas as pd

imp = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': dt.feature_importances_,
}).sort_values('Importance', ascending=False)

print(imp)

plt.figure(figsize=(7, 4))
plt.barh(imp['Feature'], imp['Importance'], color='steelblue')
plt.xlabel('Feature Importance (Gini-based)')
plt.title('Decision Tree Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()</code></pre>

    <h3>৬. Decision Boundary Visualization</h3>
    <pre><code"># ২টি feature দিয়ে decision boundary দেখো
from sklearn.inspection import DecisionBoundaryDisplay

dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_2d.fit(X_train[:, 2:], y_train)   # petal length ও width

fig, ax = plt.subplots(figsize=(7, 5))
DecisionBoundaryDisplay.from_estimator(
    dt_2d, X_train[:, 2:], ax=ax,
    response_method='predict',
    cmap='RdYlGn', alpha=0.4,
)
scatter = ax.scatter(X_train[:, 2], X_train[:, 3],
                     c=y_train, cmap='RdYlGn', edgecolors='k')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Decision Tree Boundary (depth=3)')
plt.show()</code></pre>

    <h3>৭. Complete Pipeline</h3>
    <pre><code">from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Decision Tree-এ Scaling দরকার নেই কিন্তু pipeline-এ রাখা ভালো অভ্যাস
pipe = Pipeline([
    ('model', DecisionTreeClassifier(random_state=42)),
])

param_grid = {
    'model__max_depth':         [3, 5, 7, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf':  [1, 2, 5],
    'model__criterion':         ['gini', 'entropy'],
}

grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"সেরা parameter: {grid.best_params_}")
print(f"Test Accuracy:  {grid.score(X_test, y_test):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>Parameter</th><th>কী করে</th><th>সাধারণ মান</th></tr></thead>
      <tbody>
        <tr><td>max_depth</td><td>tree-র সর্বোচ্চ গভীরতা</td><td>3–10 (None = Overfitting)</td></tr>
        <tr><td>min_samples_split</td><td>split করতে minimum sample</td><td>2–20</td></tr>
        <tr><td>min_samples_leaf</td><td>leaf-এ minimum sample</td><td>1–10</td></tr>
        <tr><td>criterion</td><td>split measure</td><td>'gini' (default)</td></tr>
        <tr><td>Scaling</td><td>দরকার নেই</td><td>tree distance-independent</td></tr>
      </tbody>
    </table>
  `,
};
