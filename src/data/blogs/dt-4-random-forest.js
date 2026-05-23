export const dt_4_random_forest = {
  title: "Random Forest গভীরে — Bagging, OOB Error ও Feature Importance",
  description: "Random Forest কীভাবে Decision Tree-র দুর্বলতা কাটিয়ে ওঠে, Out-of-Bag Error, Permutation Importance, এবং সম্পূর্ণ tuning গাইড বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 12,
  slug: "dt-random-forest-deep",
  content: `
    <h3>১. একটি Tree-র সমস্যা</h3>
    <p>Decision Tree অনেক বেশি sensitive — ডেটা একটু বদলালে সম্পূর্ণ ভিন্ন tree তৈরি হতে পারে। এই High Variance সমস্যার সমাধান: <strong>অনেকগুলো আলাদা tree তৈরি করো, তাদের সিদ্ধান্তের গড় নাও।</strong></p>

    <h3>২. Bagging (Bootstrap Aggregating)</h3>
    <p>Random Forest-এর মূল কৌশল হলো Bagging:</p>
    <ul>
      <li>মূল ডেটা থেকে <strong>replacement সহ random sample</strong> (bootstrap sample) নাও</li>
      <li>প্রতিটি bootstrap sample দিয়ে একটি করে tree তৈরি করো</li>
      <li>Classification: সব tree-র vote গণনা করো (majority wins)</li>
      <li>Regression: সব tree-র prediction-এর গড় নাও</li>
    </ul>
    <pre><code">import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Bagging manually (Random Forest ছাড়া)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,      # প্রতিটি tree-তে 80% data
    max_features=0.8,     # প্রতিটি tree-তে 80% feature
    bootstrap=True,       # replacement সহ sampling
    random_state=42,
)
bagging.fit(X_train, y_train)
print(f"Bagging Accuracy: {bagging.score(X_test, y_test):.4f}")</code></pre>

    <h3>৩. Random Forest = Bagging + Random Feature Selection</h3>
    <p>Random Forest-এ একটি অতিরিক্ত randomness আছে: প্রতিটি split-এ শুধু <strong>random subset of features</strong> বিবেচনা করা হয়। এটি tree-গুলোকে আরও আলাদা (diverse) করে — correlation কমায়।</p>
    <pre><code">rf = RandomForestClassifier(
    n_estimators=100,        # tree সংখ্যা
    max_features='sqrt',     # প্রতি split-এ √p feature (classification default)
    # max_features='log2'    # log₂(p) feature
    # max_features=0.5       # 50% feature
    max_depth=None,          # প্রতিটি tree পূর্ণ grow করো
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,          # Out-of-Bag error হিসাব করো
    random_state=42,
    n_jobs=-1,               # সব CPU core
)
rf.fit(X_train, y_train)

print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test  Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"OOB   Accuracy: {rf.oob_score_:.4f}")   # extra validation!</code></pre>

    <h3>৪. Out-of-Bag (OOB) Error</h3>
    <p>Bootstrap sampling-এ প্রতিটি tree তার training data-র বাইরে থাকা sample দিয়ে নিজেকে validate করতে পারে। এই বিনামূল্যে validation-কে বলে <strong>OOB Score</strong>।</p>
    <pre><code">import pandas as pd
import matplotlib.pyplot as plt

# n_estimators বাড়ার সাথে OOB error কমে
oob_errors = []
n_trees_list = range(10, 210, 10)

for n in n_trees_list:
    rf_n = RandomForestClassifier(
        n_estimators=n, oob_score=True,
        random_state=42, n_jobs=-1
    )
    rf_n.fit(X_train, y_train)
    oob_errors.append(1 - rf_n.oob_score_)

plt.figure(figsize=(8, 4))
plt.plot(n_trees_list, oob_errors, 'o-', color='steelblue')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error')
plt.title('OOB Error vs Number of Trees')
plt.grid(True, alpha=0.3)
plt.show()
# সাধারণত ১০০–২০০টি tree-এ plateau আসে</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 115" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Random Forest — Bootstrap ও OOB</text>
        <!-- Original data -->
        <rect x="15" y="28" width="80" height="75" rx="4" fill="#f3f4f6" stroke="#d1d5db"/>
        <text x="55" y="45" text-anchor="middle" font-size="9" font-weight="600" fill="#374151">Original</text>
        <text x="55" y="58" text-anchor="middle" font-size="8" fill="#6b7280">1,2,3,4,5</text>
        <text x="55" y="70" text-anchor="middle" font-size="8" fill="#6b7280">6,7,8,9,10</text>
        <!-- Bootstrap 1 -->
        <rect x="120" y="28" width="90" height="34" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="165" y="42" text-anchor="middle" font-size="8" font-weight="600" fill="#1e40af">Bootstrap 1</text>
        <text x="165" y="55" text-anchor="middle" font-size="8" fill="#3b82f6">1,3,3,7,9,2</text>
        <rect x="120" y="69" width="90" height="16" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
        <text x="165" y="81" text-anchor="middle" font-size="8" fill="#991b1b">OOB: 4,5,6,8,10</text>
        <!-- Bootstrap 2 -->
        <rect x="230" y="28" width="90" height="34" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="275" y="42" text-anchor="middle" font-size="8" font-weight="600" fill="#166534">Bootstrap 2</text>
        <text x="275" y="55" text-anchor="middle" font-size="8" fill="#16a34a">2,5,5,8,1,10</text>
        <rect x="230" y="69" width="90" height="16" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
        <text x="275" y="81" text-anchor="middle" font-size="8" fill="#991b1b">OOB: 3,4,6,7,9</text>
        <!-- Arrows -->
        <line x1="95" y1="55" x2="118" y2="45" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="95" y1="65" x2="228" y2="45" stroke="#94a3b8" stroke-width="1.5"/>
        <!-- Final -->
        <rect x="350" y="40" width="115" height="35" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="407" y="55" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Vote / Average</text>
        <text x="407" y="70" text-anchor="middle" font-size="8" fill="#d97706">Final Prediction</text>
        <line x1="212" y1="48" x2="348" y2="55" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="322" y1="48" x2="348" y2="55" stroke="#94a3b8" stroke-width="1.5"/>
      </svg>
    </div>

    <h3>৫. Feature Importance-র দুটি পদ্ধতি</h3>
    <pre><code">import pandas as pd

# পদ্ধতি ১: Gini-based (Mean Decrease in Impurity)
gini_imp = pd.DataFrame({
    'Feature':    data.feature_names,
    'MDI':        rf.feature_importances_,
}).sort_values('MDI', ascending=False)

# পদ্ধতি ২: Permutation Importance (বেশি নির্ভরযোগ্য)
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30, random_state=42, n_jobs=-1
)
perm_imp = pd.DataFrame({
    'Feature': data.feature_names,
    'Mean':    perm.importances_mean,
    'Std':     perm.importances_std,
}).sort_values('Mean', ascending=False)

print("Top 5 (Gini-based):")
print(gini_imp.head().to_string(index=False))
print("\\nTop 5 (Permutation):")
print(perm_imp.head().to_string(index=False))

# Permutation importance বেশি নির্ভরযোগ্য কারণ:
# Gini-based high-cardinality feature-কে overestimate করে</code></pre>

    <h3>৬. Hyperparameter Tuning</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [3, 5, 10, 15, None],
    'max_features':     ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split':[2, 5, 10],
}

search = RandomizedSearchCV(
    RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1),
    param_dist,
    n_iter=30, cv=5,
    scoring='roc_auc',
    random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"সেরা params: {search.best_params_}")
print(f"Test AUC:    {search.score(X_test, y_test):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Bagging</td><td>bootstrap sample দিয়ে অনেক tree — variance কমায়</td></tr>
        <tr><td>Random feature</td><td>প্রতি split-এ √p feature — tree-গুলো diverse করে</td></tr>
        <tr><td>OOB Score</td><td>বিনামূল্যে validation — cross-validation-এর বিকল্প</td></tr>
        <tr><td>n_estimators</td><td>বেশি হলে ভালো কিন্তু একটু ধীর, ১০০+ সাধারণত যথেষ্ট</td></tr>
        <tr><td>Permutation Importance</td><td>Gini-based-এর চেয়ে বেশি নির্ভরযোগ্য</td></tr>
      </tbody>
    </table>
  `,
};
