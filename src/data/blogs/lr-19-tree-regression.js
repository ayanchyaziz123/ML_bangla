export const lr_19_tree_regression = {
  title: "Decision Tree ও Random Forest Regression — Tree দিয়ে সংখ্যা ভবিষ্যদ্বাণী",
  description: "Decision Tree কীভাবে split করে, Random Forest কেন ভালো, Feature Importance, এবং Linear Regression-এর সাথে তুলনা — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 11,
  slug: "lr-tree-regression",
  content: `
    <h3>১. Decision Tree Regression কী?</h3>
    <p>লিনিয়ার রিগ্রেশন একটি সরল রেখা আঁকে। কিন্তু যদি ডেটার সম্পর্ক non-linear হয়? <strong>Decision Tree</strong> ডেটাকে একটি গাছের মতো ভাগ করে — প্রতিটি leaf node-এ একটি গড় মান থাকে।</p>
    <p>উদাহরণ: বাড়ির দাম ভবিষ্যদ্বাণীতে —</p>
    <pre><code>আয়তন > 1500 sqft?
├── হ্যাঁ → বয়স < 5 বছর?
│         ├── হ্যাঁ → দাম = 95 লক্ষ  (leaf)
│         └── না  → দাম = 75 লক্ষ  (leaf)
└── না  → এলাকা = "ঢাকা"?
          ├── হ্যাঁ → দাম = 60 লক্ষ  (leaf)
          └── না  → দাম = 40 লক্ষ  (leaf)</code></pre>

    <h3>২. Split কীভাবে হয়?</h3>
    <p>প্রতিটি split-এ tree এমন feature ও threshold বেছে নেয় যা <strong>MSE সবচেয়ে বেশি কমায়</strong>।</p>
    <pre><code">import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10,
                            random_state=42)
dt.fit(X_train, y_train)

print(f"Train R²: {r2_score(y_train, dt.predict(X_train)):.4f}")
print(f"Test  R²: {r2_score(y_test,  dt.predict(X_test)):.4f}")

# max_depth না দিলে পুরো ডেটা মুখস্থ করে ফেলে (Overfitting)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Decision Tree কাঠামো</text>
        <!-- Root -->
        <rect x="180" y="24" width="120" height="24" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="240" y="40" text-anchor="middle" font-size="9" fill="#1e40af">আয়তন > 1500?</text>
        <!-- Level 2 left -->
        <rect x="60" y="70" width="110" height="24" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="115" y="86" text-anchor="middle" font-size="9" fill="#1e40af">বয়স &lt; 5?</text>
        <!-- Level 2 right -->
        <rect x="310" y="70" width="110" height="24" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="365" y="86" text-anchor="middle" font-size="9" fill="#1e40af">এলাকা=ঢাকা?</text>
        <!-- Leaves -->
        <rect x="20"  y="112" width="70" height="14" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="55"  y="123" text-anchor="middle" font-size="8" fill="#16a34a">95 লক্ষ</text>
        <rect x="105" y="112" width="70" height="14" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="140" y="123" text-anchor="middle" font-size="8" fill="#16a34a">75 লক্ষ</text>
        <rect x="270" y="112" width="70" height="14" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="305" y="123" text-anchor="middle" font-size="8" fill="#16a34a">60 লক্ষ</text>
        <rect x="390" y="112" width="70" height="14" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="425" y="123" text-anchor="middle" font-size="8" fill="#16a34a">40 লক্ষ</text>
        <!-- Lines -->
        <line x1="210" y1="48" x2="115" y2="70" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="270" y1="48" x2="365" y2="70" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="95"  y1="94" x2="55"  y2="112" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="135" y1="94" x2="140" y2="112" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="345" y1="94" x2="305" y2="112" stroke="#94a3b8" stroke-width="1.5"/>
        <line x1="385" y1="94" x2="425" y2="112" stroke="#94a3b8" stroke-width="1.5"/>
      </svg>
    </div>

    <h3>৩. Decision Tree-র সমস্যা</h3>
    <p>একটি tree <strong>Overfitting</strong>-প্রবণ। সমাধান: অনেকগুলো tree একসাথে বানাও — এটাই <strong>Random Forest</strong>।</p>

    <h3>৪. Random Forest Regression</h3>
    <p>Random Forest দুটি কৌশল ব্যবহার করে:</p>
    <ul>
      <li><strong>Bagging:</strong> প্রতিটি tree আলাদা random sample দিয়ে train হয়</li>
      <li><strong>Random feature subset:</strong> প্রতিটি split-এ random কিছু feature বিবেচনা করে</li>
    </ul>
    <p>সবশেষে সব tree-র prediction-এর <strong>গড়</strong> নেওয়া হয়।</p>
    <pre><code">from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,      # কতটি tree
    max_depth=10,          # প্রতিটি tree-র গভীরতা
    min_samples_leaf=5,
    max_features='sqrt',   # প্রতি split-এ √p ফিচার বিবেচনা
    random_state=42,
    n_jobs=-1              # সব CPU core ব্যবহার করো
)
rf.fit(X_train, y_train)

print(f"Train R²: {r2_score(y_train, rf.predict(X_train)):.4f}")
print(f"Test  R²: {r2_score(y_test,  rf.predict(X_test)):.4f}")</code></pre>

    <h3>৫. Feature Importance</h3>
    <pre><code">import pandas as pd
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    'Feature':   feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()</code></pre>

    <h3>৬. Hyperparameter Tuning</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators':   [50, 100, 200],
    'max_depth':      [5, 10, 15, None],
    'min_samples_leaf': [1, 5, 10],
    'max_features':   ['sqrt', 'log2'],
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"সেরা parameter: {search.best_params_}")
print(f"Test R²:        {search.score(X_test, y_test):.4f}")</code></pre>

    <h3>৭. Linear Regression vs Decision Tree vs Random Forest</h3>
    <table>
      <thead><tr><th></th><th>Linear Regression</th><th>Decision Tree</th><th>Random Forest</th></tr></thead>
      <tbody>
        <tr><td><strong>Non-linear data</strong></td><td>খারাপ</td><td>ভালো</td><td>খুব ভালো</td></tr>
        <tr><td><strong>Interpretability</strong></td><td>সহজ</td><td>মাঝামাঝি</td><td>কঠিন</td></tr>
        <tr><td><strong>Overfitting</strong></td><td>কম</td><td>বেশি</td><td>কম (ensemble)</td></tr>
        <tr><td><strong>Speed</strong></td><td>দ্রুত</td><td>দ্রুত</td><td>ধীর</td></tr>
        <tr><td><strong>Scaling দরকার?</strong></td><td>হ্যাঁ</td><td>না</td><td>না</td></tr>
        <tr><td><strong>Feature importance</strong></td><td>coefficient</td><td>node importance</td><td>averaged importance</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Decision Tree</td><td>MSE কমিয়ে split করে, Overfitting-প্রবণ</td></tr>
        <tr><td>max_depth</td><td>tree-র গভীরতা সীমিত রাখো — Overfitting এড়াতে</td></tr>
        <tr><td>Random Forest</td><td>অনেক tree-র গড় — variance কমায়, accuracy বাড়ায়</td></tr>
        <tr><td>Feature Importance</td><td>কোন ফিচার বেশি গুরুত্বপূর্ণ তা জানা যায়</td></tr>
        <tr><td>Scaling দরকার নেই</td><td>Tree-based model distance-independent</td></tr>
      </tbody>
    </table>
  `,
};
