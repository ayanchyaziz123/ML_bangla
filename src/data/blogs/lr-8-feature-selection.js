export const lr_8_feature_selection = {
  title: "Feature Selection — কোন ফিচার রাখবো, কোনটা বাদ দেবো?",
  description: "লিনিয়ার রিগ্রেশনে সেরা ফিচার বেছে নেওয়ার কৌশল — Correlation Analysis, Forward Selection, Backward Elimination, এবং Lasso-based selection বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 9,
  slug: "lr-feature-selection",
  content: `
    <h3>১. Feature Selection কেন দরকার?</h3>
    <p>বেশি ফিচার মানেই ভালো মডেল নয়। অপ্রয়োজনীয় ফিচার থাকলে:</p>
    <ul>
      <li>মডেল overfitting করে</li>
      <li>ট্রেনিং ধীর হয়</li>
      <li>Multicollinearity বাড়ে</li>
      <li>মডেলের ব্যাখ্যা কঠিন হয়</li>
    </ul>
    <p>লক্ষ্য: সবচেয়ে কম ফিচার দিয়ে সবচেয়ে ভালো মডেল বানানো।</p>

    <h3>২. পদ্ধতি ১ — Correlation Analysis</h3>
    <p>প্রথমে দেখো কোন ফিচার target-এর সাথে বেশি correlated:</p>
    <pre><code>import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Target-এর সাথে correlation
corr_with_target = df.corr()['price_lakh'].sort_values(ascending=False)
print(corr_with_target)

# |correlation| > 0.5 → ভালো ফিচার
# |correlation| < 0.1 → বাদ দেওয়ার কথা ভাবো

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">ফিচার Selection পদ্ধতি</text>
        <rect x="15"  y="30" width="100" height="50" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="65"  y="53" text-anchor="middle" font-size="9" font-weight="600" fill="#1e3a8a">Filter</text>
        <text x="65"  y="66" text-anchor="middle" font-size="8" fill="#1e3a8a">Correlation,</text>
        <text x="65"  y="77" text-anchor="middle" font-size="8" fill="#1e3a8a">VIF, Chi²</text>
        <rect x="190" y="30" width="100" height="50" rx="5" fill="#fef3c7" stroke="#f59e0b"/>
        <text x="240" y="53" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Wrapper</text>
        <text x="240" y="66" text-anchor="middle" font-size="8" fill="#92400e">Forward/Backward</text>
        <text x="240" y="77" text-anchor="middle" font-size="8" fill="#92400e">Selection</text>
        <rect x="365" y="30" width="100" height="50" rx="5" fill="#d1fae5" stroke="#6ee7b7"/>
        <text x="415" y="53" text-anchor="middle" font-size="9" font-weight="600" fill="#065f46">Embedded</text>
        <text x="415" y="66" text-anchor="middle" font-size="8" fill="#065f46">Lasso, Ridge</text>
        <text x="415" y="77" text-anchor="middle" font-size="8" fill="#065f46">(মডেলের মধ্যে)</text>
        <text x="65"  y="100" text-anchor="middle" font-size="8" fill="#6b7280">দ্রুত, সরল</text>
        <text x="240" y="100" text-anchor="middle" font-size="8" fill="#6b7280">নির্ভরযোগ্য</text>
        <text x="415" y="100" text-anchor="middle" font-size="8" fill="#6b7280">সেরা মানের</text>
      </svg>
    </div>

    <h3>৩. পদ্ধতি ২ — Forward Selection</h3>
    <p>শূন্য থেকে শুরু করে একে একে সবচেয়ে ভালো ফিচার যোগ করা হয়।</p>
    <pre><code">from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

def forward_selection(X, y, threshold=0.01):
    remaining = list(X.columns)
    selected = []
    current_score = 0

    while remaining:
        scores = []
        for feature in remaining:
            features = selected + [feature]
            model = LinearRegression()
            score = cross_val_score(
                model, X[features], y, cv=5, scoring='r2'
            ).mean()
            scores.append((score, feature))

        scores.sort(reverse=True)
        best_score, best_feature = scores[0]

        if best_score - current_score > threshold:
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_score = best_score
            print(f"যোগ হলো: {best_feature} (R²={best_score:.4f})")
        else:
            break

    return selected

selected_features = forward_selection(X, y)
print(f"\\nনির্বাচিত ফিচার: {selected_features}")</code></pre>

    <h3>৪. পদ্ধতি ৩ — Backward Elimination</h3>
    <p>সব ফিচার দিয়ে শুরু করে একে একে সবচেয়ে কম গুরুত্বপূর্ণ ফিচার বাদ দেওয়া হয়।</p>
    <pre><code">import statsmodels.api as sm

def backward_elimination(X, y, threshold=0.05):
    features = list(X.columns)

    while True:
        X_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_const).fit()
        p_values = model.pvalues[1:]  # constant বাদ দিয়ে

        max_p = p_values.max()
        if max_p > threshold:
            worst = p_values.idxmax()
            features.remove(worst)
            print(f"বাদ দেওয়া হলো: {worst} (p={max_p:.4f})")
        else:
            break

    print(f"\\nচূড়ান্ত ফিচার: {features}")
    return features

# p-value > 0.05 → statistically significant না → বাদ
final_features = backward_elimination(X, y)</code></pre>

    <h3>৫. পদ্ধতি ৪ — Lasso-based Selection (সেরা পদ্ধতি)</h3>
    <pre><code">from sklearn.linear_model import LassoCV
import numpy as np

# Lasso স্বয়ংক্রিয়ভাবে অপ্রয়োজনীয় ফিচারের coefficient শূন্য করে
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)

# কোন ফিচার টিকে আছে?
selected = X.columns[lasso_cv.coef_ != 0].tolist()
dropped  = X.columns[lasso_cv.coef_ == 0].tolist()

print(f"নির্বাচিত ({len(selected)}): {selected}")
print(f"বাদ পড়া  ({len(dropped)}):  {dropped}")
print(f"Best alpha: {lasso_cv.alpha_:.4f}")</code></pre>

    <h3>৬. sklearn-এর RFE (Recursive Feature Elimination)</h3>
    <pre><code">from sklearn.feature_selection import RFE

model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=3)  # সেরা ৩টি ফিচার
rfe.fit(X_scaled, y)

print("নির্বাচিত ফিচার:")
for feature, selected, rank in zip(X.columns, rfe.support_, rfe.ranking_):
    status = "✅" if selected else f"❌ (rank {rank})"
    print(f"  {feature}: {status}")</code></pre>

    <h3>৭. কোন পদ্ধতি কখন ব্যবহার করবে?</h3>
    <table>
      <thead><tr><th>পদ্ধতি</th><th>সুবিধা</th><th>কখন ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>Correlation</td><td>দ্রুত, বুঝতে সহজ</td><td>প্রাথমিক EDA-তে</td></tr>
        <tr><td>Forward Selection</td><td>ফিচার কম থেকে শুরু</td><td>ফিচার সংখ্যা মাঝারি হলে</td></tr>
        <tr><td>Backward Elimination</td><td>p-value ব্যাখ্যাযোগ্য</td><td>Statistical inference দরকার হলে</td></tr>
        <tr><td>Lasso</td><td>স্বয়ংক্রিয়, নির্ভরযোগ্য</td><td>অনেক ফিচার থাকলে (সেরা পছন্দ)</td></tr>
        <tr><td>RFE</td><td>নির্দিষ্ট সংখ্যা বেছে নেয়</td><td>ঠিক কতটা ফিচার চাই জানা থাকলে</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Feature Selection কেন</td><td>Overfitting কমাতে, মডেল সহজ রাখতে</td></tr>
        <tr><td>Filter Method</td><td>Correlation — দ্রুত প্রাথমিক চেক</td></tr>
        <tr><td>Wrapper Method</td><td>Forward/Backward — নির্ভরযোগ্য কিন্তু ধীর</td></tr>
        <tr><td>Embedded Method</td><td>Lasso — সেরা পদ্ধতি, স্বয়ংক্রিয়</td></tr>
        <tr><td>সাধারণ নিয়ম</td><td>Lasso দিয়ে শুরু করো, তারপর correlation চেক করো</td></tr>
      </tbody>
    </table>
  `,
};
