export const dt_5_adaboost = {
  title: "AdaBoost — দুর্বল Learner থেকে শক্তিশালী মডেল",
  description: "AdaBoost কীভাবে ভুলের উপর focus করে, sample weight update, Decision Stump, এবং Random Forest-এর সাথে পার্থক্য — বাংলায় সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 11,
  slug: "dt-adaboost",
  content: `
    <h3>১. AdaBoost-এর মূল ধারণা</h3>
    <p>Random Forest-এ tree-গুলো স্বাধীনভাবে তৈরি হয়। <strong>AdaBoost (Adaptive Boosting)</strong>-এ tree-গুলো ক্রমানুসারে তৈরি হয় — প্রতিটি নতুন tree আগের tree-র ভুলের দিকে বেশি মনোযোগ দেয়।</p>
    <p>কৌশল: ভুলভাবে classified sample-গুলোর <strong>weight বাড়াও</strong> — পরের tree সেগুলো ঠিক করতে বেশি চেষ্টা করবে।</p>

    <h3>২. AdaBoost-এর ধাপগুলো</h3>
    <pre><code>import numpy as np

# সরলীকৃত AdaBoost algorithm:

# ধাপ ১: সব sample-এর weight সমান (1/n)
n = 10
weights = np.ones(n) / n

# ধাপ ২: প্রতিটি iteration-এ:
#   a) weighted ডেটায় একটি weak learner (decision stump) fit করো
#   b) এই learner-এর error হিসাব করো
#   c) learner-এর গুরুত্ব (alpha) হিসাব করো
#   d) weight update করো: ভুল → weight বাড়াও, ঠিক → কমাও

# error হিসাব:
# ε = Σ(wᵢ × I(yᵢ ≠ ŷᵢ)) / Σwᵢ

# learner গুরুত্ব:
# α = 0.5 × log((1 − ε) / ε)

# weight update:
# wᵢ = wᵢ × exp(α × I(yᵢ ≠ ŷᵢ))   ← ভুল হলে বাড়ে
# wᵢ = wᵢ × exp(−α × I(yᵢ = ŷᵢ))  ← ঠিক হলে কমে

# ধাপ ৩: চূড়ান্ত prediction
# F(x) = sign(Σ αₜ × hₜ(x))
# weighted majority vote</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">AdaBoost — ভুলের উপর ফোকাস</text>
        <!-- Round 1 -->
        <rect x="15" y="28" width="85" height="75" rx="4" fill="#f3f4f6" stroke="#d1d5db"/>
        <text x="57" y="44" text-anchor="middle" font-size="8" font-weight="600" fill="#374151">Round 1</text>
        <circle cx="35" cy="58" r="4" fill="#1e40af"/>
        <circle cx="55" cy="58" r="4" fill="#1e40af"/>
        <circle cx="75" cy="58" r="4" fill="#dc2626" stroke="#dc2626" stroke-width="2"/>
        <circle cx="35" cy="72" r="4" fill="#dc2626"/>
        <circle cx="55" cy="72" r="4" fill="#1e40af"/>
        <circle cx="75" cy="72" r="7" fill="none" stroke="#dc2626" stroke-width="2"/>
        <text x="57" y="93" text-anchor="middle" font-size="7" fill="#6b7280">ভুল sample চিহ্নিত</text>
        <!-- Arrow -->
        <text x="108" y="70" text-anchor="middle" font-size="14" fill="#94a3b8">→</text>
        <!-- Round 2 -->
        <rect x="120" y="28" width="85" height="75" rx="4" fill="#f3f4f6" stroke="#d1d5db"/>
        <text x="162" y="44" text-anchor="middle" font-size="8" font-weight="600" fill="#374151">Round 2</text>
        <circle cx="140" cy="58" r="4" fill="#1e40af"/>
        <circle cx="160" cy="58" r="4" fill="#1e40af"/>
        <circle cx="180" cy="58" r="7" fill="#dc2626"/>
        <circle cx="140" cy="72" r="7" fill="#dc2626"/>
        <circle cx="160" cy="72" r="4" fill="#1e40af"/>
        <text x="162" y="93" text-anchor="middle" font-size="7" fill="#6b7280">ভুলের weight বড়</text>
        <text x="213" y="70" text-anchor="middle" font-size="14" fill="#94a3b8">→</text>
        <!-- Round 3 -->
        <rect x="225" y="28" width="85" height="75" rx="4" fill="#f3f4f6" stroke="#d1d5db"/>
        <text x="267" y="44" text-anchor="middle" font-size="8" font-weight="600" fill="#374151">Round 3</text>
        <circle cx="245" cy="65" r="4" fill="#1e40af"/>
        <circle cx="265" cy="65" r="4" fill="#1e40af"/>
        <circle cx="285" cy="65" r="4" fill="#dc2626"/>
        <text x="267" y="93" text-anchor="middle" font-size="7" fill="#6b7280">আগের ভুল ঠিক</text>
        <text x="318" y="70" text-anchor="middle" font-size="14" fill="#94a3b8">→</text>
        <!-- Final -->
        <rect x="335" y="40" width="130" height="45" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="400" y="58" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Weighted Vote</text>
        <text x="400" y="72" text-anchor="middle" font-size="8" fill="#16a34a">α₁h₁ + α₂h₂ + α₃h₃</text>
      </svg>
    </div>

    <h3>৩. Python দিয়ে AdaBoost</h3>
    <pre><code">from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# AdaBoost with Decision Stump (depth=1)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision Stump
    n_estimators=200,        # কতটি weak learner
    learning_rate=0.5,       # প্রতিটি learner-এর contribution
    algorithm='SAMME',       # multiclass support
    random_state=42,
)
ada.fit(X_train, y_train)

print(f"Train Accuracy: {ada.score(X_train, y_train):.4f}")
print(f"Test  Accuracy: {ada.score(X_test,  y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, ada.predict_proba(X_test)[:,1]):.4f}")
print(classification_report(y_test, ada.predict(X_test),
      target_names=['Benign', 'Malignant']))</code></pre>

    <h3>৪. n_estimators ও learning_rate tradeoff</h3>
    <pre><code">results = []
for n_est in [10, 50, 100, 200, 500]:
    for lr in [0.1, 0.5, 1.0]:
        ada_t = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n_est, learning_rate=lr,
            random_state=42,
        )
        cv = cross_val_score(ada_t, X, y, cv=5, scoring='roc_auc').mean()
        results.append({'n_estimators': n_est, 'learning_rate': lr, 'CV AUC': cv})

import pandas as pd
df_r = pd.DataFrame(results).pivot(
    index='n_estimators', columns='learning_rate', values='CV AUC'
)
print(df_r.round(4))
# learning_rate ছোট → বেশি n_estimators দরকার (কিন্তু generalization ভালো)</code></pre>

    <h3>৫. Staged Prediction — Error কমার পথ দেখো</h3>
    <pre><code">import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# প্রতিটি step-এর পর accuracy দেখো
train_errors, test_errors = [], []

for y_pred_train, y_pred_test in zip(
    ada.staged_predict(X_train),
    ada.staged_predict(X_test)
):
    train_errors.append(1 - accuracy_score(y_train, y_pred_train))
    test_errors.append(1 - accuracy_score(y_test,  y_pred_test))

plt.figure(figsize=(8, 4))
plt.plot(train_errors, label='Train Error')
plt.plot(test_errors,  label='Test Error')
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('AdaBoost — Staged Prediction Error')
plt.legend()
plt.show()</code></pre>

    <h3>৬. AdaBoost vs Random Forest</h3>
    <table>
      <thead><tr><th></th><th>AdaBoost</th><th>Random Forest</th></tr></thead>
      <tbody>
        <tr><td><strong>Tree তৈরির ধরন</strong></td><td>Sequential (ক্রমানুসারে)</td><td>Parallel (স্বাধীনভাবে)</td></tr>
        <tr><td><strong>Tree-র আকার</strong></td><td>Stump (depth=1) — অনেক</td><td>গভীর tree — কম</td></tr>
        <tr><td><strong>Bias-Variance</strong></td><td>Bias কমায় (boosting)</td><td>Variance কমায় (bagging)</td></tr>
        <tr><td><strong>Outlier sensitivity</strong></td><td>বেশি sensitive</td><td>কম sensitive</td></tr>
        <tr><td><strong>Speed</strong></td><td>ধীর (sequential)</td><td>দ্রুত (parallel)</td></tr>
        <tr><td><strong>Overfitting</strong></td><td>কম (early stopping কার্যকর)</td><td>কম (ensemble effect)</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>AdaBoost</td><td>ভুল sample-এর weight বাড়িয়ে পরের tree focus করায়</td></tr>
        <tr><td>Decision Stump</td><td>depth=1 tree — weak learner হিসেবে সবচেয়ে জনপ্রিয়</td></tr>
        <tr><td>learning_rate</td><td>ছোট → ধীরে শেখা কিন্তু ভালো generalization</td></tr>
        <tr><td>staged_predict</td><td>প্রতিটি step-এর error দেখে সেরা n_estimators বেছে নাও</td></tr>
        <tr><td>Outlier</td><td>AdaBoost outlier-এ sensitive — আগে outlier সরাও</td></tr>
      </tbody>
    </table>
  `,
};
