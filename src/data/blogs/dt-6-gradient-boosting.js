export const dt_6_gradient_boosting = {
  title: "Gradient Boosting গভীরে — Residual থেকে শেখা",
  description: "Gradient Boosting কীভাবে residual fit করে, Loss Function, sklearn GradientBoosting vs HistGradientBoosting, এবং LightGBM পরিচিতি বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 12,
  slug: "dt-gradient-boosting",
  content: `
    <h3>১. Gradient Boosting-এর মূল ধারণা</h3>
    <p>AdaBoost sample weight বদলায়। <strong>Gradient Boosting</strong> ভিন্ন পথে যায়: প্রতিটি নতুন tree আগের সব tree-র <strong>residual (ভুলের পরিমাণ)</strong> fit করার চেষ্টা করে।</p>
    <pre><code">import numpy as np

# সরলীকৃত Gradient Boosting (Regression):

np.random.seed(42)
X = np.linspace(0, 5, 30).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.3 * np.random.randn(30)

# ধাপ ১: প্রাথমিক prediction = mean(y)
F0 = np.full(len(y), y.mean())
print(f"F0 (mean) = {F0[0]:.3f}")

# ধাপ ২: প্রতিটি iteration-এ:
from sklearn.tree import DecisionTreeRegressor

learning_rate = 0.3
F = F0.copy()
trees = []

for t in range(5):
    # Residual = আসল − এখন পর্যন্তের prediction
    residual = y - F

    # Residual fit করো একটি shallow tree দিয়ে
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, residual)
    trees.append(tree)

    # Update: পুরনো prediction + learning_rate × নতুন tree
    F += learning_rate * tree.predict(X)
    mse = np.mean((y - F)**2)
    print(f"Iteration {t+1}: MSE = {mse:.4f}")</code></pre>

    <h3>২. Gradient Boosting = Gradient Descent in Function Space</h3>
    <p>নাম "Gradient Boosting" কেন? কারণ residual হলো আসলে Loss Function-এর negative gradient।</p>
    <pre><code">
# MSE Loss এর জন্য:
# L(y, F) = (y − F)² / 2
# −∂L/∂F = y − F   ← এটাই residual!

# Log Loss (classification)-এর জন্য:
# −∂L/∂F = y − p̂  ← probability residual

# তাই gradient boosting সব differentiable loss-এ কাজ করে:
# - Regression: MSE, MAE, Huber
# - Classification: Log Loss
# - Ranking: LambdaRank

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    loss='log_loss',     # classification loss
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,       # Stochastic GB (overfitting কমায়)
    random_state=42,
)</code></pre>

    <h3>৩. sklearn GradientBoostingClassifier</h3>
    <pre><code">from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gbc = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=5,
    subsample=0.8,        # Stochastic Gradient Boosting
    max_features='sqrt',  # Random subspace
    random_state=42,
)
gbc.fit(X_train, y_train)

print(f"Train Accuracy: {gbc.score(X_train, y_train):.4f}")
print(f"Test  Accuracy: {gbc.score(X_test,  y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, gbc.predict_proba(X_test)[:,1]):.4f}")

# Training loss curve দেখো
import matplotlib.pyplot as plt
plt.plot(gbc.train_score_, label='Train Loss')
if hasattr(gbc, 'oob_improvement_'):
    plt.plot(np.cumsum(gbc.oob_improvement_), label='OOB')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Gradient Boosting — Residual fit করার পথ</text>
        <rect x="15" y="28" width="85" height="35" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="57" y="44" text-anchor="middle" font-size="8" font-weight="600" fill="#1e40af">F₀ = mean(y)</text>
        <text x="57" y="58" text-anchor="middle" font-size="7" fill="#6b7280">প্রাথমিক guess</text>
        <text x="107" y="50" font-size="10" fill="#94a3b8">→</text>
        <rect x="118" y="28" width="85" height="35" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="160" y="41" text-anchor="middle" font-size="7" fill="#92400e">residual₁= y−F₀</text>
        <text x="160" y="53" text-anchor="middle" font-size="7" fill="#d97706">tree₁ fit করো</text>
        <text x="160" y="65" text-anchor="middle" font-size="7" fill="#d97706">F₁ = F₀ + α×h₁</text>
        <text x="212" y="50" font-size="10" fill="#94a3b8">→</text>
        <rect x="223" y="28" width="85" height="35" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="265" y="41" text-anchor="middle" font-size="7" fill="#92400e">residual₂= y−F₁</text>
        <text x="265" y="53" text-anchor="middle" font-size="7" fill="#d97706">tree₂ fit করো</text>
        <text x="265" y="65" text-anchor="middle" font-size="7" fill="#d97706">F₂ = F₁ + α×h₂</text>
        <text x="315" y="50" font-size="10" fill="#94a3b8">···</text>
        <rect x="340" y="28" width="125" height="35" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="402" y="44" text-anchor="middle" font-size="8" font-weight="600" fill="#166534">F_T = F₀ + α Σhₜ</text>
        <text x="402" y="58" text-anchor="middle" font-size="7" fill="#16a34a">চূড়ান্ত prediction</text>
        <text x="240" y="98" text-anchor="middle" font-size="9" fill="#6b7280">প্রতিটি tree আগের ভুল (residual) কমাতে কাজ করে</text>
      </svg>
    </div>

    <h3>৪. HistGradientBoosting — দ্রুত বড় ডেটায়</h3>
    <pre><code">from sklearn.ensemble import HistGradientBoostingClassifier
import time

# HistGBM: LightGBM-এর মতো histogram-based — অনেক দ্রুত
hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=5,
    l2_regularization=0.1,
    random_state=42,
)

start = time.time()
hgb.fit(X_train, y_train)
print(f"Training time: {time.time()-start:.2f}s")
print(f"Test Accuracy: {hgb.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, hgb.predict_proba(X_test)[:,1]):.4f}")

# সুবিধা:
# ১. Missing values নিজেই handle করে
# ২. Categorical features সরাসরি দেওয়া যায়
# ৩. অনেক বড় ডেটায়ও দ্রুত</code></pre>

    <h3>৫. LightGBM — আরও দ্রুত</h3>
    <pre><code"># pip install lightgbm
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,       # 2^max_depth-1 এর কম রাখো
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,       # L1
    reg_lambda=0.1,      # L2
    random_state=42,
    n_jobs=-1,
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)
print(f"Test AUC: {roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>৬. GB vs HistGB vs LightGBM vs XGBoost</h3>
    <table>
      <thead><tr><th></th><th>GradientBoosting</th><th>HistGB</th><th>LightGBM</th><th>XGBoost</th></tr></thead>
      <tbody>
        <tr><td><strong>Speed</strong></td><td>ধীর</td><td>দ্রুত</td><td>সবচেয়ে দ্রুত</td><td>দ্রুত</td></tr>
        <tr><td><strong>Missing value</strong></td><td>না</td><td>হ্যাঁ</td><td>হ্যাঁ</td><td>হ্যাঁ</td></tr>
        <tr><td><strong>Categorical</strong></td><td>না</td><td>হ্যাঁ</td><td>হ্যাঁ</td><td>না</td></tr>
        <tr><td><strong>Big data</strong></td><td>ধীর</td><td>ভালো</td><td>সেরা</td><td>ভালো</td></tr>
        <tr><td><strong>Accuracy</strong></td><td>ভালো</td><td>ভালো</td><td>খুব ভালো</td><td>খুব ভালো</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Residual fitting</td><td>প্রতিটি tree আগের সব tree-র ভুল fit করে</td></tr>
        <tr><td>learning_rate</td><td>ছোট → বেশি tree দরকার কিন্তু ভালো generalization</td></tr>
        <tr><td>subsample</td><td>Stochastic GB — overfitting কমায়, speed বাড়ায়</td></tr>
        <tr><td>HistGBM</td><td>sklearn-এ দ্রুততম — বড় ডেটায় ব্যবহার করো</td></tr>
        <tr><td>LightGBM</td><td>production-এ সবচেয়ে জনপ্রিয় boosting library</td></tr>
      </tbody>
    </table>
  `,
};
