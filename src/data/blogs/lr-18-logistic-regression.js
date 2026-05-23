export const lr_18_logistic_regression = {
  title: "Logistic Regression — রিগ্রেশন দিয়ে Classification",
  description: "Logistic Regression কীভাবে 0/1 ভবিষ্যদ্বাণী করে, Sigmoid function, Log Loss, এবং Linear Regression-এর সাথে পার্থক্য — বাংলায় সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 12,
  slug: "lr-logistic-regression",
  content: `
    <h3>১. লিনিয়ার রিগ্রেশন দিয়ে Classification কেন হয় না?</h3>
    <p>ধরো তুমি বলতে চাও: একটি ইমেইল spam (1) নাকি not spam (0)? লিনিয়ার রিগ্রেশন দিলে output হতে পারে -0.3 বা 1.7 — কিন্তু probability 0 থেকে 1-এর মধ্যে হওয়া দরকার। এই সমস্যার সমাধান <strong>Logistic Regression</strong>।</p>

    <h3>২. Sigmoid Function — 0 থেকে 1-এর মধ্যে আনো</h3>
    <pre><code># Sigmoid (Logistic) function:
σ(z) = 1 / (1 + e⁻ᶻ)

যেখানে z = m₁x₁ + m₂x₂ + ... + b  (linear combination)

# বৈশিষ্ট্য:
# z = 0   → σ = 0.5  (অনিশ্চিত)
# z → +∞  → σ → 1   (class 1)
# z → -∞  → σ → 0   (class 0)

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 100)
sigma = 1 / (1 + np.exp(-z))

plt.plot(z, sigma, color='blue', linewidth=2)
plt.axhline(0.5, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('z (linear output)')
plt.ylabel('σ(z) = probability')
plt.title('Sigmoid Function')
plt.show()</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Sigmoid Function</text>
        <line x1="40" y1="115" x2="440" y2="115" stroke="#e5e7eb" stroke-width="1"/>
        <line x1="40" y1="25" x2="40" y2="115" stroke="#e5e7eb" stroke-width="1"/>
        <line x1="40" y1="70" x2="440" y2="70" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4,3"/>
        <line x1="240" y1="25" x2="240" y2="115" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4,3"/>
        <path d="M 40 112 Q 100 110 160 100 Q 200 93 240 70 Q 280 47 320 40 Q 380 30 440 28" stroke="#1e40af" stroke-width="2.5" fill="none"/>
        <text x="30" y="74" text-anchor="end" font-size="8" fill="#6b7280">0.5</text>
        <text x="30" y="118" text-anchor="end" font-size="8" fill="#6b7280">0</text>
        <text x="30" y="30" text-anchor="end" font-size="8" fill="#6b7280">1</text>
        <text x="240" y="125" text-anchor="middle" font-size="8" fill="#6b7280">z=0</text>
        <circle cx="240" cy="70" r="3" fill="#dc2626"/>
        <text x="255" y="68" font-size="8" fill="#dc2626">σ(0)=0.5</text>
      </svg>
    </div>

    <h3>৩. সিদ্ধান্ত নেওয়া</h3>
    <pre><code># যদি σ(z) ≥ 0.5 → class 1 (positive)
# যদি σ(z) < 0.5 → class 0 (negative)

# Default threshold = 0.5, কিন্তু পরিবর্তন করা যায়:
# Fraud detection → threshold = 0.3 (কম মিস করতে চাই)
# Spam detection  → threshold = 0.7 (genuine email miss না করতে)</code></pre>

    <h3>৪. Loss Function — Log Loss (Binary Cross-Entropy)</h3>
    <pre><code># Logistic Regression MSE ব্যবহার করে না কারণ MSE non-convex হয়।
# এর বদলে Log Loss ব্যবহার করে:

Log Loss = -(1/n) × Σ[yᵢ × log(p̂ᵢ) + (1−yᵢ) × log(1−p̂ᵢ)]

# যখন y=1: loss = -log(p̂)  → p̂ = 1 হলে loss = 0
# যখন y=0: loss = -log(1−p̂) → p̂ = 0 হলে loss = 0

# Python:
from sklearn.metrics import log_loss
y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3]
print(f"Log Loss: {log_loss(y_true, y_prob):.4f}")</code></pre>

    <h3>৫. Python দিয়ে Logistic Regression</h3>
    <pre><code">from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.datasets import load_breast_cancer
import pandas as pd

# ডেটা লোড
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling আবশ্যক
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# মডেল
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

# Prediction
y_pred      = lr.predict(X_test_s)
y_prob      = lr.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, y_pred,
      target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")</code></pre>

    <h3>৬. Confusion Matrix বোঝা</h3>
    <pre><code">from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
#               Predicted 0   Predicted 1
# Actual 0   [[  TN=40        FP=2    ]]
# Actual 1   [[  FN=3         TP=69   ]]

TP = cm[1,1]; TN = cm[0,0]; FP = cm[0,1]; FN = cm[1,0]
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")</code></pre>

    <h3>৭. Linear vs Logistic Regression তুলনা</h3>
    <table>
      <thead><tr><th></th><th>Linear Regression</th><th>Logistic Regression</th></tr></thead>
      <tbody>
        <tr><td><strong>Output</strong></td><td>যেকোনো সংখ্যা (-∞ থেকে +∞)</td><td>0 থেকে 1 (probability)</td></tr>
        <tr><td><strong>কাজ</strong></td><td>Regression (continuous prediction)</td><td>Classification (class prediction)</td></tr>
        <tr><td><strong>Loss</strong></td><td>MSE</td><td>Log Loss (Binary Cross-Entropy)</td></tr>
        <tr><td><strong>Output function</strong></td><td>Identity (z)</td><td>Sigmoid σ(z)</td></tr>
        <tr><td><strong>Assumption</strong></td><td>Linear relationship with y</td><td>Linear relationship with log-odds</td></tr>
      </tbody>
    </table>

    <h3>৮. Regularization Logistic Regression-এ</h3>
    <pre><code">from sklearn.linear_model import LogisticRegressionCV

# C = 1/alpha (C বড় → কম regularization, C ছোট → বেশি regularization)
# penalty: 'l1'=Lasso, 'l2'=Ridge (default), 'elasticnet'

lr_cv = LogisticRegressionCV(
    Cs=10,              # 10টি C মান test করবে
    cv=5,
    penalty='l2',       # Ridge
    scoring='roc_auc',
    max_iter=1000,
    random_state=42,
)
lr_cv.fit(X_train_s, y_train)
print(f"সেরা C: {lr_cv.C_[0]:.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, lr_cv.predict_proba(X_test_s)[:,1]):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Sigmoid</td><td>linear output-কে 0–1 probability-তে রূপান্তর করে</td></tr>
        <tr><td>Log Loss</td><td>Logistic Regression-এর loss function — MSE নয়</td></tr>
        <tr><td>Threshold</td><td>default 0.5 — problem অনুযায়ী বদলাও</td></tr>
        <tr><td>C parameter</td><td>1/alpha — বড় C মানে কম regularization</td></tr>
        <tr><td>Scaling আবশ্যক</td><td>StandardScaler ব্যবহার না করলে convergence ধীর হয়</td></tr>
      </tbody>
    </table>
  `,
};
