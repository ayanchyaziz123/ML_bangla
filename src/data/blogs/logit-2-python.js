export const logit_2_python = {
  title: "Python দিয়ে Logistic Regression — সম্পূর্ণ কোড গাইড",
  description: "ডেটা প্রস্তুতি থেকে মডেল training, probability calibration, threshold tuning পর্যন্ত — লজিস্টিক রিগ্রেশনের সম্পূর্ণ Python workflow বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 12,
  slug: "logit-python-guide",
  content: `
    <h3>১. ডেটা লোড ও বিশ্লেষণ</h3>
    <pre><code>import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt

# ডেটা লোড
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target   # 1 = malignant, 0 = benign

print(df.shape)              # (569, 31)
print(df['target'].value_counts())
# 1    357  (malignant)
# 0    212  (benign)

print(df.describe().T[['mean', 'std', 'min', 'max']].round(2))</code></pre>

    <h3>২. Train/Test Split ও Scaling</h3>
    <pre><code>X = df.drop('target', axis=1)
y = df['target']

# stratify=y → class ratio বজায় রাখে
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)     # শুধু transform, fit নয়!

print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")</code></pre>

    <h3>৩. মডেল Training</h3>
    <pre><code">model = LogisticRegression(
    C=1.0,              # 1/regularization strength
    penalty='l2',       # Ridge regularization
    solver='lbfgs',     # optimization algorithm
    max_iter=1000,
    random_state=42,
)
model.fit(X_train_s, y_train)

# Coefficients দেখো
coef_df = pd.DataFrame({
    'Feature':     data.feature_names,
    'Coefficient': model.coef_[0],
}).sort_values('Coefficient', key=abs, ascending=False)
print("সবচেয়ে গুরুত্বপূর্ণ ফিচার:")
print(coef_df.head(5))</code></pre>

    <h3>৪. Prediction — class ও probability</h3>
    <pre><code"># Class prediction (0 বা 1)
y_pred = model.predict(X_test_s)

# Probability prediction
y_prob = model.predict_proba(X_test_s)
print(y_prob[:5])
# [[0.02, 0.98],   ← 98% malignant
#  [0.89, 0.11],   ← 89% benign
#  [0.05, 0.95],   ...
#  ...]

# শুধু class 1-এর probability (ROC-AUC-এর জন্য)
y_prob_pos = y_prob[:, 1]

# Decision score (log-odds)
y_score = model.decision_function(X_test_s)
print("Decision scores (log-odds):", y_score[:5].round(3))</code></pre>

    <h3>৫. Threshold Tuning</h3>
    <p>Default threshold = 0.5। কিন্তু cancer detection-এ False Negative (ক্যান্সার মিস) অনেক বড় ক্ষতি — তাই threshold কমিয়ে আরও সতর্ক হওয়া উচিত।</p>
    <pre><code>thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>6}")
print("-" * 45)
for t in thresholds:
    y_pred_t = (y_prob_pos >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    print(f"{t:>10.1f} | {p:>10.3f} | {r:>8.3f} | {f:>6.3f}")

# Threshold কমালে: Recall বাড়ে, Precision কমে
# Cancer detection-এ Recall বেশি গুরুত্বপূর্ণ</code></pre>

    <h3>৬. সম্পূর্ণ Evaluation</h3>
    <pre><code">print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Benign', 'Malignant']))

# ROC-AUC
auc = roc_auc_score(y_test, y_prob_pos)
print(f"ROC-AUC: {auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='blue')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)</code></pre>

    <h3>৭. Probability Calibration</h3>
    <p>কখনো মডেলের probability সঠিক নাও হতে পারে (overconfident)। Calibration দিয়ে এটি ঠিক করা যায়।</p>
    <pre><code">from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Calibrated model
cal_model = CalibratedClassifierCV(
    LogisticRegression(C=1.0, max_iter=1000),
    method='isotonic',   # 'sigmoid' বা 'isotonic'
    cv=5,
)
cal_model.fit(X_train_s, y_train)
y_prob_cal = cal_model.predict_proba(X_test_s)[:, 1]

# Calibration Curve (reliability diagram)
prob_true, prob_pred = calibration_curve(y_test, y_prob_pos, n_bins=10)
prob_true_c, prob_pred_c = calibration_curve(y_test, y_prob_cal, n_bins=10)

plt.plot(prob_pred, prob_true, 'o-', label='Original')
plt.plot(prob_pred_c, prob_true_c, 's-', label='Calibrated')
plt.plot([0,1], [0,1], 'k--', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()</code></pre>

    <h3>৮. সম্পূর্ণ Pipeline</h3>
    <pre><code">from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
])

# 5-fold CV
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final training
pipe.fit(X_train, y_train)
print(f"Test ROC-AUC: {roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>ধাপ</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Scaling</td><td>StandardScaler আবশ্যক — শুধু train-এ fit করো</td></tr>
        <tr><td>predict_proba</td><td>probability পাওয়ার জন্য — threshold নিজে ঠিক করতে হলে দরকার</td></tr>
        <tr><td>Threshold</td><td>0.5 সবসময় সেরা নয় — problem অনুযায়ী বাছো</td></tr>
        <tr><td>ROC-AUC</td><td>imbalanced class-এ accuracy-র চেয়ে ভালো metric</td></tr>
        <tr><td>Calibration</td><td>probability সঠিক রাখতে — medical/finance-এ জরুরি</td></tr>
      </tbody>
    </table>
  `,
};
