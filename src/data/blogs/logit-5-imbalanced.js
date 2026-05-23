export const logit_5_imbalanced = {
  title: "Imbalanced Dataset — Class Imbalance কীভাবে সামলাবো?",
  description: "SMOTE, class_weight, threshold tuning, undersampling — যখন একটি class অনেক কম, মডেল কীভাবে ঠিকঠাক কাজ করবে তা বাংলায় বিস্তারিত।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 11,
  slug: "logit-imbalanced-dataset",
  content: `
    <h3>১. সমস্যাটা কী?</h3>
    <p>Fraud detection-এ হয়তো ১০,০০০ লেনদেনের মধ্যে মাত্র ৫০টি fraud। মডেল সহজেই শেখে "সব normal" বলো — Accuracy 99.5%! কিন্তু এটি সম্পূর্ণ ব্যর্থ মডেল।</p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Imbalanced ডেটা তৈরি (1:20 ratio)
X, y = make_classification(
    n_samples=2000,
    n_features=10,
    weights=[0.95, 0.05],   # 95% class-0, 5% class-1
    random_state=42
)

print(f"Class 0: {(y==0).sum()}")   # 1900
print(f"Class 1: {(y==1).sum()}")   # 100
print(f"Ratio: 1:{(y==0).sum()//(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# সাধারণ মডেল
model_plain = LogisticRegression(max_iter=1000, random_state=42)
model_plain.fit(X_train_s, y_train)
y_pred_plain = model_plain.predict(X_test_s)

print("\\n=== সাধারণ মডেল ===")
print(classification_report(y_test, y_pred_plain))
# Class 1 Recall অনেক কম — minority class প্রায় miss!</code></pre>

    <h3>২. class_weight — সহজ সমাধান</h3>
    <p>Minority class-এর উপর বেশি penalty দাও — মডেল সেগুলো মিস করলে বেশি ক্ষতিগ্রস্ত হয়।</p>
    <pre><code">model_cw = LogisticRegression(
    class_weight='balanced',   # স্বয়ংক্রিয়ভাবে weight ঠিক করে
    max_iter=1000,
    random_state=42
)
model_cw.fit(X_train_s, y_train)

print("=== class_weight='balanced' ===")
print(classification_report(y_test, model_cw.predict(X_test_s)))

# নিজে weight দিতে চাইলে:
model_custom = LogisticRegression(
    class_weight={0: 1, 1: 20},   # minority class 20 গুণ ওজন
    max_iter=1000
)
model_custom.fit(X_train_s, y_train)

# class_weight='balanced' হলে weight = n_samples / (n_classes × n_class_i)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Imbalanced সামলানোর কৌশল</text>
        <rect x="15"  y="28" width="100" height="60" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="65"  y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">class_weight</text>
        <text x="65"  y="65" text-anchor="middle" font-size="8" fill="#3b82f6">ডেটা না বদলে</text>
        <text x="65"  y="78" text-anchor="middle" font-size="8" fill="#3b82f6">penalty বাড়াও</text>
        <rect x="130" y="28" width="100" height="60" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="180" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Oversampling</text>
        <text x="180" y="65" text-anchor="middle" font-size="8" fill="#16a34a">minority class</text>
        <text x="180" y="78" text-anchor="middle" font-size="8" fill="#16a34a">বাড়াও (SMOTE)</text>
        <rect x="245" y="28" width="100" height="60" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="295" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Undersampling</text>
        <text x="295" y="65" text-anchor="middle" font-size="8" fill="#d97706">majority class</text>
        <text x="295" y="78" text-anchor="middle" font-size="8" fill="#d97706">কমাও</text>
        <rect x="360" y="28" width="105" height="60" rx="5" fill="#fce7f3" stroke="#f9a8d4"/>
        <text x="412" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#9d174d">Threshold</text>
        <text x="412" y="65" text-anchor="middle" font-size="8" fill="#be185d">Tuning</text>
        <text x="412" y="78" text-anchor="middle" font-size="8" fill="#be185d">0.5 থেকে কমাও</text>
      </svg>
    </div>

    <h3>৩. Threshold Tuning</h3>
    <pre><code">y_prob = model_plain.predict_proba(X_test_s)[:, 1]

# Precision-Recall tradeoff দেখো
from sklearn.metrics import precision_recall_curve, f1_score

prec, rec, thresholds = precision_recall_curve(y_test, y_prob)

# সেরা threshold খোঁজো (F1 maximize করো)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print(f"সেরা threshold: {best_threshold:.3f}")
print(f"সেরা F1: {f1_scores[best_idx]:.3f}")

# এই threshold দিয়ে predict করো
y_pred_tuned = (y_prob >= best_threshold).astype(int)
print("\\n=== Threshold tuning ===")
print(classification_report(y_test, y_pred_tuned))</code></pre>

    <h3>৪. SMOTE — Synthetic Minority Oversampling</h3>
    <p>Minority class-এর নতুন synthetic sample তৈরি করে — কাছের প্রতিবেশীদের মধ্যে interpolation করে।</p>
    <pre><code"># pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE শুধু training data-তে প্রয়োগ করো!
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_s, y_train)

print(f"আগে: {y_train.sum()} minority sample")
print(f"পরে: {y_resampled.sum()} minority sample")

model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_resampled, y_resampled)

print("\\n=== SMOTE ===")
print(classification_report(y_test, model_smote.predict(X_test_s)))

# imbalanced-learn Pipeline দিয়ে সঠিকভাবে:
pipe_smote = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(random_state=42)),
    ('model',  LogisticRegression(max_iter=1000)),
])
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe_smote, X_train, y_train,
                         cv=5, scoring='f1')
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")</code></pre>

    <h3>৫. Undersampling</h3>
    <pre><code">from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Random Undersampling: majority class থেকে random বাদ দাও
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train_s, y_train)
print(f"আগে: {len(X_train_s)}, পরে: {len(X_under)}")

# TomekLinks: boundary-র কাছের majority sample বাদ দাও
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train_s, y_train)

# Combination: SMOTE + Undersampling
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_combined, y_combined = smt.fit_resample(X_train_s, y_train)</code></pre>

    <h3>৬. কোন পদ্ধতি কখন?</h3>
    <table>
      <thead><tr><th>পদ্ধতি</th><th>কখন</th><th>সুবিধা</th><th>অসুবিধা</th></tr></thead>
      <tbody>
        <tr><td>class_weight</td><td>সবসময় প্রথম চেষ্টা</td><td>সহজ, data বদলায় না</td><td>সবসময় যথেষ্ট নয়</td></tr>
        <tr><td>Threshold tuning</td><td>model আছে, শুধু cutoff বদলাও</td><td>সহজ, post-hoc</td><td>training-এ imbalance থাকে</td></tr>
        <tr><td>SMOTE</td><td>minority data খুব কম</td><td>নতুন data তৈরি করে</td><td>noise তৈরি হতে পারে</td></tr>
        <tr><td>Undersampling</td><td>majority data অনেক বেশি</td><td>দ্রুত training</td><td>information হারায়</td></tr>
        <tr><td>SMOTETomek</td><td>দুটো একসাথে দরকার</td><td>সেরা balance</td><td>ধীর</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Imbalanced data</td><td>Accuracy দেখো না — F1, PR-AUC দেখো</td></tr>
        <tr><td>class_weight='balanced'</td><td>সবচেয়ে সহজ সমাধান — প্রথমে চেষ্টা করো</td></tr>
        <tr><td>Threshold tuning</td><td>প্রয়োজন অনুযায়ী Recall বা Precision বাড়াও</td></tr>
        <tr><td>SMOTE</td><td>Training data-তে প্রয়োগ করো, test-এ নয়</td></tr>
        <tr><td>stratify=y</td><td>train/test split-এ class ratio বজায় রাখো</td></tr>
      </tbody>
    </table>
  `,
};
