export const anomaly_5_project = {
  slug: 'anomaly-5-project',
  title: 'অ্যানোমালি ডিটেকশন প্রজেক্ট: ক্রেডিট কার্ড জালিয়াতি',
  description: 'Isolation Forest, LOF ও One-Class SVM তুলনা করে সেরা জালিয়াতি শনাক্তকরণ মডেল তৈরি করুন',
  date: 'মে ২০২৫',
  category: 'অ্যানোমালি ডিটেকশন',
  readTime: 14,
  content: `
<h3>প্রজেক্ট পরিচিতি</h3>
<p>ক্রেডিট কার্ড জালিয়াতি শনাক্তকরণ একটি ক্লাসিক অ্যানোমালি ডিটেকশন সমস্যা। মাত্র ০.১৭% লেনদেন জালিয়াতি — ভয়ানক ইমব্যালেন্সড ডেটা।</p>

<h3>ডেটা প্রস্তুতি</h3>

<pre><code class="language-python">import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ডেটা লোড (Kaggle Credit Card Fraud)
df = pd.read_csv('creditcard.csv')
print(f"আকার: {df.shape}")
print(f"জালিয়াতি: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"স্বাভাবিক: {(df['Class']==0).sum()}")

# ফিচার স্কেলিং
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

# V1-V28 ইতিমধ্যে PCA-তে রূপান্তরিত
features = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
X = df[features].values
y = df['Class'].values

print(f"\\nফিচার আকার: {X.shape}")
</code></pre>

<h3>তিনটি মডেল প্রশিক্ষণ</h3>

<pre><code class="language-python">contamination = 0.0017  # প্রকৃত জালিয়াতির হার

# ১. Isolation Forest
iforest = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
iforest.fit(X)
y_pred_if = (iforest.predict(X) == -1).astype(int)

# ২. Local Outlier Factor
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination,
    n_jobs=-1
)
y_pred_lof = (lof.fit_predict(X) == -1).astype(int)

# ৩. One-Class SVM (ছোট নমুনায়)
X_sample = X[:10000]  # SVM ধীর — নমুনা নিই
y_sample = y[:10000]

ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
ocsvm.fit(X_sample[y_sample == 0])  # শুধু স্বাভাবিক দিয়ে ট্রেন
y_pred_svm = (ocsvm.predict(X_sample) == -1).astype(int)
</code></pre>

<h3>মূল্যায়ন ও তুলনা</h3>

<pre><code class="language-python">def evaluate_model(y_true, y_pred, model_name, scores=None):
    print(f"\\n{'='*50}")
    print(f"মডেল: {model_name}")
    print(classification_report(y_true, y_pred,
                                 target_names=['স্বাভাবিক', 'জালিয়াতি']))
    if scores is not None:
        auc = roc_auc_score(y_true, scores)
        print(f"ROC-AUC: {auc:.4f}")

# Isolation Forest স্কোর
if_scores = -iforest.score_samples(X)  # স্কোর বিপরীত করি

evaluate_model(y, y_pred_if, "Isolation Forest", if_scores)
evaluate_model(y, y_pred_lof, "Local Outlier Factor")  # LOF-এ predict_proba নেই
evaluate_model(y_sample, y_pred_svm, "One-Class SVM (১০,০০০ নমুনা)")
</code></pre>

<h4>ফলাফল তুলনা</h4>
<table>
<tr><th>মডেল</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
<tr><td>Isolation Forest</td><td>0.27</td><td>0.31</td><td>0.29</td><td>0.92</td></tr>
<tr><td>LOF</td><td>0.05</td><td>0.02</td><td>0.03</td><td>N/A</td></tr>
<tr><td>One-Class SVM</td><td>0.09</td><td>0.71</td><td>0.16</td><td>N/A</td></tr>
</table>

<h3>থ্রেশহোল্ড অপ্টিমাইজেশন</h3>

<pre><code class="language-python">from sklearn.metrics import precision_recall_curve, f1_score

# Isolation Forest-এর স্কোর দিয়ে সর্বোত্তম থ্রেশহোল্ড
precision, recall, thresholds = precision_recall_curve(y, if_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"সর্বোত্তম থ্রেশহোল্ড: {best_threshold:.4f}")
print(f"সর্বোচ্চ F1: {f1_scores[best_idx]:.4f}")
print(f"এ স্তরে Precision: {precision[best_idx]:.4f}")
print(f"এ স্তরে Recall: {recall[best_idx]:.4f}")

# ব্যবসায়িক সিদ্ধান্ত: উচ্চ recall পছন্দ করা ভালো
# (জালিয়াতি মিস করা = বেশি ক্ষতি)
high_recall_threshold = thresholds[np.where(recall >= 0.85)[0][0]]
y_pred_hr = (if_scores >= high_recall_threshold).astype(int)
print(f"\\nউচ্চ-Recall মোড (Recall>=0.85):")
print(classification_report(y, y_pred_hr, target_names=['স্বাভাবিক', 'জালিয়াতি']))
</code></pre>

<h3>এনসেম্বল পদ্ধতি</h3>

<pre><code class="language-python"># তিনটি মডেলের ভোটিং
from sklearn.preprocessing import MinMaxScaler

# IF স্কোর নর্মালাইজ
ms = MinMaxScaler()
if_norm = ms.fit_transform(if_scores.reshape(-1, 1)).ravel()

# LOF স্কোর
lof_scores = -lof.negative_outlier_factor_
lof_norm = ms.fit_transform(lof_scores.reshape(-1, 1)).ravel()

# এনসেম্বল স্কোর
ensemble_score = 0.6 * if_norm + 0.4 * lof_norm
threshold_ens = np.percentile(ensemble_score, (1 - contamination) * 100)
y_pred_ens = (ensemble_score >= threshold_ens).astype(int)

print("\\nএনসেম্বল ফলাফল:")
print(classification_report(y, y_pred_ens, target_names=['স্বাভাবিক', 'জালিয়াতি']))
print(f"ROC-AUC: {roc_auc_score(y, ensemble_score):.4f}")
</code></pre>

<h4>মূল উপসংহার</h4>
<ul>
<li><strong>Isolation Forest</strong> সেরা সামগ্রিক পারফরম্যান্স দেয়</li>
<li><strong>LOF</strong> ছোট স্থানীয় অ্যানোমালি ধরতে ভালো</li>
<li><strong>One-Class SVM</strong> উচ্চ recall কিন্তু অনেক false positive</li>
<li><strong>Contamination</strong> প্যারামিটার সঠিক করলে পারফরম্যান্স উন্নত হয়</li>
<li>ব্যবসায়িক প্রয়োজন অনুযায়ী threshold সামঞ্জস্য করুন</li>
</ul>
`
};
