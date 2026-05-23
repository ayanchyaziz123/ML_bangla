export const logit_4_eval_metrics = {
  title: "Classification Metrics — Precision, Recall, F1, ROC-AUC বিস্তারিত",
  description: "Confusion Matrix থেকে শুরু করে ROC Curve, Precision-Recall Curve পর্যন্ত — কোন metric কখন ব্যবহার করবে এবং কেন Accuracy একাই যথেষ্ট নয়।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 13,
  slug: "logit-eval-metrics",
  content: `
    <h3>১. কেন Accuracy যথেষ্ট নয়?</h3>
    <p>ধরো ১০০০ জনের মধ্যে মাত্র ১০ জন ক্যান্সার রোগী। যদি মডেল সবসময় "ক্যান্সার নেই" বলে — Accuracy = 990/1000 = <strong>99%</strong>! কিন্তু মডেল ১০ জন রোগীকেই মিস করলো। এই সমস্যাকে বলে <strong>Accuracy Paradox</strong>।</p>

    <h3>২. Confusion Matrix</h3>
    <pre><code>from sklearn.metrics import confusion_matrix
import numpy as np

# উদাহরণ: ক্যান্সার detection (1=positive, 0=negative)
y_true = [1,1,1,0,0,0,1,0,1,0]
y_pred = [1,0,1,0,1,0,1,0,0,0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
#            Predicted 0  Predicted 1
# Actual 0  [[  TN=3        FP=1  ]]
# Actual 1  [[  FN=1        TP=4  ]]

TN, FP = cm[0]
FN, TP = cm[1]
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")</code></pre>
    <table>
      <thead><tr><th>নাম</th><th>অর্থ</th><th>উদাহরণ</th></tr></thead>
      <tbody>
        <tr><td><strong>TP</strong> (True Positive)</td><td>আসলে positive, সঠিকভাবে positive বলা</td><td>রোগী আছে, ধরা পড়েছে ✓</td></tr>
        <tr><td><strong>TN</strong> (True Negative)</td><td>আসলে negative, সঠিকভাবে negative বলা</td><td>রোগী নেই, ঠিকই বলা ✓</td></tr>
        <tr><td><strong>FP</strong> (False Positive)</td><td>আসলে negative, ভুলে positive বলা</td><td>রোগী নেই, কিন্তু আছে বলা ✗ (Type I Error)</td></tr>
        <tr><td><strong>FN</strong> (False Negative)</td><td>আসলে positive, ভুলে negative বলা</td><td>রোগী আছে, কিন্তু নেই বলা ✗ (Type II Error)</td></tr>
      </tbody>
    </table>

    <h3>৩. Precision ও Recall</h3>
    <pre><code>from sklearn.metrics import precision_score, recall_score

# Precision: যতটা positive বলেছি, তার মধ্যে কতটা সত্যিকারের positive?
# Precision = TP / (TP + FP)
# "মডেল যা বলেছে তার নির্ভুলতা"

precision = TP / (TP + FP)
print(f"Precision = {TP} / ({TP}+{FP}) = {precision:.3f}")

# Recall (Sensitivity): সব আসল positive-এর মধ্যে কতটা ধরা পড়েছে?
# Recall = TP / (TP + FN)
# "মডেল কতটুকু মিস করেনি"

recall = TP / (TP + FN)
print(f"Recall    = {TP} / ({TP}+{FN}) = {recall:.3f}")

# sklearn দিয়ে:
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
model = LogisticRegression(max_iter=1000)
model.fit(scaler.fit_transform(X_train), y_train)
y_pred = model.predict(scaler.transform(X_test))

print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Precision vs Recall — Tradeoff</text>
        <rect x="20" y="28" width="200" height="65" rx="6" fill="#fee2e2" stroke="#fca5a5"/>
        <text x="120" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#991b1b">High Recall, Low Precision</text>
        <text x="120" y="65" text-anchor="middle" font-size="8" fill="#dc2626">Threshold কম → বেশি positive বলো</text>
        <text x="120" y="80" text-anchor="middle" font-size="8" fill="#dc2626">কম miss, কিন্তু false alarm বেশি</text>
        <text x="120" y="92" text-anchor="middle" font-size="8" fill="#6b7280">(cancer screening)</text>
        <rect x="260" y="28" width="200" height="65" rx="6" fill="#dcfce7" stroke="#86efac"/>
        <text x="360" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">High Precision, Low Recall</text>
        <text x="360" y="65" text-anchor="middle" font-size="8" fill="#16a34a">Threshold বেশি → কম positive বলো</text>
        <text x="360" y="80" text-anchor="middle" font-size="8" fill="#16a34a">নিশ্চিত positive, কিন্তু miss বেশি</text>
        <text x="360" y="92" text-anchor="middle" font-size="8" fill="#6b7280">(spam filter)</text>
      </svg>
    </div>

    <h3>৪. F1 Score — Precision ও Recall-এর সমন্বয়</h3>
    <pre><code>from sklearn.metrics import f1_score

# F1 = Harmonic mean of Precision and Recall
# F1 = 2 × (Precision × Recall) / (Precision + Recall)
# দুটো কাছাকাছি হলে F1 বেশি, একটি খুব কম হলে F1 কম

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.3f}")

# F-beta Score: beta > 1 → Recall বেশি গুরুত্বপূর্ণ
#              beta < 1 → Precision বেশি গুরুত্বপূর্ণ
from sklearn.metrics import fbeta_score
f2 = fbeta_score(y_test, y_pred, beta=2)   # Recall দ্বিগুণ গুরুত্ব
print(f"F2 Score: {f2:.3f}")</code></pre>

    <h3>৫. ROC Curve ও AUC</h3>
    <pre><code">from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', linewidth=2,
         label=f'AUC = {auc:.3f}')
plt.fill_between(fpr, tpr, alpha=0.1)
plt.plot([0,1], [0,1], 'k--', label='Random Classifier (AUC=0.5)')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUC ব্যাখ্যা:
# AUC = 1.0 → নিখুঁত classifier
# AUC = 0.5 → random guessing (কোনো কাজের না)
# AUC = 0.7–0.8 → ঠিকঠাক
# AUC > 0.9 → চমৎকার</code></pre>

    <h3>৬. Precision-Recall Curve</h3>
    <p>Imbalanced class-এ ROC curve বিভ্রান্তিকর হতে পারে। <strong>PR Curve</strong> বেশি উপযুক্ত।</p>
    <pre><code">from sklearn.metrics import precision_recall_curve, average_precision_score

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, color='green', linewidth=2,
         label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# AP (Average Precision): PR Curve-এর নিচের এলাকা
# 1.0 = নিখুঁত, 0.0 = সবচেয়ে খারাপ</code></pre>

    <h3>৭. কোন Metric কখন?</h3>
    <table>
      <thead><tr><th>Problem</th><th>সবচেয়ে গুরুত্বপূর্ণ</th><th>কারণ</th></tr></thead>
      <tbody>
        <tr><td>Cancer / Disease detection</td><td>Recall</td><td>রোগী miss করা মারাত্মক</td></tr>
        <tr><td>Spam detection</td><td>Precision</td><td>Important email spam ট্যাগ হওয়া খারাপ</td></tr>
        <tr><td>Fraud detection</td><td>PR-AUC বা F1</td><td>Imbalanced class, দুটোই গুরুত্বপূর্ণ</td></tr>
        <tr><td>Balanced dataset</td><td>Accuracy বা F1</td><td>সমান class distribution</td></tr>
        <tr><td>Ranking/Probability দরকার</td><td>ROC-AUC</td><td>Threshold-independent metric</td></tr>
        <tr><td>Imbalanced dataset</td><td>PR-AUC বা F1</td><td>ROC-AUC বিভ্রান্তিকর হতে পারে</td></tr>
      </tbody>
    </table>

    <h3>৮. সব একসাথে দেখো</h3>
    <pre><code">from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,
      target_names=['Benign', 'Malignant']))

# আউটপুট:
#               precision    recall  f1-score   support
#       Benign       0.97      0.96      0.97        43
#    Malignant       0.97      0.98      0.98        71
#     accuracy                           0.97       114
#    macro avg       0.97      0.97      0.97       114
# weighted avg       0.97      0.97      0.97       114</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>Metric</th><th>সূত্র</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Accuracy</td><td>(TP+TN)/Total</td><td>Imbalanced-এ misleading</td></tr>
        <tr><td>Precision</td><td>TP/(TP+FP)</td><td>positive prediction-এর নির্ভুলতা</td></tr>
        <tr><td>Recall</td><td>TP/(TP+FN)</td><td>কতটুকু ধরা পড়েছে (miss কম)</td></tr>
        <tr><td>F1</td><td>2×P×R/(P+R)</td><td>Precision ও Recall-এর সমন্বয়</td></tr>
        <tr><td>ROC-AUC</td><td>ROC curve-এর area</td><td>threshold-independent, balanced data</td></tr>
        <tr><td>PR-AUC</td><td>PR curve-এর area</td><td>imbalanced class-এ সেরা</td></tr>
      </tbody>
    </table>
  `,
};
