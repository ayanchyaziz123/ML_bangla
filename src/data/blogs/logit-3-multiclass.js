export const logit_3_multiclass = {
  title: "Multiclass Logistic Regression — একের বেশি class ভবিষ্যদ্বাণী",
  description: "One-vs-Rest, One-vs-One এবং Softmax (Multinomial) — যখন binary নয়, বহু class থাকে তখন Logistic Regression কীভাবে কাজ করে।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 10,
  slug: "logit-multiclass",
  content: `
    <h3>১. Multiclass কী?</h3>
    <p>Binary classification-এ শুধু দুটো class (0 বা 1)। কিন্তু বাস্তবে:</p>
    <ul>
      <li>ফুলের প্রজাতি: Setosa / Versicolor / Virginica</li>
      <li>সংখ্যা চেনা: 0 থেকে 9 (১০টি class)</li>
      <li>রোগ নির্ণয়: A / B / C / D টাইপ</li>
    </ul>
    <p>এই সমস্যা সমাধানে তিনটি পদ্ধতি: <strong>OvR, OvO, Softmax</strong>।</p>

    <h3>২. One-vs-Rest (OvR / OvA)</h3>
    <p>k class থাকলে k টি binary classifier তৈরি করো। প্রতিটি classifier একটি class-কে "rest" এর বিপরীতে classify করে।</p>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

data = load_iris()
X, y = data.data, data.target
# y: 0=Setosa, 1=Versicolor, 2=Virginica

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# One-vs-Rest
model_ovr = LogisticRegression(
    multi_class='ovr',    # OvR পদ্ধতি
    solver='lbfgs',
    max_iter=1000,
)
model_ovr.fit(X_train_s, y_train)
print("OvR Accuracy:", model_ovr.score(X_test_s, y_test))
print(model_ovr.coef_.shape)   # (3, 4) ← 3টি binary classifier, 4 feature</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">One-vs-Rest: ৩ class → ৩টি binary classifier</text>
        <rect x="20" y="30" width="130" height="32" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="85" y="50" text-anchor="middle" font-size="9" fill="#1e40af">Setosa vs (Versicolor+Virginica)</text>
        <rect x="170" y="30" width="140" height="32" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="240" y="50" text-anchor="middle" font-size="9" fill="#166534">Versicolor vs (Setosa+Virginica)</text>
        <rect x="330" y="30" width="130" height="32" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="395" y="50" text-anchor="middle" font-size="9" fill="#92400e">Virginica vs (Setosa+Versicolor)</text>
        <text x="240" y="95" text-anchor="middle" font-size="9" fill="#6b7280">সর্বোচ্চ probability-র class জেতে → argmax(p₀, p₁, p₂)</text>
      </svg>
    </div>

    <h3>৩. One-vs-One (OvO)</h3>
    <p>k class থাকলে k(k−1)/2 টি binary classifier তৈরি করো। প্রতিটি classifier দুটি class-এর মধ্যে vote করে।</p>
    <pre><code">from sklearn.multiclass import OneVsOneClassifier

model_ovo = OneVsOneClassifier(
    LogisticRegression(max_iter=1000)
)
model_ovo.fit(X_train_s, y_train)
print("OvO Accuracy:", model_ovo.score(X_test_s, y_test))

# 3 class → 3(3-1)/2 = 3 টি classifier
# 10 class → 45 টি classifier!</code></pre>

    <h3>৪. Softmax (Multinomial) — সবচেয়ে ভালো পদ্ধতি</h3>
    <p>সরাসরি সব class একসাথে মডেল করে। Softmax function সব class-এর probability বের করে — যেন সব class প্রতিযোগিতা করছে।</p>
    <pre><code">import numpy as np

# Softmax function:
def softmax(z):
    e_z = np.exp(z - np.max(z))   # numerical stability-র জন্য max বিয়োগ
    return e_z / e_z.sum()

# উদাহরণ: ৩ class-এর raw scores
z = np.array([2.0, 1.0, 0.5])
probs = softmax(z)
print("Softmax probabilities:", probs.round(3))
# [0.659, 0.242, 0.099]  ← সব মিলে = 1.0

# sklearn দিয়ে Multinomial:
model_softmax = LogisticRegression(
    multi_class='multinomial',   # Softmax
    solver='lbfgs',              # multinomial সমর্থন করে
    max_iter=1000,
)
model_softmax.fit(X_train_s, y_train)

y_prob = model_softmax.predict_proba(X_test_s)
print("Probability for first 3 test samples:")
print(y_prob[:3].round(3))
#        Setosa  Versicolor  Virginica
# [[0.001   0.012      0.987],
#  [0.945   0.054      0.001],
#  [0.002   0.962      0.036]]
print("Accuracy:", model_softmax.score(X_test_s, y_test))</code></pre>

    <h3>৫. তিনটি পদ্ধতির তুলনা</h3>
    <table>
      <thead><tr><th></th><th>OvR</th><th>OvO</th><th>Softmax (Multinomial)</th></tr></thead>
      <tbody>
        <tr><td><strong>Classifier সংখ্যা</strong></td><td>k টি</td><td>k(k−1)/2 টি</td><td>১টি (সব class একসাথে)</td></tr>
        <tr><td><strong>Training speed</strong></td><td>মাঝামাঝি</td><td>ধীর (বেশি classifier)</td><td>দ্রুত</td></tr>
        <tr><td><strong>Probability</strong></td><td>calibrated নয়</td><td>calibrated নয়</td><td>সঠিকভাবে calibrated</td></tr>
        <tr><td><strong>Class imbalance</strong></td><td>সমস্যা হতে পারে</td><td>তুলনামূলক ভালো</td><td>সবচেয়ে ভালো</td></tr>
        <tr><td><strong>sklearn default</strong></td><td>lbfgs solver-এ auto</td><td>-</td><td>multi_class='multinomial'</td></tr>
      </tbody>
    </table>

    <h3>৬. Multiclass Evaluation</h3>
    <pre><code">from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

y_pred = model_softmax.predict(X_test_s)

print(classification_report(y_test, y_pred,
      target_names=data.target_names))

# Macro vs Weighted Average:
# macro   → প্রতিটি class-এর স্কোরের সরল গড় (imbalanced-এ সতর্ক থাকো)
# weighted → class size অনুযায়ী ওজন দিয়ে গড়

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot()
plt.title('Multiclass Confusion Matrix')
plt.show()

# ROC-AUC for multiclass:
from sklearn.metrics import roc_auc_score
auc_ovr = roc_auc_score(y_test, model_softmax.predict_proba(X_test_s),
                         multi_class='ovr', average='macro')
print(f"ROC-AUC (OvR, macro): {auc_ovr:.4f}")</code></pre>

    <h3>৭. কোন Solver কোন পদ্ধতি সমর্থন করে?</h3>
    <table>
      <thead><tr><th>Solver</th><th>OvR</th><th>Multinomial</th><th>L1</th><th>L2</th><th>কখন ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>lbfgs</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>ছোট/মাঝারি ডেটা</td></tr>
        <tr><td>saga</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>বড় ডেটা, L1 দরকার হলে</td></tr>
        <tr><td>liblinear</td><td>✓</td><td>✗</td><td>✓</td><td>✓</td><td>ছোট ডেটা, L1</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>OvR</td><td>k টি binary classifier — সহজ কিন্তু probability সঠিক নয়</td></tr>
        <tr><td>OvO</td><td>k(k−1)/2 টি classifier — ধীর, বড় dataset-এ কম ব্যবহৃত</td></tr>
        <tr><td>Softmax</td><td>সব class একসাথে, calibrated probability — সবচেয়ে ভালো</td></tr>
        <tr><td>Macro avg</td><td>সব class সমান গুরুত্ব — imbalanced-এ বিভ্রান্তিকর</td></tr>
        <tr><td>Weighted avg</td><td>class size অনুযায়ী — imbalanced-এ বেশি উপযুক্ত</td></tr>
      </tbody>
    </table>
  `,
};
