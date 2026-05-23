export const svm_5_multiclass = {
  title: "SVM Multiclass Classification — OvR ও OvO Strategy",
  description: "SVM মূলত binary classifier — One-vs-Rest ও One-vs-One strategy দিয়ে কীভাবে multiclass সমস্যা সমাধান করে, sklearn-এ সম্পূর্ণ implementation।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 10,
  slug: "svm-multiclass",
  content: `
    <h3>১. SVM কি multiclass পারে?</h3>
    <p>SVM মূলত binary classifier। ৩ বা তার বেশি class-এর জন্য দুটো strategy ব্যবহার হয়: <strong>One-vs-Rest (OvR)</strong> এবং <strong>One-vs-One (OvO)</strong>।</p>
    <pre><code># ধরো K = 4 classes: A, B, C, D

# One-vs-Rest (OvR):
#   ৪টি binary classifier:
#   1. A vs {B,C,D}
#   2. B vs {A,C,D}
#   3. C vs {A,B,D}
#   4. D vs {A,B,C}
#   → সবচেয়ে বেশি confident class জেতে

# One-vs-One (OvO):
#   K(K-1)/2 = 4×3/2 = ৬টি binary classifier:
#   A vs B,  A vs C,  A vs D
#   B vs C,  B vs D,  C vs D
#   → majority vote দিয়ে final prediction</code></pre>

    <h3>২. sklearn-এ OvR ও OvO</h3>
    <pre><code>from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# sklearn SVC ডিফল্টে OvO ব্যবহার করে (decision_function_shape='ovr' হলে OvR)
strategies = {
    'SVC (OvO default)': Pipeline([
        ('s', StandardScaler()),
        ('m', SVC(kernel='rbf', C=1.0, gamma='scale',
                  decision_function_shape='ovo')),
    ]),
    'SVC (OvR)': Pipeline([
        ('s', StandardScaler()),
        ('m', SVC(kernel='rbf', C=1.0, gamma='scale',
                  decision_function_shape='ovr')),
    ]),
    'LinearSVC (OvR)': Pipeline([
        ('s', StandardScaler()),
        ('m', LinearSVC(C=1.0, max_iter=5000)),
    ]),
    'Explicit OvR': Pipeline([
        ('s', StandardScaler()),
        ('m', OneVsRestClassifier(SVC(kernel='rbf', C=1.0, gamma='scale'))),
    ]),
    'Explicit OvO': Pipeline([
        ('s', StandardScaler()),
        ('m', OneVsOneClassifier(SVC(kernel='rbf', C=1.0, gamma='scale'))),
    ]),
}

print(f"{'Strategy':25s}  {'Test Acc':>9}  {'CV Acc':>9}")
for name, pipe in strategies.items():
    pipe.fit(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    cv_acc   = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
    print(f"{name:25s}  {test_acc:>9.4f}  {cv_acc:>9.4f}")</code></pre>

    <h3>৩. Decision Function বিশ্লেষণ</h3>
    <pre><code">from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_s    = scaler.fit_transform(X_train)
X_ts   = scaler.transform(X_test)

# OvO decision function
clf_ovo = SVC(kernel='rbf', C=1.0, gamma='scale',
              decision_function_shape='ovo')
clf_ovo.fit(X_s, y_train)

df = clf_ovo.decision_function(X_ts)  # shape: (n_samples, K(K-1)/2)
print(f"OvO decision shape: {df.shape}")  # (30, 3) for 3 classes

# OvR decision function
clf_ovr = SVC(kernel='rbf', C=1.0, gamma='scale',
              decision_function_shape='ovr')
clf_ovr.fit(X_s, y_train)

df_ovr = clf_ovr.decision_function(X_ts)  # shape: (n_samples, K)
print(f"OvR decision shape: {df_ovr.shape}")  # (30, 3) for 3 classes

# Highest score = predicted class
preds_manual = np.argmax(df_ovr, axis=1)
print(f"Match sklearn predict: {(preds_manual == clf_ovr.predict(X_ts)).all()}")</code></pre>

    <h3>৪. Confusion Matrix ও Per-Class Metrics</h3>
    <pre><code">from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

pipe = Pipeline([
    ('s', StandardScaler()),
    ('m', SVC(kernel='rbf', C=5.0, gamma='scale', probability=True)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)

print(classification_report(y_test, y_pred,
      target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(colorbar=False)
plt.title('SVM Multiclass — Confusion Matrix')
plt.show()

# Multiclass AUC (OvR)
y_bin = label_binarize(y_test, classes=[0, 1, 2])
auc   = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
print(f"Macro ROC-AUC: {auc:.4f}")</code></pre>

    <h3>৫. OvR vs OvO তুলনা</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>One-vs-Rest (OvR)</th><th>One-vs-One (OvO)</th></tr></thead>
      <tbody>
        <tr><td>Classifier সংখ্যা</td><td>K</td><td>K(K-1)/2</td></tr>
        <tr><td>Training সময়</td><td>কম (K classifiers)</td><td>বেশি (K² classifiers)</td></tr>
        <tr><td>প্রতিটি classifier-এর data</td><td>Full (imbalanced)</td><td>ছোট (balanced)</td></tr>
        <tr><td>sklearn SVC default</td><td>decision_function_shape='ovr'</td><td>internally OvO</td></tr>
        <tr><td>বেশি ব্যবহার</td><td>LinearSVC, large K</td><td>SVC (kernel), small K</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>OvR</td><td>K binary classifiers — প্রতিটি class বাকি সবার বিরুদ্ধে</td></tr>
        <tr><td>OvO</td><td>K(K-1)/2 classifiers — প্রতিটি pair-এর বিরুদ্ধে</td></tr>
        <tr><td>sklearn SVC</td><td>internally OvO — decision_function_shape দিয়ে output control</td></tr>
        <tr><td>LinearSVC</td><td>OvR — large dataset-এ ভালো</td></tr>
        <tr><td>probability=True</td><td>Platt scaling দিয়ে multiclass probability পাওয়া যায়</td></tr>
      </tbody>
    </table>
  `,
};
