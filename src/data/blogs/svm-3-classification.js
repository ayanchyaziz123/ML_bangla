export const svm_3_classification = {
  title: "SVM Classification — sklearn SVC সম্পূর্ণ গাইড",
  description: "sklearn SVC ও LinearSVC দিয়ে binary ও multiclass classification, decision boundary visualization, probability calibration — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 11,
  slug: "svm-classification-python",
  content: `
    <h3>১. SVC vs LinearSVC</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>SVC</th><th>LinearSVC</th></tr></thead>
      <tbody>
        <tr><td>Kernel</td><td>linear, rbf, poly, sigmoid</td><td>শুধু linear</td></tr>
        <tr><td>Algorithm</td><td>libsvm (SMO)</td><td>liblinear</td></tr>
        <tr><td>Large dataset</td><td>ধীর O(n²–n³)</td><td>দ্রুত O(n)</td></tr>
        <tr><td>Probability</td><td>probability=True দিয়ে পাওয়া যায়</td><td>নেই (calibration দরকার)</td></tr>
        <tr><td>কখন ব্যবহার</td><td>n &lt; 10,000, non-linear data</td><td>n &gt; 10,000, text classification</td></tr>
      </tbody>
    </table>

    <h3>২. Basic SVC Pipeline</h3>
    <pre><code>from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline — Scaling গুরুত্বপূর্ণ! SVM scale-sensitive
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale',
                   probability=True, random_state=42)),
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred,
      target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# CV score
cv = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC:  {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>৩. Decision Boundary Visualization</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (kernel, params) in zip(axes, [
    ('linear', {}),
    ('poly',   {'degree': 3}),
    ('rbf',    {'gamma': 'scale'}),
]):
    clf = SVC(kernel=kernel, C=1.0, **params)
    clf.fit(X_s, y)

    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_s[y==0,0], X_s[y==0,1], c='steelblue', s=20, edgecolors='white', lw=0.5)
    ax.scatter(X_s[y==1,0], X_s[y==1,1], c='coral',     s=20, edgecolors='white', lw=0.5)

    # Support vectors হাইলাইট করো
    sv = clf.support_vectors_
    ax.scatter(sv[:,0], sv[:,1], s=80, facecolors='none',
               edgecolors='black', linewidths=1.5, label='Support Vectors')

    ax.set_title(f'SVM ({kernel})\nAcc={clf.score(X_s,y):.3f}, SVs={len(sv)}')
    ax.legend(fontsize=8)

plt.tight_layout(); plt.show()</code></pre>

    <h3>৪. Probability Calibration</h3>
    <pre><code">from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# SVC(probability=True) Platt scaling ব্যবহার করে
# কিন্তু ছোট dataset-এ poorly calibrated হতে পারে

# Better approach: CalibratedClassifierCV
svc_base = SVC(kernel='rbf', C=1.0, gamma='scale')
svc_cal  = CalibratedClassifierCV(svc_base, method='isotonic', cv=5)

svc_cal.fit(X_train, y_train)
y_prob_cal = svc_cal.predict_proba(X_test)[:, 1]

# Brier Score: 0 = perfect, 1 = worst
bs = brier_score_loss(y_test, y_prob_cal)
print(f"Brier Score (calibrated): {bs:.4f}")
print(f"Accuracy: {svc_cal.score(X_test, y_test):.4f}")</code></pre>

    <h3>৫. LinearSVC — Large Dataset-এর জন্য</h3>
    <pre><code">from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Text classification উদাহরণ
categories = ['sci.med', 'sci.space', 'rec.sport.hockey', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers','footers','quotes'))

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True)),
    ('svm',   LinearSVC(C=0.1, max_iter=2000)),
])
pipe.fit(train.data, train.target)
y_pred = pipe.predict(test.data)
print(classification_report(test.target, y_pred, target_names=categories))</code></pre>

    <h3>৬. Support Vectors বিশ্লেষণ</h3>
    <pre><code">from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_s    = scaler.fit_transform(X_train)
clf    = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_s, y_train)

print(f"মোট support vectors:     {clf.n_support_.sum()}")
print(f"Class-wise:              {clf.n_support_}")
print(f"Training set-এর %:      {100*clf.n_support_.sum()/len(X_train):.1f}%")
print(f"Support vector indices:  {clf.support_[:5]} ...")

# কম support vector → simpler model (better generalization)
# বেশি support vector → complex model (possible overfit)</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Scaling</td><td>SVM সবসময় StandardScaler দিয়ে scale করতে হবে</td></tr>
        <tr><td>SVC</td><td>ছোট dataset, non-linear — rbf kernel ব্যবহার করো</td></tr>
        <tr><td>LinearSVC</td><td>বড় dataset, text data — অনেক দ্রুত</td></tr>
        <tr><td>probability=True</td><td>Platt scaling — ধীর কিন্তু probability দেয়</td></tr>
        <tr><td>Support vectors কম</td><td>ভালো generalization-এর লক্ষণ</td></tr>
      </tbody>
    </table>
  `,
};
