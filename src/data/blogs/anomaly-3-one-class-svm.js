export const anomaly_3_one_class_svm = {
  slug: 'anomaly-3-one-class-svm',
  title: 'One-Class SVM: স্বাভাবিক ডেটার সীমানা',
  description: 'One-Class Classification-এর ধারণা, nu প্যারামিটার, RBF kernel, Isolation Forest-এর সাথে তুলনা এবং sklearn OneClassSVM দিয়ে বাস্তব উদাহরণ।',
  date: 'মে ২০২৬',
  category: 'অ্যানোমালি ডিটেকশন',
  readTime: 12,
  content: `
    <h3>১. One-Class Classification কী?</h3>
    <p>
      সাধারণ মেশিন লার্নিং classification-এ আমরা একাধিক ক্লাসের উদাহরণ দিয়ে মডেল তৈরি করি। কিন্তু অনেক বাস্তব সমস্যায় আমাদের কাছে শুধুমাত্র <strong>একটি ক্লাসের ডেটা</strong> থাকে — স্বাভাবিক ডেটা।
    </p>
    <p>
      উদাহরণ:
    </p>
    <ul>
      <li>ব্যাংকে সাধারণ লেনদেনের লক্ষ লক্ষ উদাহরণ আছে, কিন্তু জালিয়াতির উদাহরণ অনেক কম</li>
      <li>কারখানায় সুস্থ যন্ত্রের ডেটা আছে, কিন্তু নষ্ট যন্ত্রের ডেটা নেই</li>
      <li>নেটওয়ার্কে স্বাভাবিক ট্রাফিকের ডেটা আছে, কিন্তু নতুন ধরনের আক্রমণের ডেটা নেই</li>
    </ul>
    <p>
      <strong>One-Class SVM</strong> এই সমস্যা সমাধান করে — শুধুমাত্র স্বাভাবিক ডেটা দিয়ে একটি decision boundary শিখে, এবং নতুন ডেটা সেই boundary-র ভেতরে না বাইরে তা দেখে অ্যানোমালি সনাক্ত করে।
    </p>

    <h3>২. One-Class SVM-এর কার্যপ্রণালী</h3>
    <p>
      প্রচলিত SVM দুটি ক্লাসের মধ্যে সর্বাধিক margin তৈরি করে। One-Class SVM আলাদাভাবে কাজ করে:
    </p>
    <ol>
      <li>ডেটাকে একটি high-dimensional feature space-এ map করা হয় (kernel দিয়ে)</li>
      <li>সেই space-এ origin থেকে সর্বাধিক margin তৈরি করে একটি hyperplane আঁকা হয়</li>
      <li>স্বাভাবিক ডেটা hyperplane-এর একপাশে থাকে</li>
      <li>নতুন পয়েন্ট hyperplane-এর ভুল পাশে থাকলে = অ্যানোমালি</li>
    </ol>
    <p>
      গাণিতিকভাবে, One-Class SVM নিচের optimization সমস্যা সমাধান করে:
    </p>
    <p>
      <strong>Minimize: ½||w||² + (1/νn)Σξᵢ - ρ</strong>
    </p>
    <p>
      যেখানে ν (nu) = অ্যানোমালির উপরের সীমা, ρ = threshold, ξ = slack variables।
    </p>

    <h3>৩. nu প্যারামিটার: আউটলায়ারের অনুপাত</h3>
    <p>
      <code>nu</code> প্যারামিটার (0 থেকে 1 এর মধ্যে) দুটি অর্থ বহন করে:
    </p>
    <ul>
      <li><strong>Training error-এর উপরের সীমা:</strong> training data-র মধ্যে সর্বাধিক nu% পয়েন্ট boundary-র ভুল পাশে থাকতে পারে</li>
      <li><strong>Support Vector-এর নিচের সীমা:</strong> training data-র কমপক্ষে nu% পয়েন্ট support vector হবে</li>
    </ul>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# ডেটা তৈরি
np.random.seed(42)
X_train = np.random.normal(0, 1, (200, 2))

# বিভিন্ন nu মানে তুলনা
nu_values = [0.01, 0.05, 0.1, 0.2]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))

for ax, nu in zip(axes, nu_values):
    oc_svm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    oc_svm.fit(X_train)

    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.scatter(X_train[:, 0], X_train[:, 1], s=10, c='blue', alpha=0.5)
    ax.set_title(f'nu = {nu}')

plt.suptitle('Effect of nu Parameter on Decision Boundary', fontsize=13)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. RBF Kernel: Gamma প্যারামিটার</h3>
    <p>
      One-Class SVM-এ <strong>RBF (Radial Basis Function) kernel</strong> সবচেয়ে বেশি ব্যবহৃত হয়। এটি non-linear boundary তৈরি করতে পারে।
    </p>
    <p>
      <strong>K(x, z) = exp(-γ||x - z||²)</strong>
    </p>
    <p>
      <code>gamma</code> প্যারামিটার নিয়ন্ত্রণ করে boundary কতটা "tight" হবে:
    </p>
    <ul>
      <li><strong>High gamma:</strong> খুব tight boundary, সহজে overfitting</li>
      <li><strong>Low gamma:</strong> loose boundary, underfitting হতে পারে</li>
      <li><strong>'scale':</strong> gamma = 1/(n_features × X.var()) — sklearn-এর recommended default</li>
    </ul>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
# মুন-আকৃতির ক্লাস্টার
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
X = X[:100]  # শুধু স্বাভাবিক ডেটা

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gamma তুলনা
gamma_values = [0.01, 0.1, 1.0, 10.0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))

for ax, gam in zip(axes, gamma_values):
    oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma=gam)
    oc_svm.fit(X_scaled)

    Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], s=20, c='green', alpha=0.7)
    ax.set_title(f'gamma = {gam}')

plt.suptitle('Effect of Gamma Parameter (RBF Kernel)', fontsize=13)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. sklearn OneClassSVM: সম্পূর্ণ উদাহরণ</h3>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

np.random.seed(42)

# নেটওয়ার্ক ট্রাফিক ডেটা তৈরি করা
n_normal = 500
normal_traffic = pd.DataFrame({
    'packet_size': np.random.normal(512, 100, n_normal),
    'request_rate': np.random.normal(50, 10, n_normal),
    'response_time': np.random.normal(100, 20, n_normal),
    'error_count': np.random.poisson(2, n_normal).astype(float)
})

n_attack = 50
attack_traffic = pd.DataFrame({
    'packet_size': np.random.normal(64, 10, n_attack),   # DDoS: ছোট প্যাকেট
    'request_rate': np.random.normal(5000, 200, n_attack), # অনেক বেশি রিকোয়েস্ট
    'response_time': np.random.normal(5000, 500, n_attack),
    'error_count': np.random.normal(100, 20, n_attack)
})

# True labels (evaluation-এর জন্য)
y_true = np.array([1] * n_normal + [-1] * n_attack)
X = pd.concat([normal_traffic, attack_traffic]).values

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(normal_traffic.values)
X_test_scaled = scaler.transform(X)

# One-Class SVM
oc_svm = OneClassSVM(
    kernel='rbf',
    nu=0.05,
    gamma='scale'
)
oc_svm.fit(X_train_scaled)

# Prediction
y_pred = oc_svm.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_true, y_pred,
      target_names=['Attack (-1)', 'Normal (1)']))

# Score distribution
scores = oc_svm.decision_function(X_test_scaled)
print(f"\\nNormal traffic score (mean): {scores[:n_normal].mean():.3f}")
print(f"Attack traffic score (mean): {scores[n_normal:].mean():.3f}")</code></pre>

    <h3>৬. Isolation Forest বনাম One-Class SVM</h3>
    <p>
      দুটি পদ্ধতির মধ্যে পার্থক্য বোঝা গুরুত্বপূর্ণ:
    </p>
    <pre><code>import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
sizes = [100, 500, 1000, 5000, 10000]

print(f"{'Dataset Size':15} {'OneClassSVM (s)':18} {'IsolationForest (s)':20}")
print("-" * 55)

for n in sizes:
    X = np.random.normal(0, 1, (n, 10))
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # One-Class SVM
    t1 = time.time()
    oc = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    oc.fit(X_s)
    t_oc = time.time() - t1

    # Isolation Forest
    t2 = time.time()
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_s)
    t_iso = time.time() - t2

    print(f"{n:15} {t_oc:18.4f} {t_iso:20.4f}")</code></pre>
    <p>
      তুলনামূলক বিশ্লেষণ:
    </p>
    <ul>
      <li><strong>গতি:</strong> Isolation Forest অনেক দ্রুত, বিশেষত বড় ডেটাসেটে</li>
      <li><strong>নির্ভুলতা:</strong> One-Class SVM complex non-linear boundaries শিখতে পারে</li>
      <li><strong>Scalability:</strong> Isolation Forest O(n) — One-Class SVM O(n²) থেকে O(n³)</li>
      <li><strong>Parameter sensitivity:</strong> One-Class SVM gamma ও nu দুটোতেই sensitive</li>
      <li><strong>High-dimensional data:</strong> Isolation Forest সাধারণত ভালো করে</li>
    </ul>

    <h3>৭. Hyperparameter Tuning</h3>
    <pre><code>import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

np.random.seed(42)
X_normal = np.random.normal(0, 1, (300, 5))
X_anomaly = np.random.uniform(-5, 5, (30, 5))
X = np.vstack([X_normal, X_anomaly])
y = np.array([1]*300 + [-1]*30)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual grid search (OneClassSVM GridSearchCV সরাসরি সমর্থন করে না)
best_f1 = 0
best_params = {}

for nu in [0.01, 0.05, 0.1, 0.15]:
    for gamma in [0.01, 0.1, 1.0, 'scale']:
        oc = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        oc.fit(X_scaled[:300])  # শুধু normal দিয়ে train
        preds = oc.predict(X_scaled)
        f1 = f1_score(y, preds, pos_label=-1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = {'nu': nu, 'gamma': gamma}

print(f"Best F1: {best_f1:.4f}")
print(f"Best Parameters: {best_params}")</code></pre>

    <h3>৮. কখন One-Class SVM ব্যবহার করবেন?</h3>
    <p>
      One-Class SVM উপযুক্ত যখন:
    </p>
    <ul>
      <li>ডেটাসেট ছোট থেকে মাঝারি (10,000-এর কম)</li>
      <li>Complex non-linear boundary প্রয়োজন</li>
      <li>High precision চাই (False Positive কম চাই)</li>
      <li>ডেটা preprocessing ও scaling করা সম্ভব</li>
    </ul>
    <p>
      Isolation Forest উপযুক্ত যখন:
    </p>
    <ul>
      <li>বড় ডেটাসেট (লক্ষাধিক পয়েন্ট)</li>
      <li>দ্রুত inference দরকার</li>
      <li>Interpretability গুরুত্বপূর্ণ</li>
      <li>High-dimensional ডেটা</li>
    </ul>
  `
};
