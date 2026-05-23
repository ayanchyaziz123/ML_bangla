export const nb_2_gaussian = {
  title: "Gaussian Naive Bayes — Continuous ফিচারে Normal Distribution",
  description: "Continuous ফিচারে Naive Bayes কীভাবে Normal Distribution ব্যবহার করে, Python implementation, এবং কখন Gaussian NB সেরা পছন্দ — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 10,
  slug: "nb-gaussian-naive-bayes",
  content: `
    <h3>১. Gaussian NB কীভাবে কাজ করে?</h3>
    <p>Multinomial NB শব্দ গণনা করে। কিন্তু temperature, height, weight — এই continuous মান-গুলো গণনা করা যায় না। Gaussian NB ধরে নেয় প্রতিটি feature প্রতিটি class-এর জন্য <strong>Normal (Gaussian) Distribution</strong> অনুসরণ করে।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Gaussian PDF (Probability Density Function):
# P(x | class) = (1 / √(2πσ²)) × exp(−(x−μ)²/2σ²)

# উদাহরণ: দুটি class-এর height distribution
mu_male,   sigma_male   = 170, 8
mu_female, sigma_female = 158, 7

x = np.linspace(130, 200, 200)
plt.figure(figsize=(8, 4))
plt.plot(x, norm.pdf(x, mu_male,   sigma_male),   label='পুরুষ', color='steelblue')
plt.plot(x, norm.pdf(x, mu_female, sigma_female), label='নারী',  color='coral')
plt.xlabel('উচ্চতা (cm)')
plt.ylabel('Probability Density')
plt.title('Gaussian NB: প্রতিটি class-এর feature distribution')
plt.legend()
plt.show()

# P(x=165 | পুরুষ):
p_male   = norm.pdf(165, mu_male,   sigma_male)
p_female = norm.pdf(165, mu_female, sigma_female)
print(f"P(height=165 | পুরুষ)  = {p_male:.5f}")
print(f"P(height=165 | নারী)   = {p_female:.5f}")</code></pre>

    <h3>২. Training: μ ও σ হিসাব</h3>
    <pre><code">import numpy as np

# Gaussian NB training:
# প্রতিটি class-এর প্রতিটি feature-এর mean ও variance মনে রাখো

X_train = np.array([
    [170, 70],   # পুরুষ
    [175, 80],
    [168, 65],
    [158, 55],   # নারী
    [160, 52],
    [155, 48],
])
y_train = np.array([1, 1, 1, 0, 0, 0])  # 1=পুরুষ, 0=নারী

# ক্লাস অনুযায়ী statistics
for cls in [0, 1]:
    label = "পুরুষ" if cls == 1 else "নারী"
    X_cls = X_train[y_train == cls]
    mu    = X_cls.mean(axis=0)
    sigma = X_cls.std(axis=0)
    print(f"\\n{label}:")
    print(f"  Mean   = {mu}")
    print(f"  Std    = {sigma}")</code></pre>

    <h3>৩. Prediction: Gaussian PDF দিয়ে Likelihood</h3>
    <pre><code">from scipy.stats import norm

# নতুন sample: height=162, weight=56
x_new = np.array([162, 56])

# P(পুরুষ) = P(নারী) = 0.5 (equal prior)
log_prior_male   = np.log(0.5)
log_prior_female = np.log(0.5)

log_likelihood_male   = 0
log_likelihood_female = 0

# প্রতিটি feature-এর জন্য:
for feat_idx, x_val in enumerate(x_new):
    # পুরুষের distribution থেকে likelihood
    X_male   = X_train[y_train == 1][:, feat_idx]
    mu_m, s_m = X_male.mean(), X_male.std()
    log_likelihood_male += np.log(norm.pdf(x_val, mu_m, s_m) + 1e-10)

    # নারীর distribution থেকে likelihood
    X_female = X_train[y_train == 0][:, feat_idx]
    mu_f, s_f = X_female.mean(), X_female.std()
    log_likelihood_female += np.log(norm.pdf(x_val, mu_f, s_f) + 1e-10)

log_posterior_male   = log_prior_male   + log_likelihood_male
log_posterior_female = log_prior_female + log_likelihood_female

print(f"log P(পুরুষ | x) ∝ {log_posterior_male:.3f}")
print(f"log P(নারী  | x) ∝ {log_posterior_female:.3f}")
print(f"Prediction: {'পুরুষ' if log_posterior_male > log_posterior_female else 'নারী'}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Gaussian NB — Training ও Prediction</text>
        <rect x="15" y="28" width="130" height="75" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="80" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">Training</text>
        <text x="80" y="62" text-anchor="middle" font-size="8" fill="#3b82f6">প্রতিটি class-এ</text>
        <text x="80" y="75" text-anchor="middle" font-size="8" fill="#3b82f6">প্রতিটি feature-এর</text>
        <text x="80" y="88" text-anchor="middle" font-size="8" fill="#3b82f6">μ ও σ সংরক্ষণ</text>
        <text x="153" y="68" font-size="14" fill="#94a3b8">→</text>
        <rect x="168" y="28" width="145" height="75" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="240" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Prediction</text>
        <text x="240" y="62" text-anchor="middle" font-size="8" fill="#d97706">Gaussian PDF দিয়ে</text>
        <text x="240" y="75" text-anchor="middle" font-size="8" fill="#d97706">প্রতিটি feature-এর</text>
        <text x="240" y="88" text-anchor="middle" font-size="8" fill="#d97706">likelihood হিসাব</text>
        <text x="321" y="68" font-size="14" fill="#94a3b8">→</text>
        <rect x="336" y="28" width="129" height="75" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="400" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">argmax class</text>
        <text x="400" y="62" text-anchor="middle" font-size="8" fill="#16a34a">P(class) ×</text>
        <text x="400" y="75" text-anchor="middle" font-size="8" fill="#16a34a">Π P(xᵢ|class)</text>
        <text x="400" y="88" text-anchor="middle" font-size="8" fill="#16a34a">সর্বোচ্চটি জেতে</text>
      </svg>
    </div>

    <h3>৪. sklearn দিয়ে Gaussian NB</h3>
    <pre><code">from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gaussian NB — Scaling দরকার নেই কিন্তু করা যায়
gnb = GaussianNB(
    var_smoothing=1e-9,   # numerical stability-র জন্য
)
gnb.fit(X_train, y_train)

print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")
print(f"CV Accuracy:   {cross_val_score(gnb, X, y, cv=5).mean():.4f}")
print()
print(classification_report(y_test, gnb.predict(X_test),
      target_names=data.target_names))

# মডেল শিখেছে:
print("প্রতিটি class-এর Mean:")
for i, cls in enumerate(data.target_names):
    print(f"  {cls}: {gnb.theta_[i].round(2)}")

print("\\nপ্রতিটি class-এর Variance:")
for i, cls in enumerate(data.target_names):
    print(f"  {cls}: {gnb.var_[i].round(3)}")</code></pre>

    <h3>৫. var_smoothing Tuning</h3>
    <pre><code">from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {'var_smoothing': np.logspace(-12, 0, 100)}

grid = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"সেরা var_smoothing: {grid.best_params_['var_smoothing']:.2e}")
print(f"Test Accuracy:      {grid.score(X_test, y_test):.4f}")</code></pre>

    <h3>৬. কখন Gaussian NB ব্যবহার করবে?</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>Gaussian NB উপযুক্ত?</th></tr></thead>
      <tbody>
        <tr><td>Continuous numeric features</td><td>হ্যাঁ — সবচেয়ে ভালো</td></tr>
        <tr><td>Features normally distributed</td><td>হ্যাঁ — assumption মিলে</td></tr>
        <tr><td>Training data কম</td><td>হ্যাঁ — কম data-তেও ভালো কাজ করে</td></tr>
        <tr><td>Real-time prediction দরকার</td><td>হ্যাঁ — অত্যন্ত দ্রুত</td></tr>
        <tr><td>Features strongly correlated</td><td>না — independence assumption ভাঙে</td></tr>
        <tr><td>Non-Gaussian distribution</td><td>সতর্কতার সাথে — accuracy কমতে পারে</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Gaussian NB</td><td>continuous feature-এ Normal distribution ধরে</td></tr>
        <tr><td>Training</td><td>প্রতিটি class-feature-এর μ ও σ মনে রাখে</td></tr>
        <tr><td>Likelihood</td><td>Gaussian PDF দিয়ে P(x|class) হিসাব</td></tr>
        <tr><td>var_smoothing</td><td>numerical stability-র জন্য, GridSearch দিয়ে tune করো</td></tr>
        <tr><td>সুবিধা</td><td>দ্রুত, কম data-তে ভালো, incremental learning সমর্থন করে</td></tr>
      </tbody>
    </table>
  `,
};
