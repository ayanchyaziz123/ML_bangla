export const lr_16_elastic_net = {
  title: "Elastic Net — Ridge ও Lasso-র সেরা সমন্বয়",
  description: "Elastic Net কীভাবে L1 ও L2 Regularization একসাথে ব্যবহার করে, কখন Ridge বা Lasso-র চেয়ে ভালো, এবং Python-এ সম্পূর্ণ implementation।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 9,
  slug: "lr-elastic-net",
  content: `
    <h3>১. সমস্যাটা কোথায়?</h3>
    <p>আমরা জানি:</p>
    <ul>
      <li><strong>Ridge (L2):</strong> সব coefficient ছোট রাখে, কোনোটা শূন্য করে না</li>
      <li><strong>Lasso (L1):</strong> কিছু coefficient শূন্য করে — feature selection করে</li>
    </ul>
    <p>কিন্তু Lasso-র একটি সমস্যা আছে: যদি অনেক correlated ফিচার থাকে, Lasso তাদের মধ্যে যেকোনো একটি বেছে নেয় এবং বাকিগুলো বাদ দেয় — কোনটি বেছে নেবে তা random হতে পারে।</p>
    <p><strong>Elastic Net</strong> এই সমস্যার সমাধান — এটি L1 এবং L2 দুটো penalty একসাথে ব্যবহার করে।</p>

    <h3>২. Elastic Net-এর সূত্র</h3>
    <pre><code># Loss Function:
Loss = MSE + α × [ρ × Σ|mᵢ| + (1−ρ)/2 × Σmᵢ²]
                   ↑ L1 (Lasso)    ↑ L2 (Ridge)

যেখানে:
  α (alpha)  = regularization strength (কতটুকু penalty)
  ρ (l1_ratio) = L1 ও L2-এর অনুপাত
    ρ = 1.0 → pure Lasso
    ρ = 0.0 → pure Ridge
    ρ = 0.5 → সমান ভাগ (সবচেয়ে সাধারণ)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Regularization তুলনা</text>
        <rect x="20"  y="30" width="120" height="55" rx="6" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
        <text x="80"  y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#1e40af">Ridge (L2)</text>
        <text x="80"  y="68" text-anchor="middle" font-size="9" fill="#3b82f6">coefficient ছোট</text>
        <text x="80"  y="80" text-anchor="middle" font-size="9" fill="#3b82f6">শূন্য নয়</text>
        <rect x="180" y="30" width="120" height="55" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1"/>
        <text x="240" y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#16a34a">Lasso (L1)</text>
        <text x="240" y="68" text-anchor="middle" font-size="9" fill="#22c55e">কিছু শূন্য</text>
        <text x="240" y="80" text-anchor="middle" font-size="9" fill="#22c55e">feature selection</text>
        <rect x="340" y="30" width="120" height="55" rx="6" fill="#fef3c7" stroke="#fcd34d" stroke-width="1"/>
        <text x="400" y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#d97706">Elastic Net</text>
        <text x="400" y="68" text-anchor="middle" font-size="9" fill="#f59e0b">L1 + L2 দুটোই</text>
        <text x="400" y="80" text-anchor="middle" font-size="9" fill="#f59e0b">সেরা সমন্বয়</text>
      </svg>
    </div>

    <h3>৩. Python দিয়ে Elastic Net</h3>
    <pre><code>from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
# শুধু ৫টি ফিচার আসলে গুরুত্বপূর্ণ
true_coef = np.zeros(p)
true_coef[:5] = [3, -2, 1.5, -1, 2]
y = X @ true_coef + np.random.randn(n) * 0.5

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Elastic Net
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
])
pipe.fit(X_train, y_train)

coefs = pipe.named_steps['model'].coef_
print("Coefficients:")
for i, c in enumerate(coefs):
    status = "✓ গুরুত্বপূর্ণ" if abs(c) > 0.1 else "✗ বাদ"
    print(f"  X{i+1:2d}: {c:6.3f}  {status}")

print(f"\\nTest R²: {pipe.score(X_test, y_test):.4f}")</code></pre>

    <h3>৪. সেরা Alpha ও l1_ratio খোঁজা</h3>
    <pre><code># ElasticNetCV — cross-validation দিয়ে সেরা hyperparameter খোঁজে
from sklearn.linear_model import ElasticNetCV

alphas   = np.logspace(-3, 1, 50)     # 0.001 থেকে 10
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

enet_cv = ElasticNetCV(
    alphas=alphas,
    l1_ratio=l1_ratios,
    cv=5,
    max_iter=10000,
    random_state=42
)

pipe_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  enet_cv),
])
pipe_cv.fit(X_train, y_train)

best_alpha    = pipe_cv.named_steps['model'].alpha_
best_l1_ratio = pipe_cv.named_steps['model'].l1_ratio_
print(f"সেরা alpha:    {best_alpha:.4f}")
print(f"সেরা l1_ratio: {best_l1_ratio:.2f}")
print(f"Test R²:       {pipe_cv.score(X_test, y_test):.4f}")</code></pre>

    <h3>৫. তিনটি পদ্ধতির তুলনা</h3>
    <pre><code>from sklearn.linear_model import Ridge, Lasso

models = {
    'Ridge':       Ridge(alpha=0.1),
    'Lasso':       Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('m', model)])
    pipe.fit(X_train, y_train)
    coefs = pipe.named_steps['m'].coef_
    n_zero = np.sum(np.abs(coefs) < 1e-4)
    r2 = pipe.score(X_test, y_test)
    print(f"{name:12s} | শূন্য coefficient: {n_zero:2d}/20 | R²: {r2:.4f}")</code></pre>

    <h3>৬. কখন Elastic Net বেছে নেবে?</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>সেরা পছন্দ</th></tr></thead>
      <tbody>
        <tr><td>ফিচার correlated এবং feature selection দরকার</td><td><strong>Elastic Net</strong></td></tr>
        <tr><td>সব ফিচার গুরুত্বপূর্ণ, শুধু coefficient ছোট রাখতে হবে</td><td>Ridge</td></tr>
        <tr><td>অনেক ফিচার, বেশিরভাগ অপ্রয়োজনীয়</td><td>Lasso</td></tr>
        <tr><td>কোনটা ভালো বুঝতে পারছো না</td><td>Elastic Net (ডিফল্ট হিসেবে নিরাপদ)</td></tr>
        <tr><td>p &gt; n (ফিচার সংখ্যা &gt; sample সংখ্যা)</td><td><strong>Elastic Net</strong> (Lasso unstable হয়)</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Elastic Net = L1 + L2</td><td>Ridge ও Lasso-র সুবিধা একসাথে</td></tr>
        <tr><td>l1_ratio = 1</td><td>pure Lasso</td></tr>
        <tr><td>l1_ratio = 0</td><td>pure Ridge</td></tr>
        <tr><td>l1_ratio = 0.5</td><td>সমান ভাগ — ভালো starting point</td></tr>
        <tr><td>Correlated features</td><td>Lasso random বাছে, Elastic Net দলগতভাবে রাখে বা বাদ দেয়</td></tr>
        <tr><td>ElasticNetCV</td><td>cross-validation দিয়ে সেরা alpha ও l1_ratio খোঁজো</td></tr>
      </tbody>
    </table>
  `,
};
