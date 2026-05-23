export const svm_4_svr = {
  title: "Support Vector Regression (SVR) — Continuous Value Prediction",
  description: "SVR কীভাবে epsilon-insensitive tube দিয়ে regression করে, sklearn SVR ও NuSVR, এবং Linear Regression-এর সাথে পার্থক্য বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 10,
  slug: "svm-regression-svr",
  content: `
    <h3>১. SVR কীভাবে কাজ করে?</h3>
    <p>Linear Regression প্রতিটি point-এর error minimize করে। SVR একটি <strong>ε-tube</strong> তৈরি করে — এই tube-এর ভেতরের points-এর কোনো error নেই। শুধু tube-এর বাইরের points penalty পায়।</p>
    <pre><code># SVR Objective:
# Minimize:  (1/2)||w||² + C Σᵢ (ξᵢ + ξᵢ*)
#
# Subject to:  yᵢ - (w·xᵢ + b) ≤ ε + ξᵢ    (upper tube boundary)
#              (w·xᵢ + b) - yᵢ ≤ ε + ξᵢ*   (lower tube boundary)
#              ξᵢ, ξᵢ* ≥ 0
#
# ε = tube-এর প্রস্থ (এর মধ্যে error = 0)
# C = tube-এর বাইরের violation-এর penalty
# ξ, ξ* = slack variables (tube-এর উপরে ও নিচে violation)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="15" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">SVR — ε-Insensitive Tube</text>
        <!-- Regression line -->
        <line x1="40" y1="90" x2="440" y2="40" stroke="#111827" stroke-width="2"/>
        <!-- Upper tube -->
        <line x1="40" y1="72" x2="440" y2="22" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,3"/>
        <!-- Lower tube -->
        <line x1="40" y1="108" x2="440" y2="58" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,3"/>
        <!-- Points inside tube (no loss) -->
        <circle cx="100" cy="80" r="5" fill="#16a34a"/>
        <circle cx="180" cy="65" r="5" fill="#16a34a"/>
        <circle cx="260" cy="58" r="5" fill="#16a34a"/>
        <circle cx="330" cy="48" r="5" fill="#16a34a"/>
        <!-- Points outside tube (loss) -->
        <circle cx="150" cy="50" r="5" fill="#ef4444"/>
        <line x1="150" y1="50" x2="150" y2="63" stroke="#ef4444" stroke-width="1.5"/>
        <text x="158" y="48" font-size="8" fill="#ef4444">ξ</text>
        <circle cx="380" cy="75" r="5" fill="#ef4444"/>
        <line x1="380" y1="75" x2="380" y2="63" stroke="#ef4444" stroke-width="1.5"/>
        <text x="388" y="78" font-size="8" fill="#ef4444">ξ*</text>
        <!-- Labels -->
        <text x="450" y="38" font-size="8" fill="#111827">f(x)</text>
        <text x="450" y="20" font-size="8" fill="#3b82f6">+ε</text>
        <text x="450" y="57" font-size="8" fill="#3b82f6">-ε</text>
        <text x="90" y="120" font-size="8" fill="#16a34a">● inside tube: no loss</text>
        <text x="270" y="120" font-size="8" fill="#ef4444">● outside: ξ loss</text>
      </svg>
    </div>

    <h3>২. sklearn SVR</h3>
    <pre><code>from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()
X, y = data.data[:2000], data.target[:2000]  # SVR is slow on large n

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(
        kernel='rbf',
        C=10,        # penalty for violations outside tube
        epsilon=0.1, # tube width — predictions within ε of y have 0 loss
        gamma='scale',
    )),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

cv = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"CV R²: {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>৩. epsilon-এর প্রভাব</h3>
    <pre><code">from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_s = scaler.fit_transform(X_train)
X_ts = scaler.transform(X_test)

print(f"{'epsilon':>9}  {'R²':>8}  {'Support Vectors':>16}")
for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
    svr = SVR(kernel='rbf', C=10, epsilon=eps, gamma='scale')
    svr.fit(X_s, y_train)
    r2 = r2_score(y_test, svr.predict(X_ts))
    print(f"{eps:>9.2f}  {r2:>8.4f}  {svr.n_support_[0]:>16}")

# epsilon ছোট → বেশি support vectors → complex model
# epsilon বড়  → কম support vectors → simple model</code></pre>

    <h3>৪. NuSVR — ν দিয়ে Support Vector সংখ্যা Control</h3>
    <pre><code">from sklearn.svm import NuSVR

# NuSVR: epsilon-এর বদলে ν ব্যবহার করে
# ν ∈ (0, 1]: support vector-এর fraction-এর upper bound
# ν = 0.5 → training set-এর ≥ 50% support vectors

for nu in [0.1, 0.3, 0.5, 0.7, 0.9]:
    svr = NuSVR(nu=nu, C=10, kernel='rbf', gamma='scale')
    svr.fit(X_s, y_train)
    r2 = r2_score(y_test, svr.predict(X_ts))
    frac = svr.n_support_[0] / len(X_train)
    print(f"ν={nu:.1f}  SVs={svr.n_support_[0]} ({frac:.1%})  R²={r2:.4f}")</code></pre>

    <h3>৫. SVR vs Linear Regression তুলনা</h3>
    <pre><code">from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)':       Ridge(alpha=1.0),
    'SVR (rbf)':         SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
    'SVR (linear)':      SVR(kernel='linear', C=1.0, epsilon=0.1),
}

for name, model in models.items():
    pipe = Pipeline([('s', StandardScaler()), ('m', model)])
    pipe.fit(X_train, y_train)
    r2   = r2_score(y_test, pipe.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
    print(f"{name:22s}  R²={r2:.4f}  RMSE={rmse:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>ε-tube</td><td>এর ভেতরে prediction error = 0, বাইরে penalty</td></tr>
        <tr><td>C</td><td>tube-এর বাইরের violation-এর penalty</td></tr>
        <tr><td>epsilon</td><td>tube-এর প্রস্থ — বড় হলে কম SVs, সহজ model</td></tr>
        <tr><td>NuSVR</td><td>ν দিয়ে support vector fraction control করো</td></tr>
        <tr><td>SVR vs LR</td><td>SVR outlier-এর প্রতি কম sensitive</td></tr>
      </tbody>
    </table>
  `,
};
