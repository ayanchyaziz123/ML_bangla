export const lr_7_polynomial = {
  title: "Polynomial Regression — যখন সরল রেখা যথেষ্ট নয়",
  description: "Linear regression কখন কাজ করে না? Polynomial regression কী এবং কীভাবে curved সম্পর্ক ধরা যায় — degree নির্বাচন, overfitting এবং Python কোড সহ।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 9,
  slug: "lr-polynomial-regression",
  content: `
    <h3>১. কখন Linear Regression কাজ করে না?</h3>
    <p>সব সম্পর্ক সরল রেখার মতো নয়। উদাহরণ:</p>
    <ul>
      <li>গাছের বয়স বাড়লে উচ্চতা প্রথমে দ্রুত বাড়ে, পরে ধীর হয়</li>
      <li>তাপমাত্রা এবং বিক্রির সম্পর্ক — অনেক গরম বা ঠান্ডায় বিক্রি কমে, মাঝারিতে বাড়ে</li>
      <li>ড্রাগের ডোজ এবং কার্যকারিতা</li>
    </ul>
    <p>এই ধরনের curved সম্পর্কে linear regression ব্যর্থ হয় — তখন <strong>Polynomial Regression</strong> দরকার।</p>

    <h3>২. Polynomial Regression কী?</h3>
    <pre><code># Linear:    Y = m₁X + b
# Quadratic: Y = m₁X + m₂X² + b          (degree 2)
# Cubic:     Y = m₁X + m₂X² + m₃X³ + b   (degree 3)

# মজার বিষয়: Polynomial Regression আসলে Linear Regression-ই!
# শুধু X² এবং X³-কে নতুন ফিচার হিসেবে ধরা হয়

# X = [2, 4, 6] হলে degree=2 করলে:
# ফিচার হয়: [X, X²] = [[2,4], [4,16], [6,36]]
# তারপর সাধারণ Linear Regression চালানো হয়</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 500 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="250" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Linear vs Polynomial Fit</text>
        <line x1="30" y1="140" x2="470" y2="140" stroke="#e5e7eb" stroke-width="1.5"/>
        <line x1="30" y1="20"  x2="30"  y2="145" stroke="#e5e7eb" stroke-width="1.5"/>
        <!-- Curved data points -->
        <circle cx="55"  cy="125" r="4" fill="#1e40af"/>
        <circle cx="95"  cy="100" r="4" fill="#1e40af"/>
        <circle cx="135" cy="68"  r="4" fill="#1e40af"/>
        <circle cx="175" cy="45"  r="4" fill="#1e40af"/>
        <circle cx="215" cy="35"  r="4" fill="#1e40af"/>
        <circle cx="255" cy="40"  r="4" fill="#1e40af"/>
        <circle cx="295" cy="60"  r="4" fill="#1e40af"/>
        <circle cx="335" cy="90"  r="4" fill="#1e40af"/>
        <circle cx="375" cy="120" r="4" fill="#1e40af"/>
        <circle cx="415" cy="135" r="4" fill="#1e40af"/>
        <!-- Linear fit (bad) -->
        <line x1="40" y1="130" x2="450" y2="75" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3"/>
        <text x="455" y="73" font-size="8" fill="#dc2626">Linear</text>
        <!-- Polynomial fit (good) -->
        <path d="M45,128 Q140,30 255,35 Q370,40 430,130" fill="none" stroke="#065f46" stroke-width="2"/>
        <text x="435" y="130" font-size="8" fill="#065f46">Poly</text>
        <text x="250" y="158" text-anchor="middle" font-size="9" fill="#6b7280">লাল = Linear (ভুল ফিট) · সবুজ = Polynomial (সঠিক ফিট)</text>
      </svg>
    </div>

    <h3>৩. Python দিয়ে Polynomial Regression</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ডেটা তৈরি (curved relationship)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 8, 18, 25, 28, 27, 22, 15, 8, 3])  # inverted U

# Degree 2 (Quadratic)
poly_model = Pipeline([
    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])
poly_model.fit(X, y)
y_pred = poly_model.predict(X)

print(f"R² (degree=2): {r2_score(y, y_pred):.4f}")

# ফিচারগুলো দেখো
pf = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pf.fit_transform(X)
print("Original X:", X[:3].flatten())
print("Poly X:    \n", X_poly[:3])
# [[1, 1], [2, 4], [3, 9]] → [X, X²]</code></pre>

    <h3>৪. সঠিক Degree কীভাবে বেছে নেবে?</h3>
    <pre><code">from sklearn.model_selection import cross_val_score

degrees = [1, 2, 3, 4, 5, 6]
train_scores, cv_scores = [], []

for d in degrees:
    model = Pipeline([
        ('poly',  PolynomialFeatures(degree=d)),
        ('model', LinearRegression())
    ])
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_scores.append(cv.mean())
    model.fit(X, y)
    train_scores.append(model.score(X, y))

# সেরা degree: CV score সবচেয়ে বেশি যেখানে
best_degree = degrees[np.argmax(cv_scores)]
print(f"সেরা degree: {best_degree}")</code></pre>

    <h3>৫. Underfitting vs Overfitting</h3>
    <table>
      <thead><tr><th>Degree</th><th>Train R²</th><th>Test R²</th><th>সমস্যা</th></tr></thead>
      <tbody>
        <tr><td>1 (Linear)</td><td>0.45</td><td>0.42</td><td>Underfitting — সরল রেখা curve ধরতে পারছে না</td></tr>
        <tr><td>2 (Quadratic)</td><td>0.96</td><td>0.94</td><td>✅ সেরা — ভালো ফিট</td></tr>
        <tr><td>3 (Cubic)</td><td>0.97</td><td>0.90</td><td>ঠিক আছে</td></tr>
        <tr><td>6</td><td>0.99</td><td>0.55</td><td>Overfitting — training ডেটা মুখস্থ করে ফেলেছে</td></tr>
        <tr><td>10</td><td>1.00</td><td>-2.3</td><td>মারাত্মক Overfitting</td></tr>
      </tbody>
    </table>

    <h3>৬. Regularization দিয়ে Overfitting কমানো</h3>
    <pre><code">from sklearn.linear_model import Ridge

# Degree বড় কিন্তু Ridge দিয়ে নিয়ন্ত্রণ
poly_ridge = Pipeline([
    ('poly',  PolynomialFeatures(degree=5)),
    ('model', Ridge(alpha=1.0))   # L2 regularization
])
poly_ridge.fit(X_train, y_train)
print(f"Ridge Poly R²: {poly_ridge.score(X_test, y_test):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>কখন ব্যবহার</td><td>Residual plot-এ curve দেখলে</td></tr>
        <tr><td>কাজের পদ্ধতি</td><td>X² বা X³-কে নতুন ফিচার বানিয়ে Linear Regression</td></tr>
        <tr><td>Degree নির্বাচন</td><td>Cross-validation দিয়ে — সাধারণত 2 বা 3 যথেষ্ট</td></tr>
        <tr><td>Overfitting</td><td>Degree বড় হলে ঝুঁকি → Ridge বা Lasso ব্যবহার করো</td></tr>
        <tr><td>sklearn</td><td>PolynomialFeatures + LinearRegression Pipeline</td></tr>
      </tbody>
    </table>
  `,
};
