export const lr_4_multicollinearity_ridge_lasso = {
  title: "মাল্টিকোলিনিয়ারিটি এবং Ridge ও Lasso দিয়ে সমাধান",
  description: "মাল্টিকোলিনিয়ারিটি কী, কীভাবে চেনা যায়, এবং Ridge ও Lasso Regularization দিয়ে কীভাবে এটি সমাধান করা যায় — বাংলায় সহজ ব্যাখ্যা।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 11,
  slug: "lr-multicollinearity-ridge-lasso",
  content: `
    <h3>১. মাল্টিকোলিনিয়ারিটি কী?</h3>
    <p>মাল্টিকোলিনিয়ারিটি ঘটে যখন দুটি বা তার বেশি <strong>independent variable (ফিচার)</strong> পরস্পরের সাথে অনেক বেশি correlated থাকে।</p>
    <p>উদাহরণ: একটি বাড়ির দাম ভবিষ্যদ্বাণী করতে তুমি ব্যবহার করছো:</p>
    <ul>
      <li>আয়তন (বর্গফুট)</li>
      <li>কক্ষ সংখ্যা</li>
      <li>প্রতি কক্ষের গড় আয়তন ← এটি প্রথম দুটি থেকে তৈরি!</li>
    </ul>
    <p>এখন মডেল বুঝতে পারে না কোন ফিচারের কতটুকু গুরুত্ব। Coefficients অস্থির ও অবিশ্বস্ত হয়ে যায়।</p>

    <h3>২. মাল্টিকোলিনিয়ারিটির সমস্যা</h3>
    <table>
      <thead><tr><th>সমস্যা</th><th>কী হয়</th></tr></thead>
      <tbody>
        <tr><td>Coefficient অস্থির</td><td>ডেটা একটু বদলালে coefficient অনেক বদলে যায়</td></tr>
        <tr><td>ভুল ব্যাখ্যা</td><td>কোন ফিচার important তা বোঝা যায় না</td></tr>
        <tr><td>বড় Standard Error</td><td>coefficient-এর uncertainty বাড়ে</td></tr>
        <tr><td>R² ঠিক, কিন্তু coefficient ভুল</td><td>মডেল মোটামুটি কাজ করে কিন্তু interpretation ভুল</td></tr>
      </tbody>
    </table>

    <h3>৩. কীভাবে চেনা যায়? — VIF</h3>
    <p><strong>VIF (Variance Inflation Factor)</strong> দিয়ে মাল্টিকোলিনিয়ারিটি মাপা হয়।</p>
    <pre><code>from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

data = {
    'area':       [1000, 1500, 2000, 800,  1200],
    'rooms':      [2,    3,    4,    1,    2   ],
    'age':        [5,    3,    1,    10,   4   ],
    'price_lakh': [50,   75,   120,  35,   60  ]
}
df = pd.DataFrame(data)
X = df[['area', 'rooms', 'age']]

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF']   = [variance_inflation_factor(X.values, i)
                     for i in range(len(X.columns))]
print(vif_data)

# VIF ব্যাখ্যা:
# VIF < 5  → ঠিক আছে
# VIF 5-10 → সতর্ক থাকো
# VIF > 10 → মাল্টিকোলিনিয়ারিটি আছে!</code></pre>

    <h3>৪. Correlation Matrix দিয়ে চেক</h3>
    <pre><code>import seaborn as sns
import matplotlib.pyplot as plt

corr = X.corr()
print(corr)

# |correlation| > 0.8 মানে সমস্যা আছে
# গরম মানচিত্র তৈরি:
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()</code></pre>

    <h3>৫. সমাধান ১ — Ridge Regression (L2)</h3>
    <p>Ridge সব coefficient-কে ছোট রাখে কিন্তু কোনোটাকে শূন্য করে না।</p>
    <pre><code># Ridge: সাধারণ MSE-র সাথে penalty যোগ করে
Loss = MSE + α × Σ(mᵢ²)
#              ↑
#         L2 Regularization (coefficient-এর বর্গের যোগফল)

# α (alpha) বড় → coefficient ছোট হয় → simpler model
# α = 0 → সাধারণ Linear Regression

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)   # alpha = regularization strength
ridge.fit(X_train, y_train)

print("Ridge Coefficients:", ridge.coef_)
print("Ridge R²:", ridge.score(X_test, y_test))</code></pre>

    <h3>৬. সমাধান ২ — Lasso Regression (L1)</h3>
    <p>Lasso কিছু coefficient-কে ঠিক শূন্য করে দেয় — অর্থাৎ সে নিজেই feature selection করে।</p>
    <pre><code># Lasso: MSE + α × Σ|mᵢ|
Loss = MSE + α × Σ|mᵢ|
#              ↑
#         L1 Regularization (coefficient-এর নিরঙ্কুশ মানের যোগফল)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print("Lasso Coefficients:", lasso.coef_)
# কিছু coefficient = 0.0 → সেই ফিচার বাদ দেওয়া হয়েছে
print("বাদ পড়া ফিচার:", [f for f, c in zip(X.columns, lasso.coef_) if c == 0])</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 500 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="250" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Linear vs Ridge vs Lasso Coefficients</text>
        <!-- Linear -->
        <rect x="30"  y="30" width="30" height="60" rx="2" fill="#1e40af"/>
        <rect x="65"  y="45" width="30" height="45" rx="2" fill="#1e40af"/>
        <rect x="100" y="55" width="30" height="35" rx="2" fill="#1e40af"/>
        <text x="80"  y="105" text-anchor="middle" font-size="9" fill="#6b7280">Linear</text>
        <!-- Ridge -->
        <rect x="185" y="42" width="30" height="48" rx="2" fill="#7c3aed"/>
        <rect x="220" y="55" width="30" height="35" rx="2" fill="#7c3aed"/>
        <rect x="255" y="62" width="30" height="28" rx="2" fill="#7c3aed"/>
        <text x="230" y="105" text-anchor="middle" font-size="9" fill="#6b7280">Ridge (সব ছোট)</text>
        <!-- Lasso -->
        <rect x="335" y="40" width="30" height="50" rx="2" fill="#065f46"/>
        <rect x="370" y="85" width="30" height="5" rx="2" fill="#065f46" opacity="0.3"/>
        <rect x="405" y="55" width="30" height="35" rx="2" fill="#065f46"/>
        <text x="375" y="105" text-anchor="middle" font-size="9" fill="#6b7280">Lasso (কিছু শূন্য)</text>
        <text x="385" y="82" text-anchor="middle" font-size="8" fill="#dc2626">≈ 0</text>
      </svg>
    </div>

    <h3>৭. সেরা Alpha খোঁজা</h3>
    <pre><code>from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np

alphas = np.logspace(-3, 3, 100)  # 0.001 থেকে 1000

# RidgeCV — cross-validation দিয়ে সেরা alpha খোঁজে
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"সেরা Ridge alpha: {ridge_cv.alpha_:.4f}")

# LassoCV
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"সেরা Lasso alpha: {lasso_cv.alpha_:.4f}")</code></pre>

    <h3>৮. Ridge vs Lasso তুলনা</h3>
    <table>
      <thead><tr><th></th><th>Ridge (L2)</th><th>Lasso (L1)</th></tr></thead>
      <tbody>
        <tr><td><strong>Penalty</strong></td><td>α × Σmᵢ²</td><td>α × Σ|mᵢ|</td></tr>
        <tr><td><strong>Coefficient শূন্য?</strong></td><td>না (শুধু ছোট করে)</td><td>হ্যাঁ (feature selection করে)</td></tr>
        <tr><td><strong>Feature selection</strong></td><td>না</td><td>হ্যাঁ (automatic)</td></tr>
        <tr><td><strong>Correlated features</strong></td><td>সবাইকে রাখে, weight ভাগ করে</td><td>একটি রাখে, বাকি বাদ দেয়</td></tr>
        <tr><td><strong>কখন ব্যবহার</strong></td><td>সব ফিচার important হলে</td><td>অনেক ফিচার, কিছু অপ্রয়োজনীয়</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>মাল্টিকোলিনিয়ারিটি</td><td>ফিচারগুলো পরস্পর correlated → coefficient অস্থির</td></tr>
        <tr><td>VIF > 10</td><td>সমস্যা আছে — ফিচার সরাও বা regularization ব্যবহার করো</td></tr>
        <tr><td>Ridge</td><td>L2 penalty — coefficient ছোট করে, শূন্য করে না</td></tr>
        <tr><td>Lasso</td><td>L1 penalty — কিছু coefficient শূন্য করে, feature selection করে</td></tr>
        <tr><td>Alpha</td><td>বড় alpha = বেশি regularization, ছোট alpha = কম</td></tr>
      </tbody>
    </table>
  `,
};
