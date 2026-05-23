export const logit_1_math = {
  title: "লজিস্টিক রিগ্রেশনের গণিত গভীরে — Log-Odds, MLE ও Decision Boundary",
  description: "Odds, Log-Odds (Logit), Maximum Likelihood Estimation — লজিস্টিক রিগ্রেশন কীভাবে সত্যিই কাজ করে তা গণিত দিয়ে বোঝো।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 12,
  slug: "logit-math-deep-dive",
  content: `
    <h3>১. Probability থেকে Odds</h3>
    <p>ধরো একটি ক্রিকেট ম্যাচে বাংলাদেশের জেতার probability = 0.75।</p>
    <pre><code>P(জয়) = 0.75
P(পরাজয়) = 1 − 0.75 = 0.25

Odds = P(জয়) / P(পরাজয়) = 0.75 / 0.25 = 3

অর্থ: প্রতি একবার হারলে ৩ বার জেতে।

# সাধারণ সূত্র:
Odds = p / (1 − p)

# Odds থেকে Probability:
p = Odds / (1 + Odds)</code></pre>
    <table>
      <thead><tr><th>Probability (p)</th><th>Odds</th><th>অর্থ</th></tr></thead>
      <tbody>
        <tr><td>0.50</td><td>1.00</td><td>সমান সম্ভাবনা</td></tr>
        <tr><td>0.75</td><td>3.00</td><td>৩:১ অনুকূলে</td></tr>
        <tr><td>0.90</td><td>9.00</td><td>৯:১ অনুকূলে</td></tr>
        <tr><td>0.10</td><td>0.11</td><td>প্রতিকূলে</td></tr>
      </tbody>
    </table>

    <h3>২. Log-Odds (Logit)</h3>
    <p>Odds-এর logarithm নিলে পাই <strong>Log-Odds</strong> বা <strong>Logit</strong>। এটি −∞ থেকে +∞ পর্যন্ত হতে পারে — একটি linear regression-এর output-এর মতো।</p>
    <pre><code>Log-Odds = log(p / (1−p))     ← natural log (ln)

p = 0.50 → Log-Odds = log(1)   = 0.00
p = 0.75 → Log-Odds = log(3)   = 1.10
p = 0.90 → Log-Odds = log(9)   = 2.20
p = 0.10 → Log-Odds = log(0.11) = −2.20

import numpy as np
p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
log_odds = np.log(p / (1 - p))
print(list(zip(p, log_odds.round(2))))
# [(0.1, -2.2), (0.25, -1.1), (0.5, 0.0), (0.75, 1.1), (0.9, 2.2)]</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Logit রূপান্তর: Probability → Log-Odds → Probability</text>
        <rect x="20"  y="35" width="110" height="40" rx="6" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="75"  y="52" text-anchor="middle" font-size="9" fill="#1e40af">Probability</text>
        <text x="75"  y="67" text-anchor="middle" font-size="9" fill="#3b82f6">p ∈ [0, 1]</text>
        <text x="148" y="58" text-anchor="middle" font-size="18" fill="#94a3b8">→</text>
        <rect x="170" y="35" width="140" height="40" rx="6" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="240" y="52" text-anchor="middle" font-size="9" fill="#92400e">Log-Odds (Logit)</text>
        <text x="240" y="67" text-anchor="middle" font-size="9" fill="#d97706">log(p/1−p) ∈ (−∞,+∞)</text>
        <text x="328" y="58" text-anchor="middle" font-size="18" fill="#94a3b8">→</text>
        <rect x="350" y="35" width="110" height="40" rx="6" fill="#dcfce7" stroke="#86efac"/>
        <text x="405" y="52" text-anchor="middle" font-size="9" fill="#166534">Sigmoid</text>
        <text x="405" y="67" text-anchor="middle" font-size="9" fill="#16a34a">ফিরে আসে [0,1]</text>
        <text x="240" y="105" text-anchor="middle" font-size="9" fill="#6b7280">Linear regression → Logit space-এ কাজ করে → Sigmoid দিয়ে probability-তে রূপান্তর</text>
      </svg>
    </div>

    <h3>৩. লজিস্টিক রিগ্রেশনের সম্পূর্ণ সূত্র</h3>
    <pre><code># মডেল বলে:
Log-Odds = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
         = z  (linear combination)

# তারপর Sigmoid দিয়ে probability:
p = σ(z) = 1 / (1 + e⁻ᶻ)

# অর্থাৎ:
log(p / 1−p) = β₀ + β₁x₁ + β₂x₂ + ...

# β₁ = 1 মানে: x₁ একটি একক বাড়লে Log-Odds 1 বাড়ে
# Odds Ratio = e^β₁

import numpy as np
beta_1 = 0.5
odds_ratio = np.exp(beta_1)
print(f"β₁ = {beta_1} → Odds Ratio = {odds_ratio:.3f}")
# x₁ একক বাড়লে Odds ≈ 1.65 গুণ বাড়ে</code></pre>

    <h3>৪. Maximum Likelihood Estimation (MLE)</h3>
    <p>লজিস্টিক রিগ্রেশন Least Squares (OLS) নয় — <strong>MLE</strong> ব্যবহার করে coefficient খোঁজে। MLE বলে: এমন coefficient বেছে নাও যেগুলো দিয়ে observed ডেটা দেখার সম্ভাবনা সবচেয়ে বেশি।</p>
    <pre><code># Likelihood:
L(β) = Π [p̂ᵢ^yᵢ × (1−p̂ᵢ)^(1−yᵢ)]

# Log-Likelihood (গুণ → যোগ, সহজ হিসাব):
ℓ(β) = Σ [yᵢ × log(p̂ᵢ) + (1−yᵢ) × log(1−p̂ᵢ)]

# Minimize করি: − ℓ(β)  →  এটাই Log Loss!
# অর্থাৎ Log Loss কমানো = MLE সমাধান করা

# Python দিয়ে নিজে দেখো:
from scipy.special import expit  # sigmoid

def log_likelihood(beta, X, y):
    z = X @ beta
    p = expit(z)
    return np.sum(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10))

# sklearn এটাই ভেতরে ভেতরে optimize করে (gradient-based)</code></pre>

    <h3>৫. Decision Boundary</h3>
    <p>মডেল যেখানে p = 0.5 সেখানেই decision boundary — অর্থাৎ z = 0 (Log-Odds = 0)।</p>
    <pre><code">from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# ২টি feature দিয়ে সহজ উদাহরণ
np.random.seed(42)
X0 = np.random.randn(50, 2) + [-2, -2]
X1 = np.random.randn(50, 2) + [2, 2]
X = np.vstack([X0, X1])
y = np.array([0]*50 + [1]*50)

model = LogisticRegression()
model.fit(X, y)

# Decision boundary: β₀ + β₁x₁ + β₂x₂ = 0
# x₂ = -(β₀ + β₁x₁) / β₂
b, (w1, w2) = model.intercept_[0], model.coef_[0]
x1_vals = np.linspace(-5, 5, 100)
x2_vals = -(b + w1 * x1_vals) / w2

plt.scatter(X0[:,0], X0[:,1], c='blue', label='Class 0')
plt.scatter(X1[:,0], X1[:,1], c='red', label='Class 1')
plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()</code></pre>

    <h3>৬. Odds Ratio — coefficient ব্যাখ্যা</h3>
    <pre><code">from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# উদাহরণ: ডায়াবেটিস পূর্বাভাস
# features: বয়স, BMI, glucose
np.random.seed(42)
n = 200
age     = np.random.randint(20, 70, n)
bmi     = np.random.uniform(18, 40, n)
glucose = np.random.uniform(70, 200, n)
y = (0.02*age + 0.05*bmi + 0.01*glucose - 3 + np.random.randn(n)*0.5 > 0).astype(int)

X = np.column_stack([age, bmi, glucose])
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

result = pd.DataFrame({
    'Feature':     ['বয়স', 'BMI', 'Glucose'],
    'Coefficient': model.coef_[0],
    'Odds Ratio':  np.exp(model.coef_[0]),
})
print(result.round(3))
# Odds Ratio > 1 → ফিচার বাড়লে diabetes সম্ভাবনা বাড়ে
# Odds Ratio < 1 → ফিচার বাড়লে diabetes সম্ভাবনা কমে</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>ধারণা</th><th>সূত্র</th><th>অর্থ</th></tr></thead>
      <tbody>
        <tr><td>Odds</td><td>p / (1−p)</td><td>জয়-পরাজয়ের অনুপাত</td></tr>
        <tr><td>Log-Odds (Logit)</td><td>log(p / 1−p)</td><td>−∞ থেকে +∞, linear মডেলের output</td></tr>
        <tr><td>Sigmoid</td><td>1 / (1 + e⁻ᶻ)</td><td>Log-Odds → probability রূপান্তর</td></tr>
        <tr><td>MLE</td><td>Log-Likelihood maximize</td><td>coefficient খোঁজার পদ্ধতি</td></tr>
        <tr><td>Odds Ratio</td><td>e^β</td><td>feature এক একক বাড়লে Odds কতগুণ বাড়ে</td></tr>
        <tr><td>Decision Boundary</td><td>z = 0 (p = 0.5)</td><td>মডেলের বিভাজন রেখা</td></tr>
      </tbody>
    </table>
  `,
};
