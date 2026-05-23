export const lr_6_residual_analysis = {
  title: "Residual Analysis ও Assumption চেক — মডেল সঠিক কিনা বুঝবো কীভাবে?",
  description: "লিনিয়ার রিগ্রেশনের ৫টি assumption কীভাবে plot দিয়ে চেক করবে — Residual Plot, Q-Q Plot, Scale-Location, এবং VIF — বাংলায় সহজ ব্যাখ্যা।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-residual-analysis",
  content: `
    <h3>১. Residual কী?</h3>
    <p>Residual হলো প্রকৃত মান এবং মডেলের ভবিষ্যদ্বাণীর মধ্যে পার্থক্য:</p>
    <pre><code>Residual (eᵢ) = Yᵢ (প্রকৃত) - Ŷᵢ (ভবিষ্যদ্বাণী)

# উদাহরণ:
# প্রকৃত নম্বর: 75
# মডেলের ভবিষ্যদ্বাণী: 71
# Residual = 75 - 71 = +4  (মডেল ৪ কম বলেছে)</code></pre>
    <p>লিনিয়ার রিগ্রেশন ভালো কাজ করতে হলে residual-গুলো কিছু নিয়ম মেনে চলতে হবে। এই নিয়মগুলো plot দিয়ে চেক করা হয়।</p>

    <h3>২. Assumption ১ — Linearity চেক</h3>
    <p><strong>কী দেখবে:</strong> Fitted values (Ŷ) vs Residuals plot-এ residual-গুলো কোনো pattern ছাড়া এলোমেলো থাকা উচিত।</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_train)
residuals = y_train - y_pred

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Fitted Values (Ŷ)')
plt.ylabel('Residuals')
plt.title('Residual Plot — Linearity চেক')
plt.show()

# ✅ ভালো: residual এলোমেলো, কোনো curve নেই
# ❌ খারাপ: U-shape বা curve দেখলে → non-linearity আছে</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 500 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="250" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Residual Plot: ভালো vs খারাপ</text>
        <!-- Good plot -->
        <text x="120" y="34" text-anchor="middle" font-size="9" fill="#6b7280">✅ ভালো (Random)</text>
        <line x1="30" y1="80" x2="210" y2="80" stroke="#e5e7eb" stroke-width="1"/>
        <line x1="30" y1="45" x2="30" y2="115" stroke="#e5e7eb" stroke-width="1"/>
        <circle cx="55"  cy="65"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="75"  cy="92"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="95"  cy="70"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="115" cy="88"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="135" cy="62"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="155" cy="95"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="175" cy="72"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <circle cx="195" cy="85"  r="3.5" fill="#1e40af" opacity="0.7"/>
        <line x1="35" y1="80" x2="205" y2="80" stroke="#dc2626" stroke-width="1" stroke-dasharray="4,2"/>
        <!-- Bad plot -->
        <text x="370" y="34" text-anchor="middle" font-size="9" fill="#6b7280">❌ খারাপ (U-shape)</text>
        <line x1="270" y1="80" x2="450" y2="80" stroke="#e5e7eb" stroke-width="1"/>
        <line x1="270" y1="45" x2="270" y2="115" stroke="#e5e7eb" stroke-width="1"/>
        <circle cx="290" cy="55"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="315" cy="75"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="340" cy="88"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="365" cy="90"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="390" cy="82"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="415" cy="68"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <circle cx="435" cy="52"  r="3.5" fill="#dc2626" opacity="0.7"/>
        <path d="M290,55 Q362,100 435,52" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
        <line x1="275" y1="80" x2="445" y2="80" stroke="#dc2626" stroke-width="1" stroke-dasharray="4,2"/>
        <text x="250" y="125" text-anchor="middle" font-size="9" fill="#6b7280">বাম: linear সম্পর্ক ঠিক আছে · ডান: non-linear সম্পর্ক আছে</text>
      </svg>
    </div>

    <h3>৩. Assumption ২ — Homoscedasticity চেক</h3>
    <p>Residual-এর variance সব জায়গায় সমান হওয়া উচিত। ফানেল আকার দেখলে সমস্যা আছে।</p>
    <pre><code># Scale-Location Plot
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6, color='steelblue')
plt.xlabel('Fitted Values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location Plot')
plt.show()

# ✅ ভালো: বিন্দুগুলো সমান ছড়ানো (horizontal band)
# ❌ খারাপ: ডানে গেলে বিন্দু বেশি ছড়ায় (funnel shape)
#           → Heteroscedasticity → log(Y) transform করো</code></pre>

    <h3>৪. Assumption ৩ — Normality of Residuals চেক</h3>
    <p>Residual-গুলো normal distribution অনুসরণ করছে কিনা Q-Q plot দিয়ে দেখা হয়।</p>
    <pre><code">import scipy.stats as stats

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot — Residuals Normality চেক')
plt.show()

# ✅ ভালো: বিন্দুগুলো সরল রেখার উপরে পড়ে
# ❌ খারাপ: বিন্দু রেখা থেকে অনেক সরে গেলে → normality নেই

# Shapiro-Wilk test (সংখ্যায়):
stat, p = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p:.4f}")
# p > 0.05 → normality আছে ✅
# p < 0.05 → normality নেই ❌</code></pre>

    <h3>৫. Assumption ৪ — Independence চেক</h3>
    <pre><code">from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.3f}")

# মান ব্যাখ্যা:
# ~2.0  → কোনো autocorrelation নেই ✅
# < 1.5 → positive autocorrelation আছে ❌
# > 2.5 → negative autocorrelation আছে ❌
# (time series ডেটায় এটা বেশি গুরুত্বপূর্ণ)</code></pre>

    <h3>৬. Assumption ৫ — No Multicollinearity চেক (VIF)</h3>
    <pre><code">from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame({
    'feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print(vif_df)

# VIF < 5  → ঠিক আছে ✅
# VIF 5-10 → সতর্ক থাকো ⚠️
# VIF > 10 → মাল্টিকোলিনিয়ারিটি আছে ❌</code></pre>

    <h3>৭. সব একসাথে — 4-in-1 Diagnostic Plot</h3>
    <pre><code">fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Residual vs Fitted
axes[0,0].scatter(y_pred, residuals, alpha=0.5)
axes[0,0].axhline(0, color='red', linestyle='--')
axes[0,0].set_title('Residuals vs Fitted')

# 2. Q-Q Plot
stats.probplot(residuals, plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot')

# 3. Scale-Location
axes[1,0].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
axes[1,0].set_title('Scale-Location')

# 4. Residuals Histogram
axes[1,1].hist(residuals, bins=20, color='steelblue', edgecolor='white')
axes[1,1].set_title('Residuals Distribution')

plt.tight_layout()
plt.show()</code></pre>

    <h3>সমস্যা এবং সমাধান</h3>
    <table>
      <thead><tr><th>সমস্যা</th><th>লক্ষণ</th><th>সমাধান</th></tr></thead>
      <tbody>
        <tr><td>Non-linearity</td><td>Residual plot-এ curve</td><td>Polynomial feature, log transform</td></tr>
        <tr><td>Heteroscedasticity</td><td>Funnel shape</td><td>log(Y), WLS regression</td></tr>
        <tr><td>Non-normality</td><td>Q-Q plot বেঁকে যায়</td><td>Outlier সরাও, log transform</td></tr>
        <tr><td>Autocorrelation</td><td>DW &lt; 1.5</td><td>Lag feature যোগ করো</td></tr>
        <tr><td>Multicollinearity</td><td>VIF &gt; 10</td><td>Ridge/Lasso, ফিচার সরাও</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>Plot/Test</th><th>কী চেক করে</th><th>ভালো লক্ষণ</th></tr></thead>
      <tbody>
        <tr><td>Residual vs Fitted</td><td>Linearity</td><td>এলোমেলো, কোনো pattern নেই</td></tr>
        <tr><td>Scale-Location</td><td>Homoscedasticity</td><td>সমান ছড়ানো বিন্দু</td></tr>
        <tr><td>Q-Q Plot</td><td>Normality</td><td>বিন্দু সরল রেখায়</td></tr>
        <tr><td>Durbin-Watson</td><td>Independence</td><td>~2.0</td></tr>
        <tr><td>VIF</td><td>No Multicollinearity</td><td>&lt; 5</td></tr>
      </tbody>
    </table>
  `,
};
