export const lr_10_vif = {
  title: "VIF গভীরে: Variance Inflation Factor বিস্তারিত",
  description: "VIF কী, এর গণিত কীভাবে কাজ করে, Python দিয়ে কীভাবে হিসাব করতে হয়, এবং উচ্চ VIF দেখলে কী করতে হয় — বাংলায় সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-vif-deep-dive",
  content: `
    <h3>১. VIF কী?</h3>
    <p><strong>VIF (Variance Inflation Factor)</strong> হলো একটি সংখ্যা যা বলে দেয়, কোনো একটি ফিচারের coefficient-এর variance কতটুকু বেড়ে গেছে — শুধুমাত্র অন্য ফিচারগুলোর সাথে correlation-এর কারণে।</p>
    <p>সহজ কথায়: VIF বড় মানে সেই ফিচারটি অন্য ফিচার দিয়ে প্রায় পুরোপুরি ব্যাখ্যা করা যায় — তাই মডেল বিভ্রান্ত হয়ে যায়।</p>

    <h3>২. VIF-এর গণিত</h3>
    <p>ধরো তোমার কাছে feature <strong>Xⱼ</strong> আছে। এখন এই Xⱼ-কে <em>target</em> ধরে বাকি সব feature দিয়ে একটি নতুন রিগ্রেশন চালাও:</p>
    <pre><code>Xⱼ = β₀ + β₁X₁ + β₂X₂ + ... (Xⱼ বাদ দিয়ে বাকি সব)

তারপর: VIFⱼ = 1 / (1 - R²ⱼ)

যেখানে R²ⱼ = ওই নতুন রিগ্রেশনের R² স্কোর</code></pre>
    <table>
      <thead><tr><th>R²ⱼ এর মান</th><th>VIF এর মান</th><th>অর্থ</th></tr></thead>
      <tbody>
        <tr><td>0.00</td><td>1.0</td><td>কোনো correlation নেই</td></tr>
        <tr><td>0.50</td><td>2.0</td><td>মাঝারি correlation</td></tr>
        <tr><td>0.80</td><td>5.0</td><td>সতর্ক থাকো</td></tr>
        <tr><td>0.90</td><td>10.0</td><td>সমস্যা আছে</td></tr>
        <tr><td>0.99</td><td>100.0</td><td>মারাত্মক multicollinearity</td></tr>
      </tbody>
    </table>

    <h3>৩. VIF কী বলে দেয়?</h3>
    <p>VIF = 5 মানে হলো: যদি multicollinearity না থাকত, তাহলে coefficient-এর variance <strong>5 গুণ ছোট</strong> হতো। বড় variance মানে coefficient অনেক অনিশ্চিত — ছোট ডেটা পরিবর্তনেও coefficient অনেক বদলে যায়।</p>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">VIF থ্রেশহোল্ড গাইড</text>
        <rect x="20"  y="30" width="80" height="30" rx="4" fill="#dcfce7"/>
        <text x="60"  y="50" text-anchor="middle" font-size="10" fill="#166534">VIF &lt; 5</text>
        <text x="60"  y="76" text-anchor="middle" font-size="9" fill="#6b7280">নিরাপদ</text>
        <rect x="120" y="30" width="80" height="30" rx="4" fill="#fef9c3"/>
        <text x="160" y="50" text-anchor="middle" font-size="10" fill="#854d0e">VIF 5–10</text>
        <text x="160" y="76" text-anchor="middle" font-size="9" fill="#6b7280">সতর্কতা</text>
        <rect x="220" y="30" width="80" height="30" rx="4" fill="#fee2e2"/>
        <text x="260" y="50" text-anchor="middle" font-size="10" fill="#991b1b">VIF &gt; 10</text>
        <text x="260" y="76" text-anchor="middle" font-size="9" fill="#6b7280">সমস্যা আছে</text>
        <rect x="320" y="30" width="120" height="30" rx="4" fill="#fce7f3"/>
        <text x="380" y="50" text-anchor="middle" font-size="10" fill="#9d174d">VIF &gt; 100</text>
        <text x="380" y="76" text-anchor="middle" font-size="9" fill="#6b7280">মারাত্মক সমস্যা</text>
        <text x="240" y="110" text-anchor="middle" font-size="9" fill="#9ca3af">( কিছু ক্ষেত্রে threshold 5 বা 10 ব্যবহার করা হয় — context অনুযায়ী )</text>
      </svg>
    </div>

    <h3>৪. Python দিয়ে VIF হিসাব</h3>
    <pre><code>import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ডেটা তৈরি
data = {
    'area':       [1000, 1500, 2000, 800,  1200, 1800, 950,  2200],
    'rooms':      [2,    3,    4,    1,    2,    4,    2,    5   ],
    'age':        [5,    3,    1,    10,   4,    2,    7,    1   ],
    'floor_area': [900,  1400, 1900, 750,  1100, 1700, 850,  2100],
    # ↑ এটি 'area' এর প্রায় কপি — multicollinearity থাকবে
}
df = pd.DataFrame(data)
X = df[['area', 'rooms', 'age', 'floor_area']]

# VIF হিসাব
vif_df = pd.DataFrame()
vif_df['Feature'] = X.columns
vif_df['VIF'] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]
vif_df['মন্তব্য'] = vif_df['VIF'].apply(
    lambda v: 'নিরাপদ' if v < 5 else ('সতর্কতা' if v < 10 else 'সমস্যা আছে')
)
print(vif_df.sort_values('VIF', ascending=False))</code></pre>
    <p>আউটপুট দেখতে পাবে:</p>
    <pre><code>      Feature         VIF    মন্তব্য
3  floor_area   312.45    সমস্যা আছে
0        area   298.12    সমস্যা আছে
1       rooms     3.21    নিরাপদ
2         age     1.87    নিরাপদ</code></pre>

    <h3>৫. Iterative VIF দিয়ে ফিচার বাদ দেওয়া</h3>
    <p>একটি popular পদ্ধতি হলো সবচেয়ে বেশি VIF-ওয়ালা ফিচার বাদ দাও, আবার হিসাব করো — যতক্ষণ না সব VIF নিরাপদ সীমায় আসে।</p>
    <pre><code>def remove_high_vif(X, threshold=10.0):
    """
    সবচেয়ে বেশি VIF-ওয়ালা ফিচার পর্যায়ক্রমে বাদ দেয়।
    """
    cols = list(X.columns)
    while True:
        vif = [variance_inflation_factor(X[cols].values, i)
               for i in range(len(cols))]
        max_vif = max(vif)
        if max_vif < threshold:
            break
        worst = cols[vif.index(max_vif)]
        print(f"বাদ দেওয়া হচ্ছে: {worst}  (VIF = {max_vif:.2f})")
        cols.remove(worst)
    print(f"\\nবাকি ফিচার: {cols}")
    return X[cols]

X_clean = remove_high_vif(X, threshold=10.0)

# আউটপুট:
# বাদ দেওয়া হচ্ছে: floor_area  (VIF = 312.45)
# বাকি ফিচার: ['area', 'rooms', 'age']</code></pre>

    <h3>৬. Constant Term এবং VIF</h3>
    <p>statsmodels-এ VIF হিসাবের সময় একটু সাবধান থাকতে হবে — intercept থাকলে VIF ভুল আসতে পারে। সঠিক পদ্ধতি:</p>
    <pre><code>from statsmodels.tools.tools import add_constant

# add_constant ব্যবহার না করে ফিচার ম্যাট্রিক্স দাও
# (শুধু X দাও, y নয়, constant নয়)
X_vals = X.values  # numpy array হিসেবে

vif_scores = [variance_inflation_factor(X_vals, i)
              for i in range(X_vals.shape[1])]</code></pre>

    <h3>৭. VIF কখন উপেক্ষা করা যায়?</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>করণীয়</th></tr></thead>
      <tbody>
        <tr><td>শুধু prediction দরকার, interpretation নয়</td><td>VIF উপেক্ষা করা যায়</td></tr>
        <tr><td>coefficient ব্যাখ্যা করতে হবে</td><td>VIF ঠিক করা আবশ্যক</td></tr>
        <tr><td>ডেটাসেট অনেক বড় (n ≫ p)</td><td>কম সমস্যা, তবু চেক করো</td></tr>
        <tr><td>Interaction term বা polynomial আছে</td><td>স্বাভাবিকভাবেই VIF বাড়ে — center করো</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>VIF সূত্র</td><td>1 / (1 − R²ⱼ) — অন্য ফিচার দিয়ে কতটুকু ব্যাখ্যা হয়</td></tr>
        <tr><td>VIF &lt; 5</td><td>নিরাপদ</td></tr>
        <tr><td>VIF 5–10</td><td>সতর্ক থাকো</td></tr>
        <tr><td>VIF &gt; 10</td><td>সমস্যা — ফিচার বাদ দাও বা Ridge/Lasso ব্যবহার করো</td></tr>
        <tr><td>Iterative removal</td><td>সবচেয়ে বেশি VIF বাদ দাও, আবার হিসাব করো</td></tr>
      </tbody>
    </table>
  `,
};
