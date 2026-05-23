export const lr_12_feature_scaling = {
  title: "Feature Scaling: কেন দরকার এবং কীভাবে করতে হয়",
  description: "StandardScaler, MinMaxScaler, RobustScaler — কোনটা কী করে, কখন ব্যবহার করতে হয়, এবং লিনিয়ার রিগ্রেশনে scaling না করলে কী সমস্যা হয়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-feature-scaling",
  content: `
    <h3>১. Feature Scaling কেন দরকার?</h3>
    <p>ধরো তোমার ডেটায় দুটি ফিচার আছে:</p>
    <ul>
      <li><strong>বয়স:</strong> ২০ থেকে ৬০ (range = ৪০)</li>
      <li><strong>বেতন:</strong> ১৫,০০০ থেকে ৫,০০,০০০ (range = ৪৮৫,০০০)</li>
    </ul>
    <p>এখন মডেল যদি Gradient Descent ব্যবহার করে, তাহলে বেতনের coefficient খুঁজতে অনেক ছোট ছোট পদক্ষেপ নিতে হয় — কিন্তু বয়সের coefficient খুঁজতে বড় পদক্ষেপ। এই অসমতার কারণে:</p>
    <ul>
      <li>Gradient Descent ধীরে converge করে</li>
      <li>Learning rate ঠিক করা কঠিন হয়</li>
      <li>Regularization (Ridge/Lasso) সমানভাবে কাজ করে না</li>
    </ul>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Scaling ছাড়া বনাম Scaling সহ Gradient Descent</text>
        <!-- Without scaling: elongated ellipse -->
        <ellipse cx="120" cy="75" rx="80" ry="30" fill="none" stroke="#e5e7eb" stroke-width="1.5"/>
        <ellipse cx="120" cy="75" rx="55" ry="20" fill="none" stroke="#d1d5db" stroke-width="1.5"/>
        <ellipse cx="120" cy="75" rx="30" ry="10" fill="none" stroke="#9ca3af" stroke-width="1.5"/>
        <circle cx="120" cy="75" r="3" fill="#111827"/>
        <!-- zigzag path -->
        <polyline points="30,105 75,90 65,80 100,78 95,76 115,75" stroke="#dc2626" stroke-width="1.5" fill="none" stroke-dasharray="3,2"/>
        <text x="120" y="118" text-anchor="middle" font-size="9" fill="#dc2626">Scaling ছাড়া (zigzag)</text>
        <!-- With scaling: circular -->
        <ellipse cx="360" cy="75" rx="50" ry="45" fill="none" stroke="#e5e7eb" stroke-width="1.5"/>
        <ellipse cx="360" cy="75" rx="32" ry="29" fill="none" stroke="#d1d5db" stroke-width="1.5"/>
        <ellipse cx="360" cy="75" rx="15" ry="13" fill="none" stroke="#9ca3af" stroke-width="1.5"/>
        <circle cx="360" cy="75" r="3" fill="#111827"/>
        <polyline points="310,115 330,100 342,89 352,82 358,77 360,75" stroke="#16a34a" stroke-width="1.5" fill="none" stroke-dasharray="3,2"/>
        <text x="360" y="118" text-anchor="middle" font-size="9" fill="#16a34a">Scaling সহ (সরাসরি)</text>
      </svg>
    </div>

    <h3>২. StandardScaler (Z-score Normalization)</h3>
    <p>প্রতিটি মান থেকে mean বাদ দিয়ে standard deviation দিয়ে ভাগ করা হয়।</p>
    <pre><code># সূত্র: z = (x − μ) / σ
# ফলাফল: mean = 0, std = 1

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = {'বয়স': [25, 30, 35, 40, 45],
        'বেতন': [20000, 35000, 50000, 65000, 80000]}
df = pd.DataFrame(data)

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_scaled)
#        বয়স     বেতন
# 0  -1.414  -1.414
# 1  -0.707  -0.707
# 2   0.000   0.000
# 3   0.707   0.707
# 4   1.414   1.414

print("Mean:", df_scaled.mean().round(10))   # ≈ 0
print("Std:",  df_scaled.std())              # ≈ 1</code></pre>
    <p><strong>কখন ব্যবহার করবে:</strong> যখন ডেটা প্রায় normally distributed এবং outlier কম। Linear Regression, Logistic Regression, SVM, PCA-তে এটিই সবচেয়ে বেশি ব্যবহৃত।</p>

    <h3>৩. MinMaxScaler (Normalization)</h3>
    <p>প্রতিটি মানকে ০ থেকে ১-এর মধ্যে নিয়ে আসা হয়।</p>
    <pre><code># সূত্র: x_scaled = (x − x_min) / (x_max − x_min)
# ফলাফল: সব মান [0, 1] রেঞ্জে

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_minmax)
#     বয়স   বেতন
# 0   0.00   0.00
# 1   0.25   0.25
# 2   0.50   0.50
# 3   0.75   0.75
# 4   1.00   1.00

# Custom range [0, 10]:
scaler_custom = MinMaxScaler(feature_range=(0, 10))
df_custom = scaler_custom.fit_transform(df)</code></pre>
    <p><strong>সমস্যা:</strong> Outlier থাকলে সব মান একটি ছোট range-এ চাপা পড়ে যায়। উদাহরণ: যদি একটি বেতন ৫০,০০,০০০ হয়, তাহলে বাকি সব মান ০-এর কাছাকাছি চলে আসবে।</p>

    <h3>৪. RobustScaler</h3>
    <p>Median এবং IQR (Interquartile Range) ব্যবহার করে — outlier-এ প্রভাবিত হয় না।</p>
    <pre><code># সূত্র: x_scaled = (x − median) / IQR
# IQR = Q3 − Q1 (75th percentile − 25th percentile)

from sklearn.preprocessing import RobustScaler

# Outlier সহ ডেটা
data_with_outlier = {
    'বয়স':   [25, 30, 35, 40, 45, 200],   # 200 একটি outlier
    'বেতন': [20000, 35000, 50000, 65000, 80000, 9000000]
}
df2 = pd.DataFrame(data_with_outlier)

robust = RobustScaler()
df2_scaled = pd.DataFrame(
    robust.fit_transform(df2),
    columns=df2.columns
)
print(df2_scaled)
# Outlier থাকলেও বাকি মানগুলো সঠিকভাবে scale হয়</code></pre>
    <p><strong>কখন ব্যবহার করবে:</strong> ডেটায় উল্লেখযোগ্য outlier থাকলে।</p>

    <h3>৫. তিনটি Scaler পাশাপাশি তুলনা</h3>
    <table>
      <thead><tr><th>Scaler</th><th>সূত্র</th><th>ফলাফলের range</th><th>Outlier-এ কেমন?</th><th>কখন ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td><strong>StandardScaler</strong></td><td>(x − μ) / σ</td><td>mean=0, std=1</td><td>প্রভাবিত হয়</td><td>Normal distribution, outlier কম</td></tr>
        <tr><td><strong>MinMaxScaler</strong></td><td>(x − min) / (max − min)</td><td>[0, 1]</td><td>খুব প্রভাবিত হয়</td><td>Image data, Neural Network</td></tr>
        <tr><td><strong>RobustScaler</strong></td><td>(x − median) / IQR</td><td>নির্দিষ্ট নয়</td><td>প্রায় প্রভাবিত হয় না</td><td>Outlier বেশি থাকলে</td></tr>
      </tbody>
    </table>

    <h3>৬. Scaling-এর সঠিক Pipeline</h3>
    <p>সবচেয়ে গুরুত্বপূর্ণ নিয়ম: <strong>test ডেটায় training ডেটার scaler ব্যবহার করো</strong> — test ডেটায় আলাদা fit করো না।</p>
    <pre><code>from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ সঠিক পদ্ধতি — Pipeline ব্যবহার করো
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression()),
])
pipeline.fit(X_train, y_train)
print("Test R²:", pipeline.score(X_test, y_test))

# ❌ ভুল পদ্ধতি — Data Leakage!
# scaler.fit(X)              ← পুরো X দিয়ে fit — test তথ্য ঢুকে যায়
# X_train_s = scaler.transform(X_train)
# X_test_s  = scaler.transform(X_test)</code></pre>

    <h3>৭. Scaling-এর পর Coefficient ব্যাখ্যা</h3>
    <p>Scaling করলে coefficient-এর magnitude বদলে যায়, কিন্তু মডেলের prediction একই থাকে। Scaled coefficient তুলনা করতে পারবে — কোন ফিচার বেশি গুরুত্বপূর্ণ।</p>
    <pre><code>import numpy as np

# Scaling ছাড়া
lr_raw = LinearRegression()
lr_raw.fit(X_train, y_train)
print("Unscaled coefficients:", lr_raw.coef_)
# [0.0002, 1.5, 300]  ← বেতনের coefficient অনেক ছোট দেখায়

# Scaling সহ
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_s, y_train)
print("Scaled coefficients:", lr_scaled.coef_)
# [0.45, 0.82, 0.61]  ← এখন সরাসরি তুলনা করা যাচ্ছে</code></pre>

    <h3>৮. কখন Scaling দরকার নেই?</h3>
    <table>
      <thead><tr><th>মডেল</th><th>Scaling দরকার?</th><th>কারণ</th></tr></thead>
      <tbody>
        <tr><td>Linear / Logistic Regression</td><td>হ্যাঁ</td><td>Gradient descent, regularization-এ প্রভাব পড়ে</td></tr>
        <tr><td>Ridge / Lasso</td><td>হ্যাঁ (আবশ্যক)</td><td>Penalty সমানভাবে প্রয়োগ হওয়া দরকার</td></tr>
        <tr><td>Decision Tree / Random Forest</td><td>না</td><td>Split rule scale-independent</td></tr>
        <tr><td>Gradient Boosting (XGBoost)</td><td>সাধারণত না</td><td>Tree-based, scale-independent</td></tr>
        <tr><td>KNN / SVM / PCA</td><td>হ্যাঁ</td><td>Distance-based — scale অনেক প্রভাব ফেলে</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>কেন scaling?</td><td>ফিচারের range আলাদা হলে Gradient Descent এবং regularization সঠিকভাবে কাজ করে না</td></tr>
        <tr><td>StandardScaler</td><td>Mean=0, Std=1 — সাধারণ ক্ষেত্রে সবচেয়ে ভালো</td></tr>
        <tr><td>MinMaxScaler</td><td>[0,1] range — outlier থাকলে ব্যবহার করো না</td></tr>
        <tr><td>RobustScaler</td><td>Outlier প্রতিরোধী — median ও IQR ব্যবহার করে</td></tr>
        <tr><td>Data Leakage এড়াও</td><td>শুধু train ডেটায় fit করো, test-এ শুধু transform</td></tr>
      </tbody>
    </table>
  `,
};
