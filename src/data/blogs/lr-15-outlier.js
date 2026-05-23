export const lr_15_outlier = {
  title: "Outlier Detection ও Treatment — ডেটার বিচ্ছিন্ন মানগুলো কীভাবে সামলাবো?",
  description: "Z-score, IQR পদ্ধতিতে outlier খোঁজা, লিনিয়ার রিগ্রেশনে outlier কী ক্ষতি করে, এবং সেগুলো সামলানোর বাস্তব কৌশল — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-outlier-detection",
  content: `
    <h3>১. Outlier কী?</h3>
    <p><strong>Outlier</strong> হলো এমন ডেটা পয়েন্ট যা বাকি সব ডেটা থেকে অনেক দূরে। উদাহরণ: ১০০ জন কর্মীর বেতনের ডেটায় ৯৯ জনের বেতন ২০,০০০–৮০,০০০ টাকা, কিন্তু একজন CEO-এর বেতন ৫০,০০,০০০ — এটি outlier।</p>
    <p>লিনিয়ার রিগ্রেশন outlier-এর প্রতি অনেক sensitive — কারণ MSE বর্গ করে, তাই বড় error আরও বড় হয়ে যায়।</p>

    <h3>২. Outlier রিগ্রেশনে কী করে?</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# স্বাভাবিক ডেটা
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([2,4,5,4,5,7,8,9,10,11])

# Outlier যোগ করা
X_out = np.append(X, [[15]], axis=0)
y_out = np.append(y, [2])   # outlier: x=15 কিন্তু y=2 (উলটো দিকে)

lr1 = LinearRegression().fit(X, y)
lr2 = LinearRegression().fit(X_out, y_out)

print(f"Outlier ছাড়া: slope={lr1.coef_[0]:.3f}")
print(f"Outlier সহ:   slope={lr2.coef_[0]:.3f}")
# Outlier ছাড়া: slope=0.927
# Outlier সহ:   slope=0.412  ← outlier slope অনেক বদলে দিয়েছে!</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Outlier রিগ্রেশন লাইন টেনে নামিয়ে দেয়</text>
        <!-- data points -->
        <circle cx="60"  cy="100" r="4" fill="#1e40af"/>
        <circle cx="100" cy="88"  r="4" fill="#1e40af"/>
        <circle cx="140" cy="80"  r="4" fill="#1e40af"/>
        <circle cx="180" cy="72"  r="4" fill="#1e40af"/>
        <circle cx="220" cy="60"  r="4" fill="#1e40af"/>
        <circle cx="260" cy="52"  r="4" fill="#1e40af"/>
        <circle cx="300" cy="44"  r="4" fill="#1e40af"/>
        <!-- outlier -->
        <circle cx="430" cy="108" r="6" fill="#dc2626"/>
        <text x="430" y="125" text-anchor="middle" font-size="8" fill="#dc2626">Outlier</text>
        <!-- correct line -->
        <line x1="40" y1="108" x2="320" y2="36" stroke="#16a34a" stroke-width="2" stroke-dasharray="5,3"/>
        <text x="180" y="30" font-size="8" fill="#16a34a">সঠিক লাইন</text>
        <!-- distorted line -->
        <line x1="40" y1="100" x2="450" y2="82" stroke="#dc2626" stroke-width="2"/>
        <text x="340" y="78" font-size="8" fill="#dc2626">Outlier-এ টানা লাইন</text>
      </svg>
    </div>

    <h3>৩. Z-score পদ্ধতি</h3>
    <p>যদি কোনো মান mean থেকে ৩ standard deviation-এর বেশি দূরে থাকে, সেটি outlier।</p>
    <pre><code>import pandas as pd
from scipy import stats

data = {
    'বেতন': [25000, 30000, 28000, 32000, 27000,
              29000, 500000, 31000, 26000, 33000]
}
df = pd.DataFrame(data)

# Z-score হিসাব
df['z_score'] = np.abs(stats.zscore(df['বেতন']))

# Z-score > 3 হলে outlier
outliers = df[df['z_score'] > 3]
print("Outlier পাওয়া গেছে:")
print(outliers)

# Clean data
df_clean = df[df['z_score'] <= 3].drop(columns='z_score')
print(f"\\nপরিষ্কার ডেটার আকার: {len(df_clean)}")</code></pre>

    <h3>৪. IQR পদ্ধতি (Interquartile Range)</h3>
    <p>Z-score Normal distribution ধরে নেয় — IQR পদ্ধতি আরও robust।</p>
    <pre><code>Q1 = df['বেতন'].quantile(0.25)
Q3 = df['বেতন'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Lower fence: {lower:,.0f}")
print(f"Upper fence: {upper:,.0f}")

outliers_iqr = df[(df['বেতন'] < lower) | (df['বেতন'] > upper)]
print(f"Outlier সংখ্যা: {len(outliers_iqr)}")

df_clean_iqr = df[(df['বেতন'] >= lower) & (df['বেতন'] <= upper)]</code></pre>

    <h3>৫. Isolation Forest — অনেক ফিচার থাকলে</h3>
    <pre><code>from sklearn.ensemble import IsolationForest

# একাধিক ফিচারে outlier খোঁজা
X_multi = np.column_stack([
    [25,30,28,32,27,29,200,31,26,33],  # বয়স
    [25000,30000,28000,32000,27000,
     29000,500000,31000,26000,33000]   # বেতন
])

iso = IsolationForest(contamination=0.1, random_state=42)
labels = iso.fit_predict(X_multi)
# -1 = outlier, 1 = inlier

print("Outlier indices:", np.where(labels == -1)[0])
X_clean = X_multi[labels == 1]</code></pre>

    <h3>৬. Outlier সামলানোর কৌশল</h3>
    <table>
      <thead><tr><th>কৌশল</th><th>কখন ব্যবহার</th><th>সতর্কতা</th></tr></thead>
      <tbody>
        <tr><td><strong>বাদ দেওয়া</strong></td><td>ডেটা entry error বা genuine mistake</td><td>বৈধ rare event হলে বাদ দিও না</td></tr>
        <tr><td><strong>Cap/Winsorize</strong></td><td>extreme মান আছে কিন্তু বাদ দিতে চাও না</td><td>তথ্য হারায় না কিন্তু বিকৃত হয়</td></tr>
        <tr><td><strong>Log Transform</strong></td><td>right-skewed data (বেতন, দাম)</td><td>শুধু positive মানে কাজ করে</td></tr>
        <tr><td><strong>RobustScaler</strong></td><td>scaling দরকার, outlier আছে</td><td>outlier সরায় না, প্রভাব কমায়</td></tr>
        <tr><td><strong>Robust Regression</strong></td><td>outlier রাখতে হবে কিন্তু প্রভাব কমাতে হবে</td><td>সাধারণ OLS-এর চেয়ে ধীর</td></tr>
      </tbody>
    </table>
    <pre><code># Winsorize — upper/lower 5% capping
from scipy.stats.mstats import winsorize

df['বেতন_capped'] = winsorize(df['বেতন'], limits=[0.05, 0.05])

# Log Transform
df['বেতন_log'] = np.log1p(df['বেতন'])  # log1p = log(1+x), 0 সামলায়

# Robust Regression (HuberRegressor — outlier-এ কম sensitive)
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)
print("Huber coef:", huber.coef_)</code></pre>

    <h3>৭. Outlier বাদ দেওয়ার আগে ভাবো</h3>
    <ul>
      <li>এটি কি সত্যিই ভুল ডেটা, নাকি বিরল কিন্তু বাস্তব ঘটনা?</li>
      <li>এই outlier কি domain-specific গুরুত্বপূর্ণ? (যেমন fraud detection-এ outlier-ই target)</li>
      <li>Outlier বাদ দিলে model কতটুকু উন্নত হয়? দুটো model তুলনা করো।</li>
    </ul>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Outlier-এর প্রভাব</td><td>regression line টেনে নামায়, coefficient বিকৃত করে</td></tr>
        <tr><td>Z-score</td><td>|z| &gt; 3 হলে outlier — normal data-তে ভালো</td></tr>
        <tr><td>IQR</td><td>Q1−1.5×IQR থেকে Q3+1.5×IQR — robust পদ্ধতি</td></tr>
        <tr><td>Isolation Forest</td><td>multivariate outlier খোঁজার জন্য</td></tr>
        <tr><td>Treatment</td><td>বাদ দেওয়া / cap / log transform / robust regression</td></tr>
      </tbody>
    </table>
  `,
};
