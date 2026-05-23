export const lr_9_project = {
  title: "End-to-End Project: বাড়ির দাম ভবিষ্যদ্বাণী করো",
  description: "Linear Regression দিয়ে সম্পূর্ণ একটি real-world project — EDA, feature engineering, model training, evaluation, এবং deployment পর্যন্ত বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 14,
  slug: "lr-project-house-price",
  content: `
    <h3>প্রজেক্ট পরিচিতি</h3>
    <p>আমরা একটি বাড়ির বিভিন্ন তথ্য (আয়তন, বেডরুম, বয়স, অবস্থান) দিয়ে তার দাম ভবিষ্যদ্বাণী করবো। এটি লিনিয়ার রিগ্রেশন সিরিজের সব ধারণার প্রায়োগিক প্রয়োগ।</p>
    <table>
      <thead><tr><th>ধাপ</th><th>কী করবো</th></tr></thead>
      <tbody>
        <tr><td>১</td><td>ডেটা লোড ও দেখা (EDA)</td></tr>
        <tr><td>২</td><td>ডেটা পরিষ্কার ও Feature Engineering</td></tr>
        <tr><td>৩</td><td>Feature Selection</td></tr>
        <tr><td>৪</td><td>মডেল ট্রেনিং ও Regularization</td></tr>
        <tr><td>৫</td><td>মূল্যায়ন ও Residual Analysis</td></tr>
        <tr><td>৬</td><td>নতুন বাড়ির দাম বলা</td></tr>
      </tbody>
    </table>

    <h3>ধাপ ১ — ডেটা তৈরি ও দেখা</h3>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ডেটাসেট তৈরি (বাস্তবসম্মত)
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'area_sqft':   np.random.randint(600, 3500, n),
    'bedrooms':    np.random.randint(1, 6, n),
    'bathrooms':   np.random.randint(1, 4, n),
    'age_years':   np.random.randint(0, 40, n),
    'distance_km': np.random.randint(1, 30, n),
    'parking':     np.random.randint(0, 3, n),
    'floor':       np.random.randint(1, 8, n),
})

# দাম তৈরি (বাস্তবসম্মত সম্পর্ক)
df['price_lakh'] = (
    df['area_sqft'] * 0.045 +
    df['bedrooms'] * 3 +
    df['bathrooms'] * 2 -
    df['age_years'] * 0.8 -
    df['distance_km'] * 1.2 +
    df['parking'] * 5 +
    np.random.normal(0, 8, n)
).round(1)

print(f"ডেটাসেটের আকার: {df.shape}")
print(df.describe().round(2))</code></pre>

    <h3>ধাপ ২ — EDA (Exploratory Data Analysis)</h3>
    <pre><code># Target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['price_lakh'], bins=30, color='steelblue', edgecolor='white')
plt.title('দামের বিতরণ')
plt.xlabel('লক্ষ টাকা')

# Correlation with target
plt.subplot(1, 3, 2)
corr = df.corr()['price_lakh'].drop('price_lakh').sort_values()
corr.plot(kind='barh', color=['#dc2626' if v < 0 else '#1e40af' for v in corr])
plt.title('ফিচার Correlation (দামের সাথে)')

# Scatter: আয়তন vs দাম
plt.subplot(1, 3, 3)
plt.scatter(df['area_sqft'], df['price_lakh'], alpha=0.4, color='steelblue')
plt.xlabel('আয়তন (বর্গফুট)')
plt.ylabel('দাম (লক্ষ)')
plt.title('আয়তন vs দাম')

plt.tight_layout()
plt.show()</code></pre>

    <h3>ধাপ ৩ — Feature Engineering ও পরিষ্কার</h3>
    <pre><code"># নতুন ফিচার তৈরি
df['price_per_sqft'] = df['price_lakh'] / df['area_sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['is_new'] = (df['age_years'] < 5).astype(int)

# Outlier চেক
Q1 = df['price_lakh'].quantile(0.25)
Q3 = df['price_lakh'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[
    (df['price_lakh'] >= Q1 - 1.5 * IQR) &
    (df['price_lakh'] <= Q3 + 1.5 * IQR)
]
print(f"Outlier সরানোর পর: {len(df)} → {len(df_clean)} ডেটা")</code></pre>

    <h3>ধাপ ৪ — মডেল ট্রেনিং</h3>
    <pre><code">features = ['area_sqft', 'bedrooms', 'bathrooms', 'age_years',
            'distance_km', 'parking', 'floor', 'total_rooms', 'is_new']

X = df_clean[features]
y = df_clean['price_lakh']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# তিনটি মডেল তুলনা
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)':       Ridge(alpha=1.0),
    'Lasso (α=0.1)':     Lasso(alpha=0.1),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    results[name] = {
        'MAE':  mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²':   r2_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T.round(3)
print(results_df)</code></pre>

    <h3>ধাপ ৫ — Residual Analysis</h3>
    <pre><code">import scipy.stats as stats

best_model = Ridge(alpha=1.0)
best_model.fit(X_train_s, y_train)
y_pred_train = best_model.predict(X_train_s)
residuals = y_train - y_pred_train

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Residual plot
axes[0].scatter(y_pred_train, residuals, alpha=0.4, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuals vs Fitted')
axes[0].set_xlabel('Predicted Price')
axes[0].set_ylabel('Residual')

# Histogram
axes[1].hist(residuals, bins=25, color='steelblue', edgecolor='white')
axes[1].set_title('Residual বিতরণ')

# Q-Q Plot
stats.probplot(residuals, plot=axes[2])
axes[2].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()</code></pre>

    <h3>ধাপ ৬ — নতুন বাড়ির দাম বলো</h3>
    <pre><code"># নতুন একটি বাড়ির তথ্য
new_house_data = pd.DataFrame([{
    'area_sqft':   1800,
    'bedrooms':    3,
    'bathrooms':   2,
    'age_years':   4,
    'distance_km': 8,
    'parking':     1,
    'floor':       3,
    'total_rooms': 5,    # bedrooms + bathrooms
    'is_new':      1,    # age < 5 years
}])

new_house_scaled = scaler.transform(new_house_data)
price = best_model.predict(new_house_scaled)[0]
print(f"ভবিষ্যদ্বাণী করা দাম: {price:.1f} লক্ষ টাকা")</code></pre>

    <h3>চূড়ান্ত ফলাফল</h3>
    <table>
      <thead><tr><th>মডেল</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
      <tbody>
        <tr><td>Linear Regression</td><td>~7.2</td><td>~9.1</td><td>~0.87</td></tr>
        <tr><td>Ridge (α=1)</td><td>~6.9</td><td>~8.8</td><td>~0.88</td></tr>
        <tr><td>Lasso (α=0.1)</td><td>~7.1</td><td>~9.0</td><td>~0.87</td></tr>
      </tbody>
    </table>

    <h3>কী শিখলাম এই প্রজেক্টে?</h3>
    <table>
      <thead><tr><th>ধাপ</th><th>মূল শিক্ষা</th></tr></thead>
      <tbody>
        <tr><td>EDA</td><td>ডেটা বোঝা আগে, মডেল পরে</td></tr>
        <tr><td>Feature Engineering</td><td>নতুন ফিচার (মোট_রুম, নতুন_বাড়ি) মডেল উন্নত করে</td></tr>
        <tr><td>Outlier Removal</td><td>IQR পদ্ধতিতে চরম মান সরানো</td></tr>
        <tr><td>Scaling</td><td>StandardScaler ছাড়া Ridge/Lasso সঠিক কাজ করে না</td></tr>
        <tr><td>Model Comparison</td><td>Ridge প্রায়ই সাধারণ LR-এর চেয়ে ভালো</td></tr>
        <tr><td>Residual Analysis</td><td>মডেল ঠিক আছে কিনা চেক করা জরুরি</td></tr>
      </tbody>
    </table>
  `,
};
