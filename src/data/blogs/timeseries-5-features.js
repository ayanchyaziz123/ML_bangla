export const timeseries_5_features = {
  slug: 'timeseries-5-features',
  title: 'টাইম সিরিজ ফিচার ইঞ্জিনিয়ারিং',
  description: 'Lag ফিচার, রোলিং স্ট্যাটিস্টিক্স, ক্যালেন্ডার ফিচার ও Fourier রূপান্তর দিয়ে শক্তিশালী টাইম সিরিজ ফিচার তৈরি করুন',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 12,
  content: `
<h3>কেন ফিচার ইঞ্জিনিয়ারিং দরকার?</h3>
<p>ভালো ফিচার তৈরি করলে সাধারণ মডেল (XGBoost, LightGBM) প্রায়ই LSTM-এর চেয়ে ভালো ফলাফল দেয় এবং ব্যাখ্যা করা অনেক সহজ।</p>

<h3>ডেটা প্রস্তুতি</h3>

<pre><code class="language-python">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# মাসিক বিক্রয় ডেটা
df = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date')
df = df.sort_index()
print(df.head())
#             sales
# 2020-01-01   1234
# 2020-01-02   1456
# ...
</code></pre>

<h3>Lag ফিচার</h3>
<p>Lag ফিচার মানে আগের সময়ের মান। <code>lag_7</code> মানে ৭ দিন আগের বিক্রয়।</p>

<pre><code class="language-python">def add_lag_features(df, target='sales', lags=[1, 7, 14, 30]):
    for lag in lags:
        df[f'lag_{lag}'] = df[target].shift(lag)
    return df

df = add_lag_features(df)
print("Lag ফিচার যোগ করা হয়েছে:")
print(df[['sales', 'lag_1', 'lag_7', 'lag_14']].head(35).tail(5))
</code></pre>

<h3>রোলিং স্ট্যাটিস্টিক্স</h3>

<pre><code class="language-python">def add_rolling_features(df, target='sales', windows=[7, 30]):
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df[target].shift(1).rolling(window).std()
        df[f'rolling_min_{window}'] = df[target].shift(1).rolling(window).min()
        df[f'rolling_max_{window}'] = df[target].shift(1).rolling(window).max()
        # শতাংশ পরিবর্তন
        df[f'rolling_pct_{window}'] = (
            (df[target] - df[target].shift(window)) / df[target].shift(window)
        )
    return df

df = add_rolling_features(df)
</code></pre>

<h3>ক্যালেন্ডার ফিচার</h3>

<pre><code class="language-python">def add_calendar_features(df):
    df['বছর'] = df.index.year
    df['মাস'] = df.index.month
    df['সপ্তাহ'] = df.index.isocalendar().week.astype(int)
    df['সপ্তাহের_দিন'] = df.index.dayofweek      # 0=সোম, 6=রবি
    df['মাসের_দিন'] = df.index.day
    df['বছরের_দিন'] = df.index.dayofyear
    df['ত্রৈমাসিক'] = df.index.quarter
    df['সাপ্তাহান্ত'] = (df.index.dayofweek >= 5).astype(int)
    df['মাসের_শুরু'] = df.index.is_month_start.astype(int)
    df['মাসের_শেষ'] = df.index.is_month_end.astype(int)
    return df

df = add_calendar_features(df)
</code></pre>

<h3>Fourier ফিচার (ঋতু ধরার জন্য)</h3>

<pre><code class="language-python">def add_fourier_features(df, period=365, n_harmonics=3):
    """সাইনাসোইডাল ফিচার দিয়ে ঋতু ধরা"""
    t = np.arange(len(df))
    for k in range(1, n_harmonics + 1):
        df[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df

# বার্ষিক এবং সাপ্তাহিক ঋতু
df = add_fourier_features(df, period=365, n_harmonics=3)  # বার্ষিক
df = add_fourier_features(df, period=7, n_harmonics=2)    # সাপ্তাহিক
</code></pre>

<h3>মডেল ও ফিচার গুরুত্ব</h3>

<pre><code class="language-python"># NaN বাদ দেওয়া (lag/rolling কারণে প্রথম কিছু সারি NaN)
df_clean = df.dropna()
feature_cols = [c for c in df_clean.columns if c != 'sales']
X = df_clean[feature_cols]
y = df_clean['sales']

# ট্রেন-টেস্ট বিভাজন (সময়-ক্রমানুসারে)
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# GBM মডেল
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# ফিচার গুরুত্ব
importance_df = pd.DataFrame({
    'ফিচার': feature_cols,
    'গুরুত্ব': model.feature_importances_
}).sort_values('গুরুত্ব', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['ফিচার'][:15], importance_df['গুরুত্ব'][:15])
plt.xlabel('গুরুত্ব')
plt.title('শীর্ষ ১৫ ফিচার গুরুত্ব')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
</code></pre>

<h4>সেরা ফিচারের তালিকা (সাধারণত)</h4>
<ol>
<li><strong>lag_1:</strong> গতকালের বিক্রয় — সবচেয়ে গুরুত্বপূর্ণ</li>
<li><strong>lag_7:</strong> এক সপ্তাহ আগের বিক্রয়</li>
<li><strong>rolling_mean_7:</strong> সাপ্তাহিক গড়</li>
<li><strong>সপ্তাহের_দিন:</strong> দিনের ধরনগত প্যাটার্ন</li>
<li><strong>sin/cos Fourier:</strong> ঋতু ধরার জন্য</li>
</ol>
`
};
