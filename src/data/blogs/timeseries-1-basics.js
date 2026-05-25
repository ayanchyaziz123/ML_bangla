export const timeseries_1_basics = {
  slug: 'timeseries-1-basics',
  title: 'টাইম সিরিজ: সময়ের সাথে ডেটা বিশ্লেষণ',
  description: 'টাইম সিরিজ কী, ট্রেন্ড/সিজনালিটি/সাইক্লিসিটি/নয়েজ ডিকম্পোজিশন, স্টেশনারিটি (ADF টেস্ট), অটোকোরিলেশন (ACF/PACF), এবং pandas দিয়ে datetime বিশ্লেষণ।',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 12,
  content: `
    <h3>১. টাইম সিরিজ কী?</h3>
    <p>
      <strong>টাইম সিরিজ</strong> হলো সময়ের ক্রমানুযায়ী সংগ্রহ করা ডেটার একটি ধারা। প্রতিটি ডেটা পয়েন্টের সাথে একটি নির্দিষ্ট সময়-স্ট্যাম্প থাকে। উদাহরণ: প্রতিদিনের তাপমাত্রা, মাসিক বিক্রয়, ঘণ্টায় ঘণ্টায় বিদ্যুৎ ব্যবহার, বার্ষিক জিডিপি।
    </p>
    <p>
      সাধারণ মেশিন লার্নিং থেকে টাইম সিরিজ আলাদা কারণ এখানে <strong>পর্যবেক্ষণগুলো স্বাধীন নয়</strong> — আজকের ডেটা গতকালের ডেটার উপর নির্ভর করে। এই temporal dependency-কে সঠিকভাবে মডেল করাটাই টাইম সিরিজ অ্যানালিসিসের মূল চ্যালেঞ্জ।
    </p>
    <p>টাইম সিরিজের সাধারণ প্রয়োগ:</p>
    <ul>
      <li><strong>আর্থিক পূর্বাভাস:</strong> শেয়ার দাম, মুদ্রার বিনিময় হার</li>
      <li><strong>আবহাওয়া পূর্বাভাস:</strong> বৃষ্টিপাত, তাপমাত্রা</li>
      <li><strong>চাহিদা পরিকল্পনা:</strong> পণ্যের বিক্রয় পূর্বাভাস</li>
      <li><strong>স্বাস্থ্যসেবা:</strong> রোগীর পর্যবেক্ষণ, মহামারির ট্র্যাকিং</li>
      <li><strong>শিল্প:</strong> সেন্সর ডেটা, যন্ত্রপাতির রক্ষণাবেক্ষণ</li>
    </ul>

    <h3>২. টাইম সিরিজের উপাদান: ডিকম্পোজিশন</h3>
    <p>
      যেকোনো টাইম সিরিজকে চারটি মূল উপাদানে ভাগ করা যায়। এই উপাদানগুলো বোঝা সঠিক মডেল বেছে নেওয়ার জন্য অপরিহার্য।
    </p>

    <h4>২.১ ট্রেন্ড (Trend)</h4>
    <p>
      দীর্ঘমেয়াদী উপর বা নিচে যাওয়ার প্রবণতা। উদাহরণ: গত ২০ বছরে বাংলাদেশের জিডিপি ধীরে ধীরে বৃদ্ধি পাচ্ছে — এটি একটি <strong>upward trend</strong>। আবার কোনো পণ্যের চাহিদা বছরের পর বছর কমছে — এটি <strong>downward trend</strong>।
    </p>

    <h4>২.২ সিজনালিটি (Seasonality)</h4>
    <p>
      নিয়মিত, পুনরাবৃত্তিমূলক প্যাটার্ন যা নির্দিষ্ট সময়ের ব্যবধানে ঘটে। উদাহরণ: প্রতি বছর ঈদের সময় কাপড়ের বিক্রয় বাড়ে, প্রতি সপ্তাহে সোমবার ট্রাফিক বেশি থাকে। সিজনালিটির period নির্দিষ্ট (দৈনিক, সাপ্তাহিক, মাসিক, বার্ষিক)।
    </p>

    <h4>২.৩ সাইক্লিসিটি (Cyclicity)</h4>
    <p>
      অনিয়মিত, দীর্ঘ-সময়ের ওঠানামা যার কোনো নির্দিষ্ট period নেই। উদাহরণ: অর্থনৈতিক মন্দা ও সমৃদ্ধির চক্র। এটি সিজনালিটি থেকে আলাদা কারণ এর সময়কাল পরিবর্তনশীল।
    </p>

    <h4>২.৪ নয়েজ / অনিয়মিততা (Noise/Irregular)</h4>
    <p>
      এলোমেলো, অপ্রত্যাশিত ওঠানামা যা কোনো প্যাটার্ন অনুসরণ করে না। এটি পরিমাপের ত্রুটি বা অপ্রত্যাশিত ঘটনার (যেমন করোনা মহামারি) কারণে হতে পারে।
    </p>

    <h4>Additive vs Multiplicative ডিকম্পোজিশন</h4>
    <p>
      <strong>Additive:</strong> y(t) = Trend + Seasonality + Cyclicity + Noise — যখন seasonal variation সময়ের সাথে একই থাকে।<br/>
      <strong>Multiplicative:</strong> y(t) = Trend × Seasonality × Cyclicity × Noise — যখন seasonal variation ট্রেন্ডের সাথে বাড়ে।
    </p>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# সিন্থেটিক টাইম সিরিজ তৈরি
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 5, 365)
y = trend + seasonality + noise

ts = pd.Series(y, index=dates, name='value')

# Additive ডিকম্পোজিশন
result = seasonal_decompose(ts, model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
result.observed.plot(ax=axes[0], title='Original Series')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonality')
result.resid.plot(ax=axes[3], title='Residual (Noise)')
plt.tight_layout()
plt.show()

print("Decomposition সম্পন্ন!")
print(f"ট্রেন্ডের পরিসর: {result.trend.dropna().min():.2f} থেকে {result.trend.dropna().max():.2f}")</code></pre>

    <h3>৩. স্টেশনারিটি (Stationarity)</h3>
    <p>
      একটি টাইম সিরিজকে <strong>স্টেশনারি</strong> বলা হয় যদি এর statistical properties (mean, variance, autocorrelation) সময়ের সাথে পরিবর্তিত না হয়। বেশিরভাগ ক্লাসিক্যাল টাইম সিরিজ মডেল (ARIMA) স্টেশনারি ডেটা ধরে নেয়।
    </p>
    <p>
      <strong>Non-stationary সিরিজের লক্ষণ:</strong>
    </p>
    <ul>
      <li>স্পষ্ট ট্রেন্ড (mean পরিবর্তন হচ্ছে)</li>
      <li>পরিবর্তনশীল variance (heteroscedasticity)</li>
      <li>সিজনাল প্যাটার্ন</li>
    </ul>

    <h4>৩.১ ADF টেস্ট (Augmented Dickey-Fuller Test)</h4>
    <p>
      ADF টেস্ট সবচেয়ে জনপ্রিয় স্টেশনারিটি পরীক্ষা। এটি null hypothesis হিসেবে ধরে নেয় যে সিরিজ non-stationary (unit root আছে)।
    </p>
    <ul>
      <li>p-value &lt; 0.05: সিরিজটি স্টেশনারি (null reject করা হয়)</li>
      <li>p-value &gt; 0.05: সিরিজটি non-stationary</li>
    </ul>
    <pre><code>from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f"--- ADF টেস্ট: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, val in result[4].items():
        print(f"  {key}: {val:.4f}")
    if result[1] &lt; 0.05:
        print("সিদ্ধান্ত: সিরিজটি STATIONARY (p &lt; 0.05)")
    else:
        print("সিদ্ধান্ত: সিরিজটি NON-STATIONARY (p &gt;= 0.05)")
    print()

# Non-stationary সিরিজ পরীক্ষা
adf_test(ts, name='মূল সিরিজ')

# Differencing দিয়ে Stationary করা
ts_diff = ts.diff().dropna()
adf_test(ts_diff, name='প্রথম ডিফারেন্স')

# Log transformation + differencing
ts_log_diff = np.log(ts).diff().dropna()
adf_test(ts_log_diff, name='Log + Difference')</code></pre>

    <h4>৩.২ KPSS টেস্ট</h4>
    <p>
      KPSS টেস্ট ADF-এর বিপরীত — এটি null hypothesis হিসেবে ধরে নেয় যে সিরিজ <em>stationary</em>। দুটি টেস্ট একসাথে ব্যবহার করলে আরও নিশ্চিত সিদ্ধান্ত নেওয়া যায়।
    </p>
    <pre><code>from statsmodels.tsa.stattools import kpss

def kpss_test(series, name=''):
    result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"--- KPSS টেস্ট: {name} ---")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] &gt; 0.05:
        print("সিদ্ধান্ত: সিরিজটি STATIONARY (p &gt; 0.05)")
    else:
        print("সিদ্ধান্ত: সিরিজটি NON-STATIONARY (p &lt;= 0.05)")

kpss_test(ts, name='মূল সিরিজ')</code></pre>

    <h3>৪. অটোকোরিলেশন: ACF ও PACF</h3>
    <p>
      <strong>Autocorrelation</strong> হলো একটি সিরিজ এবং তার নিজেরই time-lagged version-এর মধ্যে correlation। এটি আমাদের বলে দেয় অতীতের কোন মুহূর্তের মান বর্তমান মানকে কতটা প্রভাবিত করে।
    </p>

    <h4>৪.১ ACF (Autocorrelation Function)</h4>
    <p>
      ACF lag k-এ y(t) এবং y(t-k)-এর মধ্যে correlation দেখায়। এটি সরাসরি এবং পরোক্ষ উভয় প্রভাবই ধরে।
    </p>

    <h4>৪.২ PACF (Partial Autocorrelation Function)</h4>
    <p>
      PACF মাঝের lag-গুলোর প্রভাব বাদ দিয়ে শুধু y(t) এবং y(t-k)-এর মধ্যে সরাসরি correlation দেখায়। ARIMA মডেলের parameter (p, q) নির্ধারণে ACF ও PACF অপরিহার্য।
    </p>
    <pre><code>from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# স্টেশনারি সিরিজের ACF ও PACF প্লট
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

plot_acf(ts_diff, lags=40, ax=axes[0], title='ACF (প্রথম ডিফারেন্স)')
plot_pacf(ts_diff, lags=40, ax=axes[1], title='PACF (প্রথম ডিফারেন্স)')

plt.tight_layout()
plt.show()

# Numerical values
acf_values = acf(ts_diff, nlags=10)
pacf_values = pacf(ts_diff, nlags=10)

print("ACF মান (lag 1-10):")
for i, v in enumerate(acf_values[1:], 1):
    print(f"  Lag {i}: {v:.4f}")

print("\nPACF মান (lag 1-10):")
for i, v in enumerate(pacf_values[1:], 1):
    print(f"  Lag {i}: {v:.4f}")</code></pre>

    <h3>৫. Pandas দিয়ে Datetime বিশ্লেষণ</h3>
    <p>
      Pandas টাইম সিরিজ কাজের জন্য অত্যন্ত শক্তিশালী। <code>DatetimeIndex</code> ব্যবহার করে অনেক সুবিধাজনক অপারেশন করা যায়।
    </p>

    <h4>৫.১ Datetime Index তৈরি ও Parsing</h4>
    <pre><code>import pandas as pd
import numpy as np

# CSV থেকে ডেটা লোড করার সময় datetime parsing
df = pd.read_csv('energy_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# অথবা manually convert করা
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.set_index('date')
df = df.sort_index()  # সময় অনুযায়ী sort করা

# Date range তৈরি
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
print(f"মোট দিন: {len(dates)}")

# সময়-ভিত্তিক feature extract করা
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek  # 0=সোমবার, 6=রবিবার
df['hour'] = df.index.hour
df['quarter'] = df.index.quarter
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

print(df.head())</code></pre>

    <h4>৫.২ Resampling: সময়ের granularity পরিবর্তন</h4>
    <p>
      Resampling হলো একটি সিরিজকে ভিন্ন time frequency-তে রূপান্তর করা। Upsampling মানে ঘন ঘন (ঘণ্টা → মিনিট), Downsampling মানে বিরল (ঘণ্টা → দিন)।
    </p>
    <pre><code># ঘণ্টায় ঘণ্টায় ডেটা তৈরি
hourly_data = pd.Series(
    np.random.randn(24 * 30),
    index=pd.date_range('2023-01-01', periods=24*30, freq='h')
)

# Downsampling: ঘণ্টা → দিন
daily_mean = hourly_data.resample('D').mean()
daily_max = hourly_data.resample('D').max()
daily_sum = hourly_data.resample('D').sum()

# Downsampling: দিন → সপ্তাহ
weekly_data = hourly_data.resample('W').mean()

# Downsampling: দিন → মাস
monthly_data = hourly_data.resample('ME').mean()

# Custom aggregation
monthly_stats = hourly_data.resample('ME').agg({
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    'max': 'max'
})

print("দৈনিক গড়:")
print(daily_mean.head(7))
print(f"\nমাসিক ডেটা:")
print(monthly_data)</code></pre>

    <h4>৫.৩ Rolling Statistics</h4>
    <pre><code># Rolling window statistics
ts_series = pd.Series(np.random.randn(365),
                      index=pd.date_range('2023-01-01', periods=365, freq='D'))

# 7-দিনের rolling mean
rolling_7 = ts_series.rolling(window=7).mean()

# 30-দিনের rolling std
rolling_std_30 = ts_series.rolling(window=30).std()

# Exponentially Weighted Mean
ewm_span = ts_series.ewm(span=7).mean()

# Expanding window (cumulative)
expanding_mean = ts_series.expanding().mean()

plt.figure(figsize=(14, 6))
plt.plot(ts_series, alpha=0.4, label='মূল ডেটা')
plt.plot(rolling_7, label='7-দিনের Rolling Mean', linewidth=2)
plt.plot(ewm_span, label='EWM (span=7)', linewidth=2, linestyle='--')
plt.legend()
plt.title('Rolling Statistics')
plt.show()</code></pre>

    <h3>৬. Time Series Visualization</h3>
    <pre><code>import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# বছরব্যাপী বিক্রয় ডেটা সিমুলেশন
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
trend = np.linspace(1000, 1500, len(dates))
seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
weekly = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
noise = np.random.normal(0, 30, len(dates))
sales = trend + seasonal + weekly + noise

df = pd.DataFrame({'sales': sales}, index=dates)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# মূল সিরিজ
axes[0].plot(df.index, df['sales'], color='steelblue', alpha=0.8, linewidth=0.8)
axes[0].set_title('দৈনিক বিক্রয় (২০২২-২০২৩)', fontsize=14)
axes[0].set_ylabel('বিক্রয় (টাকা)')

# মাসিক গড়
monthly = df['sales'].resample('ME').mean()
axes[1].bar(monthly.index, monthly.values, color='coral', alpha=0.8, width=20)
axes[1].set_title('মাসিক গড় বিক্রয়', fontsize=14)
axes[1].set_ylabel('গড় বিক্রয়')

# Rolling mean comparison
rolling_30 = df['sales'].rolling(30).mean()
axes[2].plot(df.index, df['sales'], alpha=0.3, label='দৈনিক', color='gray')
axes[2].plot(df.index, rolling_30, color='red', linewidth=2, label='30-দিন Rolling Mean')
axes[2].set_title('Smoothed সিরিজ', fontsize=14)
axes[2].legend()

plt.tight_layout()
plt.savefig('timeseries_visualization.png', dpi=150)
plt.show()</code></pre>

    <h3>৭. Lag Plot ও Seasonality Detection</h3>
    <pre><code>from pandas.plotting import lag_plot, autocorrelation_plot

# Lag Plot: consecutive observations-এর মধ্যে correlation দেখায়
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, lag in enumerate([1, 7, 30]):
    lag_plot(df['sales'], lag=lag, ax=axes[i])
    axes[i].set_title(f'Lag {lag} Plot')

plt.tight_layout()
plt.show()

# Seasonality check: box plot by month
df['month'] = df.index.month
df['weekday'] = df.index.day_name()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df.boxplot(column='sales', by='month', ax=axes[0])
axes[0].set_title('মাস অনুযায়ী বিক্রয় বিতরণ')
axes[0].set_xlabel('মাস')

# সপ্তাহের দিন অনুযায়ী
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df.groupby('weekday')['sales'].mean().reindex(weekday_order).plot(kind='bar', ax=axes[1], color='teal')
axes[1].set_title('সপ্তাহের দিন অনুযায়ী গড় বিক্রয়')
axes[1].set_xlabel('দিন')

plt.tight_layout()
plt.show()</code></pre>

    <h3>৮. সারসংক্ষেপ</h3>
    <p>
      এই পর্বে আমরা টাইম সিরিজ বিশ্লেষণের মূল ভিত্তি শিখেছি:
    </p>
    <ul>
      <li><strong>ডিকম্পোজিশন:</strong> ট্রেন্ড, সিজনালিটি, সাইক্লিসিটি, এবং নয়েজ আলাদা করা</li>
      <li><strong>স্টেশনারিটি:</strong> ADF ও KPSS টেস্ট দিয়ে পরীক্ষা করা, differencing দিয়ে ঠিক করা</li>
      <li><strong>ACF/PACF:</strong> অটোকোরিলেশন বোঝা এবং মডেল order নির্ধারণ করা</li>
      <li><strong>Pandas:</strong> DatetimeIndex, resampling, rolling statistics</li>
    </ul>
    <p>
      পরবর্তী পর্বে আমরা ARIMA মডেল দিয়ে সত্যিকারের পূর্বাভাস দিতে শিখব।
    </p>
  `
};
