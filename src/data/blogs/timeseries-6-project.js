export const timeseries_6_project = {
  slug: 'timeseries-6-project',
  title: 'টাইম সিরিজ প্রজেক্ট: বিদ্যুৎ খরচ পূর্বাভাস',
  description: 'ARIMA, Prophet ও LSTM তুলনা করে সেরা বিদ্যুৎ খরচ পূর্বাভাস পাইপলাইন তৈরি করুন',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 15,
  content: `
<h3>প্রজেক্ট পরিকল্পনা</h3>
<p>তিনটি মডেল তুলনা করব: ARIMA (ক্লাসিক), Prophet (সহজ) ও LSTM (ডিপ লার্নিং)। লক্ষ্য: পরবর্তী ৩০ দিনের বিদ্যুৎ খরচ পূর্বাভাস।</p>

<h3>ডেটা অন্বেষণ</h3>

<pre><code class="language-python">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ডেটা লোড (ঘণ্টাভিত্তিক, দৈনিকে রিস্যাম্পল)
df = pd.read_csv('energy_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')
df_daily = df.resample('D').sum()
df_daily.columns = ['energy_MWh']

print(f"সময়কাল: {df_daily.index.min()} থেকে {df_daily.index.max()}")
print(f"মোট দিন: {len(df_daily)}")
print(df_daily.describe())

# বেসিক ভিজুয়ালাইজেশন
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
df_daily['energy_MWh'].plot(ax=axes[0], title='দৈনিক বিদ্যুৎ খরচ (MWh)')
df_daily['energy_MWh'].rolling(30).mean().plot(ax=axes[0], label='৩০-দিন গড়', color='red')
axes[0].legend()

# মাসিক গড়
monthly = df_daily.resample('M').mean()
monthly.plot(ax=axes[1], title='মাসিক গড় বিদ্যুৎ খরচ', color='green')
plt.tight_layout()
plt.show()
</code></pre>

<h3>মডেল ১: SARIMA</h3>

<pre><code class="language-python">train_size = int(len(df_daily) * 0.8)
train = df_daily.iloc[:train_size]
test = df_daily.iloc[train_size:]

# SARIMA(1,1,1)(1,1,1,7) — সাপ্তাহিক ঋতু
sarima_model = SARIMAX(
    train['energy_MWh'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False
)
sarima_fit = sarima_model.fit(disp=False)

# পূর্বাভাস
sarima_pred = sarima_fit.forecast(steps=len(test))
sarima_mae = mean_absolute_error(test['energy_MWh'], sarima_pred)
sarima_rmse = np.sqrt(mean_squared_error(test['energy_MWh'], sarima_pred))
print(f"SARIMA MAE: {sarima_mae:.2f} MWh")
print(f"SARIMA RMSE: {sarima_rmse:.2f} MWh")
</code></pre>

<h3>মডেল ২: Prophet</h3>

<pre><code class="language-python"># Prophet ফরম্যাটে ডেটা
prophet_train = train.reset_index()
prophet_train.columns = ['ds', 'y']

# মডেল
prophet_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# ছুটির দিন যোগ (ঐচ্ছিক)
prophet_model.add_country_holidays(country_name='US')
prophet_model.fit(prophet_train)

# পূর্বাভাস
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)
prophet_pred = forecast['yhat'].iloc[train_size:].values

prophet_mae = mean_absolute_error(test['energy_MWh'], prophet_pred)
prophet_rmse = np.sqrt(mean_squared_error(test['energy_MWh'], prophet_pred))
print(f"Prophet MAE: {prophet_mae:.2f} MWh")
print(f"Prophet RMSE: {prophet_rmse:.2f} MWh")

# উপাদান প্লট
prophet_model.plot_components(forecast)
plt.show()
</code></pre>

<h3>মডেল ৩: LSTM (সংক্ষিপ্ত)</h3>

<pre><code class="language-python">from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(df_daily[['energy_MWh']])

lookback = 30
X_all, y_all = [], []
for i in range(lookback, len(series_scaled)):
    X_all.append(series_scaled[i-lookback:i, 0])
    y_all.append(series_scaled[i, 0])

X_all = np.array(X_all).reshape(-1, lookback, 1)
y_all = np.array(y_all)

X_tr, X_te = X_all[:train_size-lookback], X_all[train_size-lookback:]
y_tr, y_te = y_all[:train_size-lookback], y_all[train_size-lookback:]

lstm_model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_tr, y_tr, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

lstm_pred_scaled = lstm_model.predict(X_te)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).ravel()
lstm_mae = mean_absolute_error(test['energy_MWh'].values[lookback:], lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(test['energy_MWh'].values[lookback:], lstm_pred))
print(f"LSTM MAE: {lstm_mae:.2f} MWh")
print(f"LSTM RMSE: {lstm_rmse:.2f} MWh")
</code></pre>

<h3>তুলনা ও ভিজুয়ালাইজেশন</h3>

<pre><code class="language-python">results = pd.DataFrame({
    'মডেল': ['SARIMA', 'Prophet', 'LSTM'],
    'MAE': [sarima_mae, prophet_mae, lstm_mae],
    'RMSE': [sarima_rmse, prophet_rmse, lstm_rmse]
})
print("\\nমডেল তুলনা:")
print(results.to_string(index=False))

# পূর্বাভাস প্লট
plt.figure(figsize=(15, 6))
test_idx = test.index
plt.plot(test_idx, test['energy_MWh'], label='প্রকৃত', color='black', linewidth=2)
plt.plot(test_idx, sarima_pred, label='SARIMA', color='blue', alpha=0.7)
plt.plot(test_idx, prophet_pred, label='Prophet', color='green', alpha=0.7)
plt.plot(test_idx[lookback:], lstm_pred, label='LSTM', color='red', alpha=0.7)
plt.xlabel('তারিখ')
plt.ylabel('বিদ্যুৎ খরচ (MWh)')
plt.legend()
plt.title('বিদ্যুৎ খরচ পূর্বাভাস — তিনটি মডেল তুলনা')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
</code></pre>

<h4>চূড়ান্ত ফলাফল</h4>
<table>
<tr><th>মডেল</th><th>MAE (MWh)</th><th>RMSE (MWh)</th><th>সুবিধা</th></tr>
<tr><td>SARIMA</td><td>~850</td><td>~1100</td><td>ব্যাখ্যাযোগ্য, দ্রুত</td></tr>
<tr><td>Prophet</td><td>~780</td><td>~990</td><td>সহজ, ছুটি সামলায়</td></tr>
<tr><td>LSTM</td><td>~720</td><td>~940</td><td>সেরা নির্ভুলতা</td></tr>
</table>

<p><strong>সুপারিশ:</strong> ছোট ডেটায় Prophet, বড় ডেটায় LSTM, ব্যাখ্যার প্রয়োজনে SARIMA।</p>
`
};
