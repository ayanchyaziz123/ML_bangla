export const timeseriesEn = [
  {
    slug: 'timeseries-1-basics',
    title: 'Time Series: Analyzing Data Over Time',
    description: 'Trend, seasonality, stationarity (ADF test), autocorrelation, and pandas datetime',
    category: 'Time Series',
    content: `
<h3>Time Series Components</h3>
<ul>
<li><strong>Trend:</strong> Long-term increase or decrease</li>
<li><strong>Seasonality:</strong> Repeating pattern at fixed intervals</li>
<li><strong>Cyclicity:</strong> Non-fixed repeating patterns</li>
<li><strong>Residual:</strong> Random noise</li>
</ul>

<pre><code class="language-python">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create sample time series
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(0, 50, 365)
seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 5, 365)
ts = pd.Series(trend + seasonal + noise, index=dates)

# Decomposition
decomp = seasonal_decompose(ts, model='additive', period=30)
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomp.observed.plot(ax=axes[0], title='Observed')
decomp.trend.plot(ax=axes[1], title='Trend')
decomp.seasonal.plot(ax=axes[2], title='Seasonal')
decomp.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Stationarity Test
result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"Stationary: {result[1] < 0.05}")

# ACF / PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(ts, lags=50, ax=ax1, title='Autocorrelation')
plot_pacf(ts, lags=50, ax=ax2, title='Partial Autocorrelation')
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'timeseries-2-arima',
    title: 'ARIMA: Classical Time Series Forecasting',
    description: 'AR, MA, differencing, ARIMA parameter selection, and SARIMA for seasonal patterns',
    category: 'Time Series',
    content: `
<h3>ARIMA Parameters</h3>
<ul>
<li><strong>AR(p):</strong> Autoregressive — regression on past p values</li>
<li><strong>I(d):</strong> Integrated — d-th order differencing for stationarity</li>
<li><strong>MA(q):</strong> Moving Average — regression on past q errors</li>
</ul>

<pre><code class="language-python">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

# Monthly airline passengers (classic dataset)
from statsmodels.datasets import get_rdataset
data = get_rdataset('AirPassengers', 'datasets')
ts = pd.Series(data.data['x'].values,
               index=pd.date_range('1949-01', periods=144, freq='M'))

# Train-test split
train, test = ts[:-24], ts[-24:]

# SARIMA(2,1,2)(1,1,1,12) for monthly seasonal data
model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)
print(fit.summary())

# Forecast
forecast = fit.forecast(steps=24)
mae = mean_absolute_error(test, forecast)
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(12, 5))
train.plot(label='Train')
test.plot(label='Actual Test')
forecast.plot(label='SARIMA Forecast', color='red')
plt.fill_between(forecast.index,
    fit.get_forecast(24).conf_int().iloc[:, 0],
    fit.get_forecast(24).conf_int().iloc[:, 1], alpha=0.2)
plt.legend()
plt.title('SARIMA Airline Passenger Forecast')
plt.show()
</code></pre>
`
  },
  {
    slug: 'timeseries-3-prophet',
    title: 'Prophet: Facebook\'s Forecasting Tool',
    description: 'Trend, seasonality components, holiday effects, and uncertainty intervals with Prophet',
    category: 'Time Series',
    content: `
<h3>Why Prophet?</h3>
<p>Prophet is designed for business time series — it handles missing data, outliers, holidays, and multiple seasonalities automatically. No need to specify ARIMA parameters.</p>

<pre><code class="language-python">from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prophet requires 'ds' (date) and 'y' (value) columns
df = pd.read_csv('sales.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# Model with multiple seasonalities
m = Prophet(
    changepoint_prior_scale=0.05,      # Trend flexibility
    seasonality_prior_scale=10,        # Seasonality strength
    seasonality_mode='multiplicative', # or 'additive'
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add custom holidays
from prophet.make_holidays import make_holidays_df
m.add_country_holidays(country_name='US')

# Custom seasonality
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

m.fit(df)

# Future dataframe
future = m.make_future_dataframe(periods=365, freq='D')
forecast = m.predict(future)

# Plot
m.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

m.plot_components(forecast)
plt.show()

# Evaluate on last year
train = df[df['ds'] < '2024-01-01']
test_df = df[df['ds'] >= '2024-01-01']
m2 = Prophet().fit(train)
future2 = m2.make_future_dataframe(periods=len(test_df))
fc2 = m2.predict(future2)
preds = fc2[fc2['ds'].isin(test_df['ds'])]['yhat']
from sklearn.metrics import mean_absolute_error
print(f"MAE: {mean_absolute_error(test_df['y'], preds):.2f}")
</code></pre>
`
  },
  {
    slug: 'timeseries-4-lstm',
    title: 'LSTM for Time Series Forecasting',
    description: 'Sliding window sequences, bidirectional LSTM, multi-step forecasting with Keras',
    category: 'Time Series',
    content: `
<h3>Why LSTM for Time Series?</h3>
<p>LSTM's gated architecture captures long-range temporal dependencies that ARIMA misses. Ideal for complex patterns in large datasets (energy, finance, IoT).</p>

<pre><code class="language-python">import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 10 * np.pi, 2000)
series = np.sin(t) + 0.5 * np.sin(3 * t) + np.random.normal(0, 0.1, 2000)

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X)[..., np.newaxis], np.array(y)

X, y = create_sequences(series_scaled, lookback=60)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Bidirectional LSTM
model = keras.Sequential([
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True), input_shape=(60, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='huber')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
          callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)

y_pred = scaler.inverse_transform(model.predict(X_test))
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
</code></pre>
`
  },
  {
    slug: 'timeseries-5-features',
    title: 'Time Series Feature Engineering',
    description: 'Lag features, rolling statistics, calendar features, Fourier transforms for ML models',
    category: 'Time Series',
    content: `
<h3>Feature Engineering for Tree Models</h3>
<p>Well-engineered features enable gradient boosting models to outperform deep learning on many time series tasks with less data and training time.</p>

<pre><code class="language-python">import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date').sort_index()
df.columns = ['sales']

def build_features(df, target='sales'):
    # Lag features
    for lag in [1, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df[target].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'roll_mean_{window}'] = df[target].shift(1).rolling(window).mean()
        df[f'roll_std_{window}'] = df[target].shift(1).rolling(window).std()
        df[f'roll_max_{window}'] = df[target].shift(1).rolling(window).max()

    # Calendar features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.astype(int)
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    # Fourier features for seasonality
    t = np.arange(len(df))
    for period, name in [(7, 'weekly'), (365, 'annual')]:
        for k in range(1, 4):
            df[f'sin_{name}_{k}'] = np.sin(2 * np.pi * k * t / period)
            df[f'cos_{name}_{k}'] = np.cos(2 * np.pi * k * t / period)

    return df.dropna()

df_feat = build_features(df.copy())
feature_cols = [c for c in df_feat.columns if c != 'sales']
X, y = df_feat[feature_cols], df_feat['sales']

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
</code></pre>
`
  },
  {
    slug: 'timeseries-6-project',
    title: 'Time Series Project: Energy Consumption Forecasting',
    description: 'End-to-end comparison of SARIMA, Prophet, and LSTM on energy consumption data',
    category: 'Time Series',
    content: `
<h3>Complete Forecasting Pipeline</h3>
<pre><code class="language-python">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('energy.csv', parse_dates=['date'], index_col='date')
df_daily = df.resample('D').sum()
df_daily.columns = ['energy_kwh']

train_size = int(len(df_daily) * 0.8)
train, test = df_daily.iloc[:train_size], df_daily.iloc[train_size:]

# SARIMA
sarima = SARIMAX(train['energy_kwh'], order=(1,1,1), seasonal_order=(1,1,1,7))
sarima_fit = sarima.fit(disp=False)
sarima_pred = sarima_fit.forecast(len(test))
sarima_mae = mean_absolute_error(test['energy_kwh'], sarima_pred)

# Prophet
prophet_train = train.reset_index().rename(columns={'date': 'ds', 'energy_kwh': 'y'})
prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
prophet_model.fit(prophet_train)
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)
prophet_pred = forecast['yhat'].tail(len(test)).values
prophet_mae = mean_absolute_error(test['energy_kwh'], prophet_pred)

print(f"SARIMA MAE: {sarima_mae:.2f} kWh")
print(f"Prophet MAE: {prophet_mae:.2f} kWh")

# Final comparison plot
plt.figure(figsize=(15, 6))
test['energy_kwh'].plot(label='Actual', color='black', linewidth=2)
pd.Series(sarima_pred.values, index=test.index).plot(label='SARIMA', alpha=0.8)
pd.Series(prophet_pred, index=test.index).plot(label='Prophet', alpha=0.8)
plt.legend(); plt.xlabel('Date'); plt.ylabel('Energy (kWh)')
plt.title('Energy Consumption Forecasting Comparison')
plt.tight_layout()
plt.show()
</code></pre>

<h4>Method Selection Guide</h4>
<table>
<tr><th>Scenario</th><th>Best Method</th></tr>
<tr><td>Small dataset, need explainability</td><td>SARIMA</td></tr>
<tr><td>Business forecasting with holidays</td><td>Prophet</td></tr>
<tr><td>Large dataset, complex patterns</td><td>LSTM</td></tr>
<tr><td>Many series, tabular-friendly</td><td>Feature engineering + GBM</td></tr>
</table>
`
  },
];
