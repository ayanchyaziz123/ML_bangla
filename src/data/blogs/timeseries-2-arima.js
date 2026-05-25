export const timeseries_2_arima = {
  slug: 'timeseries-2-arima',
  title: 'ARIMA: সময় সিরিজ পূর্বাভাস',
  description: 'AR (অটোরিগ্রেসিভ), MA (মুভিং অ্যাভারেজ), I (ইন্টিগ্রেটেড/ডিফারেন্সিং), ARIMA(p,d,q) প্যারামিটার, ACF/PACF দিয়ে order নির্বাচন, SARIMA সিজনাল ডেটার জন্য, এবং statsmodels Python।',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 14,
  content: `
    <h3>১. ARIMA কী এবং কেন?</h3>
    <p>
      <strong>ARIMA</strong> (AutoRegressive Integrated Moving Average) হলো সবচেয়ে জনপ্রিয় ক্লাসিক্যাল টাইম সিরিজ পূর্বাভাস মডেল। এটি তিনটি উপাদানের সমন্বয়:
    </p>
    <ul>
      <li><strong>AR (AutoRegressive):</strong> অতীতের নিজের মান ব্যবহার করে পূর্বাভাস</li>
      <li><strong>I (Integrated):</strong> Differencing দিয়ে non-stationary সিরিজকে stationary করা</li>
      <li><strong>MA (Moving Average):</strong> অতীতের error terms ব্যবহার করে পূর্বাভাস</li>
    </ul>
    <p>
      ARIMA একটি <strong>ARIMA(p, d, q)</strong> নোটেশন ব্যবহার করে, যেখানে p = AR order, d = differencing order, q = MA order।
    </p>

    <h3>২. AR মডেল (AutoRegressive)</h3>
    <p>
      AR মডেলে বর্তমান মান অতীতের নিজের মানের উপর নির্ভর করে। AR(p) মানে p টি lag ব্যবহার করা:
    </p>
    <p>
      y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ... + φₚ·y(t-p) + ε(t)
    </p>
    <p>
      উদাহরণ: AR(1) — আজকের তাপমাত্রা গতকালের তাপমাত্রার উপর নির্ভর করে। AR(2) — আজকের মান গত দুইদিনের মানের উপর নির্ভর করে।
    </p>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# সিন্থেটিক AR(2) সিরিজ তৈরি
np.random.seed(42)
n = 500
y = np.zeros(n)
phi1, phi2 = 0.6, 0.3
for t in range(2, n):
    y[t] = phi1 * y[t-1] + phi2 * y[t-2] + np.random.normal(0, 1)

ts = pd.Series(y, index=pd.date_range('2020-01-01', periods=n, freq='D'))

# AR মডেল fit করা
model = AutoReg(ts[:-30], lags=2)
result = model.fit()
print(result.summary())

# পূর্বাভাস
forecast = result.predict(start=len(ts)-30, end=len(ts)-1)
print("\nশেষ ৫টি পূর্বাভাস:")
print(forecast.tail())</code></pre>

    <h3>৩. MA মডেল (Moving Average)</h3>
    <p>
      MA মডেলে বর্তমান মান অতীতের error terms (residuals)-এর উপর নির্ভর করে। MA(q):
    </p>
    <p>
      y(t) = c + ε(t) + θ₁·ε(t-1) + θ₂·ε(t-2) + ... + θ_q·ε(t-q)
    </p>
    <p>
      <strong>AR vs MA পার্থক্য:</strong> AR মডেল অতীতের মান সরাসরি ব্যবহার করে, MA মডেল অতীতের "চমক" বা ত্রুটি ব্যবহার করে। ACF plot দিয়ে MA order এবং PACF plot দিয়ে AR order নির্ধারণ করা হয়।
    </p>

    <h4>ACF/PACF Pattern থেকে Order নির্ধারণ:</h4>
    <ul>
      <li><strong>AR(p):</strong> PACF lag p-এর পরে cutoff, ACF ধীরে কমে</li>
      <li><strong>MA(q):</strong> ACF lag q-এর পরে cutoff, PACF ধীরে কমে</li>
      <li><strong>ARMA(p,q):</strong> উভয়ই ধীরে কমে</li>
    </ul>

    <h3>৪. I উপাদান: Differencing</h3>
    <p>
      Non-stationary সিরিজকে stationary করতে differencing ব্যবহার করা হয়। <strong>d=1</strong> মানে একবার differencing:
    </p>
    <p>
      y'(t) = y(t) - y(t-1)
    </p>
    <p>
      <strong>d=2</strong> মানে দুইবার differencing (দ্বিতীয় পার্থক্য)। ADF টেস্ট দিয়ে কতবার differencing প্রয়োজন তা নির্ধারণ করা হয়।
    </p>
    <pre><code>from statsmodels.tsa.stattools import adfuller

# ধাপে ধাপে d নির্ধারণ
def find_d(series, max_d=3):
    d = 0
    current = series.copy()
    while d <= max_d:
        adf_result = adfuller(current.dropna())
        p_val = adf_result[1]
        print(f"d={d}: ADF p-value = {p_val:.4f} {'(Stationary)' if p_val < 0.05 else '(Non-stationary)'}")
        if p_val < 0.05:
            print(f"Differencing order d = {d}")
            return d
        current = current.diff()
        d += 1
    return d

# Non-stationary trend সিরিজ
trend_series = pd.Series(
    np.cumsum(np.random.randn(300)) + np.linspace(0, 50, 300),
    index=pd.date_range('2020-01-01', periods=300, freq='D')
)

optimal_d = find_d(trend_series)</code></pre>

    <h3>৫. সম্পূর্ণ ARIMA মডেল</h3>
    <p>
      ARIMA(p, d, q) তিনটি উপাদান একসাথে:
    </p>
    <pre><code>from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# বাস্তব ডেটা সিমুলেশন: মাসিক বিক্রয়
np.random.seed(42)
n = 120  # ১০ বছরের মাসিক ডেটা
dates = pd.date_range('2014-01-01', periods=n, freq='ME')
trend = np.linspace(1000, 2000, n)
seasonal = 150 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 50, n)
sales = trend + seasonal + noise

ts = pd.Series(sales, index=dates, name='sales')

# Train-Test split
train = ts[:-12]  # শেষ ১২ মাস test
test = ts[-12:]

# ARIMA(1,1,1) মডেল
model = ARIMA(train, order=(1, 1, 1))
result = model.fit()

print(result.summary())

# পূর্বাভাস
forecast = result.forecast(steps=12)
conf_int = result.get_forecast(steps=12).conf_int()

# মূল্যায়ন
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

print(f"\nমূল্যায়ন:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Train', color='steelblue')
plt.plot(test.index, test, label='Test (Actual)', color='green')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 alpha=0.2, color='red', label='95% Confidence Interval')
plt.legend()
plt.title('ARIMA(1,1,1) পূর্বাভাস')
plt.show()</code></pre>

    <h3>৬. Auto ARIMA দিয়ে সেরা Parameters খোঁজা</h3>
    <p>
      Manual ACF/PACF analysis ছাড়াও <code>pmdarima</code> লাইব্রেরির <code>auto_arima</code> স্বয়ংক্রিয়ভাবে সেরা (p, d, q) খুঁজে দেয়:
    </p>
    <pre><code>import pmdarima as pm

# Auto ARIMA: সেরা parameters স্বয়ংক্রিয়ভাবে খোঁজা
auto_model = pm.auto_arima(
    train,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,          # d স্বয়ংক্রিয়ভাবে নির্ধারণ
    information_criterion='aic',
    stepwise=True,   # দ্রুত search
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

print("\nসেরা মডেল:")
print(auto_model.summary())
print(f"\nসেরা Order: ARIMA{auto_model.order}")

# AIC (Akaike Information Criterion) তুলনা
print(f"AIC: {auto_model.aic():.4f}")
print(f"BIC: {auto_model.bic():.4f}")</code></pre>

    <h3>৭. SARIMA: Seasonal ARIMA</h3>
    <p>
      সিজনাল ডেটার জন্য ARIMA এক্সটেন্ড করে <strong>SARIMA</strong> তৈরি করা হয়েছে। SARIMA(p,d,q)(P,D,Q)[m] নোটেশনে:
    </p>
    <ul>
      <li><strong>p, d, q:</strong> Non-seasonal AR, differencing, MA order</li>
      <li><strong>P, D, Q:</strong> Seasonal AR, differencing, MA order</li>
      <li><strong>m:</strong> Seasonal period (মাসিক = 12, সাপ্তাহিক = 52)</li>
    </ul>
    <pre><code>from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(1,1,1)(1,1,1)[12] — মাসিক সিজনাল ডেটার জন্য
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(disp=False)

print(sarima_result.summary())

# Seasonal differencing দিয়ে stationarity চেক
ts_seasonal_diff = ts.diff(12).dropna()  # বার্ষিক seasonal differencing
ts_both_diff = ts_seasonal_diff.diff(1).dropna()  # + regular differencing

from statsmodels.tsa.stattools import adfuller
adf_both = adfuller(ts_both_diff)
print(f"\nDouble Differenced ADF p-value: {adf_both[1]:.4f}")

# Auto SARIMA
seasonal_auto = pm.auto_arima(
    train,
    seasonal=True,
    m=12,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None, D=None,
    information_criterion='aic',
    stepwise=True,
    trace=True,
    suppress_warnings=True
)
print(f"\nSARIMA Order: {seasonal_auto.order}")
print(f"Seasonal Order: {seasonal_auto.seasonal_order}")</code></pre>

    <h3>৮. Model Diagnostics</h3>
    <p>
      মডেল fit করার পর residuals পরীক্ষা করা অত্যন্ত গুরুত্বপূর্ণ। ভালো মডেলের residuals হবে white noise (কোনো প্যাটার্ন থাকবে না):
    </p>
    <pre><code># Residual analysis
residuals = sarima_result.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual plot
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals')
axes[0, 0].axhline(y=0, color='r', linestyle='--')

# Histogram
axes[0, 1].hist(residuals, bins=30, density=True, color='steelblue', alpha=0.7)
axes[0, 1].set_title('Residual Distribution')

# ACF of residuals (should be near zero)
plot_acf(residuals, lags=40, ax=axes[1, 0], title='ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Ljung-Box test: residuals কি white noise?
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box Test:")
print(lb_test)
print("p > 0.05 হলে residuals white noise (ভালো মডেল)")</code></pre>

    <h3>৯. Walk-Forward Validation</h3>
    <p>
      টাইম সিরিজে সাধারণ cross-validation কাজ করে না কারণ ভবিষ্যতের ডেটা দিয়ে past predict করা যায় না। <strong>Walk-forward validation</strong> এই সমস্যা সমাধান করে:
    </p>
    <pre><code># Walk-forward (Expanding Window) Validation
def walk_forward_arima(series, order, n_test=24):
    history = list(series[:-n_test])
    test = list(series[-n_test:])
    predictions = []

    for t in range(n_test):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)[0]
        predictions.append(pred)
        history.append(test[t])  # নতুন সত্যিকার মান যোগ করা

    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    print(f"Walk-Forward RMSE: {rmse:.2f}")
    print(f"Walk-Forward MAE:  {mae:.2f}")
    return predictions

predictions = walk_forward_arima(ts, order=(1, 1, 1), n_test=24)</code></pre>

    <h3>১০. সারসংক্ষেপ</h3>
    <p>ARIMA মডেলের সাথে কাজ করার ধাপগুলো:</p>
    <ol>
      <li><strong>Stationarity check:</strong> ADF টেস্ট → d নির্ধারণ</li>
      <li><strong>ACF/PACF analysis:</strong> p ও q নির্ধারণ</li>
      <li><strong>Model fitting:</strong> ARIMA(p,d,q) fit করা</li>
      <li><strong>Diagnostic check:</strong> Residual analysis (Ljung-Box)</li>
      <li><strong>Forecast:</strong> পূর্বাভাস তৈরি করা</li>
      <li><strong>Evaluation:</strong> RMSE, MAE, MAPE দিয়ে মূল্যায়ন</li>
    </ol>
    <p>
      পরের পর্বে আমরা Facebook Prophet — আরও সহজ কিন্তু শক্তিশালী একটি টুল শিখব।
    </p>
  `
};
