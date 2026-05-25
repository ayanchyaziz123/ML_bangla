export const timeseries_3_prophet = {
  slug: 'timeseries-3-prophet',
  title: 'Prophet: Facebook-এর পূর্বাভাস টুল',
  description: 'Prophet মডেলের ট্রেন্ড + সিজনালিটি + হলিডে উপাদান, additive vs multiplicative seasonality, changepoints, holiday effects, uncertainty intervals, এবং Python prophet লাইব্রেরি।',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 13,
  content: `
    <h3>১. Prophet কী এবং কেন ব্যবহার করবেন?</h3>
    <p>
      <strong>Facebook Prophet</strong> (২০১৭) হলো Meta (Facebook)-এর তৈরি একটি ওপেন-সোর্স টাইম সিরিজ পূর্বাভাস লাইব্রেরি। এটি ব্যবসায়িক ডেটার জন্য বিশেষভাবে ডিজাইন করা হয়েছে — যেখানে ঋতু পরিবর্তন, ছুটির দিন, এবং হঠাৎ ট্রেন্ড পরিবর্তন থাকে।
    </p>
    <p>Prophet কেন জনপ্রিয়:</p>
    <ul>
      <li><strong>সহজ ব্যবহার:</strong> মাত্র কয়েক লাইনে পূর্বাভাস তৈরি করা যায়</li>
      <li><strong>Robust:</strong> missing data ও outliers সহজে সামলায়</li>
      <li><strong>Human-interpretable:</strong> উপাদানগুলো আলাদাভাবে দেখা যায়</li>
      <li><strong>Flexible seasonality:</strong> একাধিক seasonal period সমর্থন করে</li>
      <li><strong>Holiday effects:</strong> ছুটির দিনের প্রভাব সহজে যোগ করা যায়</li>
    </ul>

    <h3>২. Prophet-এর গণিতগত কাঠামো</h3>
    <p>Prophet একটি decomposable model ব্যবহার করে:</p>
    <p>
      y(t) = g(t) + s(t) + h(t) + ε(t)
    </p>
    <ul>
      <li><strong>g(t):</strong> Trend function (দীর্ঘমেয়াদী বৃদ্ধি বা হ্রাস)</li>
      <li><strong>s(t):</strong> Seasonality (পর্যায়ক্রমিক পরিবর্তন)</li>
      <li><strong>h(t):</strong> Holiday effects (ছুটির দিনের প্রভাব)</li>
      <li><strong>ε(t):</strong> Error term (অপ্রত্যাশিত পরিবর্তন)</li>
    </ul>

    <h4>Trend মডেল</h4>
    <p>
      Prophet দুটি trend মডেল সমর্থন করে:
    </p>
    <ul>
      <li><strong>Linear trend:</strong> ডেটা রৈখিকভাবে বৃদ্ধি পায় (growth='linear')</li>
      <li><strong>Logistic growth:</strong> ডেটা একটি saturation point-এ পৌঁছায়, যেমন ওয়েবসাইট ব্যবহারকারী (growth='logistic')</li>
    </ul>

    <h3>৩. প্রথম Prophet মডেল</h3>
    <pre><code>from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prophet-এর জন্য ডেটা format: 'ds' (date) এবং 'y' (value)
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=730, freq='D')
trend = np.linspace(100, 300, 730)
yearly_seasonal = 50 * np.sin(2 * np.pi * np.arange(730) / 365.25)
weekly_seasonal = 20 * np.sin(2 * np.pi * np.arange(730) / 7)
noise = np.random.normal(0, 10, 730)
y = trend + yearly_seasonal + weekly_seasonal + noise

df = pd.DataFrame({'ds': dates, 'y': y})

print("Prophet-এর জন্য ডেটা:")
print(df.head())
print(f"\nআকার: {df.shape}")

# মৌলিক Prophet মডেল
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'
)

model.fit(df)

# ভবিষ্যতের জন্য dataframe তৈরি
future = model.make_future_dataframe(periods=365, freq='D')
print(f"\nFuture dataframe: {future.shape}")

# পূর্বাভাস
forecast = model.predict(future)
print("\nপূর্বাভাসের কলামসমূহ:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))</code></pre>

    <h3>৪. Prophet Visualization</h3>
    <pre><code># মূল Prophet plot
fig1 = model.plot(forecast)
plt.title('Prophet পূর্বাভাস', fontsize=14)
plt.xlabel('তারিখ')
plt.ylabel('মান')
plt.show()

# উপাদান plot: trend, seasonality আলাদাভাবে দেখা
fig2 = model.plot_components(forecast)
plt.suptitle('Prophet উপাদান বিশ্লেষণ', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Custom visualization
plt.figure(figsize=(14, 6))
# Historical data
plt.plot(df['ds'], df['y'], 'o', markersize=2, alpha=0.5, label='Actual', color='steelblue')
# Forecast
future_only = forecast[forecast['ds'] > df['ds'].max()]
plt.plot(future_only['ds'], future_only['yhat'], color='red', linewidth=2, label='Forecast')
plt.fill_between(future_only['ds'],
                 future_only['yhat_lower'],
                 future_only['yhat_upper'],
                 alpha=0.2, color='red', label='Uncertainty Interval')
plt.legend()
plt.title('Prophet: Historical + Forecast')
plt.show()</code></pre>

    <h3>৫. Changepoints: ট্রেন্ড পরিবর্তনের পয়েন্ট</h3>
    <p>
      <strong>Changepoints</strong> হলো সেই মুহূর্তগুলো যেখানে ট্রেন্ডের দিক বা গতি হঠাৎ পরিবর্তিত হয়। উদাহরণ: কোভিড মহামারির শুরুতে বিমান যাত্রীর সংখ্যা হঠাৎ কমে যাওয়া।
    </p>
    <pre><code>from prophet.plot import add_changepoints_to_plot

# Changepoint sensitivity নিয়ন্ত্রণ
model_cp = Prophet(
    changepoint_prior_scale=0.05,  # ছোট = কম নমনীয় trend
    changepoint_range=0.8,          # ডেটার কতটা অংশে changepoints খুঁজবে
    n_changepoints=25               # সর্বোচ্চ changepoints সংখ্যা
)
model_cp.fit(df)
forecast_cp = model_cp.predict(future)

# Changepoints visualize করা
fig = model_cp.plot(forecast_cp)
add_changepoints_to_plot(fig.gca(), model_cp, forecast_cp)
plt.title('Prophet: Changepoints সহ পূর্বাভাস')
plt.show()

# Detected changepoints দেখা
print("Detected Changepoints:")
print(model_cp.changepoints)

# Manual changepoints নির্দিষ্ট করা
model_manual = Prophet(
    changepoints=['2021-03-01', '2022-06-15'],  # নির্দিষ্ট তারিখে changepoint
    changepoint_prior_scale=0.1
)
model_manual.fit(df)

# Prior scale-এর প্রভাব
for scale in [0.001, 0.01, 0.1, 0.5]:
    m = Prophet(changepoint_prior_scale=scale)
    m.fit(df)
    f = m.predict(future)
    print(f"scale={scale}: Trend variance = {f['trend'].std():.2f}")</code></pre>

    <h3>৬. Seasonality: Additive vs Multiplicative</h3>
    <p>
      <strong>Additive seasonality:</strong> সিজনাল variation সময়ের সাথে একই থাকে। উদাহরণ: প্রতি বছর শীতে বিক্রয় ৫০০ টাকা বাড়ে।
    </p>
    <p>
      <strong>Multiplicative seasonality:</strong> সিজনাল variation ট্রেন্ডের সাথে বাড়ে। উদাহরণ: প্রতি বছর শীতে বিক্রয় ১০% বাড়ে।
    </p>
    <pre><code># Multiplicative seasonality
model_mult = Prophet(seasonality_mode='multiplicative')
model_mult.fit(df)
forecast_mult = model_mult.predict(future)

# Custom seasonality যোগ করা
model_custom = Prophet(yearly_seasonality=False)

# মাসিক seasonality (Fourier series দিয়ে)
model_custom.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# সাপ্তাহিক seasonality (বেশি flexible)
model_custom.add_seasonality(
    name='weekly',
    period=7,
    fourier_order=3,
    prior_scale=0.1  # regularization
)

# বার্ষিক seasonality
model_custom.add_seasonality(
    name='yearly',
    period=365.25,
    fourier_order=10
)

model_custom.fit(df)
forecast_custom = model_custom.predict(future)

# Seasonality plot
fig = model_custom.plot_components(forecast_custom)
plt.show()</code></pre>

    <h3>৭. Holiday Effects: ছুটির দিনের প্রভাব</h3>
    <p>
      ব্যবসায়িক ডেটায় ঈদ, পূজা, জাতীয় দিবস-এর মতো ছুটির দিনে বিক্রয় আলাদাভাবে আচরণ করে। Prophet এই প্রভাবকে explicitly মডেল করতে পারে।
    </p>
    <pre><code># বাংলাদেশী ছুটির দিন
bangladesh_holidays = pd.DataFrame({
    'holiday': 'eid_ul_fitr',
    'ds': pd.to_datetime(['2020-05-24', '2021-05-13', '2022-05-02',
                          '2023-04-21', '2024-04-10']),
    'lower_window': -3,   # ৩ দিন আগে থেকে প্রভাব
    'upper_window': 2,    # ২ দিন পরে পর্যন্ত প্রভাব
})

eid_adha = pd.DataFrame({
    'holiday': 'eid_ul_adha',
    'ds': pd.to_datetime(['2020-07-31', '2021-07-20', '2022-07-09',
                          '2023-06-28', '2024-06-17']),
    'lower_window': -2,
    'upper_window': 2,
})

# সব holidays একত্রিত করা
all_holidays = pd.concat([bangladesh_holidays, eid_adha])

# Prophet মডেলে holidays যোগ করা
model_holiday = Prophet(
    holidays=all_holidays,
    holidays_prior_scale=10.0  # holiday effect কতটা flexible
)
model_holiday.fit(df)
forecast_holiday = model_holiday.predict(future)

# Holiday components দেখা
print("Holiday Effects:")
holiday_effects = forecast_holiday[['ds', 'eid_ul_fitr', 'eid_ul_adha']].dropna()
print(holiday_effects[holiday_effects['eid_ul_fitr'] != 0].head())

# Built-in country holidays ব্যবহার
model_builtin = Prophet()
model_builtin.add_country_holidays(country_name='BD')  # Bangladesh
print("\nBuilt-in holidays:")
print(model_builtin.train_holiday_names)</code></pre>

    <h3>৮. Uncertainty Intervals</h3>
    <p>
      Prophet দুটি ধরনের uncertainty measure করে:
    </p>
    <ul>
      <li><strong>Trend uncertainty:</strong> Changepoints কতটা পরিবর্তিত হতে পারে</li>
      <li><strong>Observation noise:</strong> ডেটার স্বাভাবিক variance</li>
    </ul>
    <pre><code># Uncertainty interval width নিয়ন্ত্রণ
model_unc = Prophet(
    interval_width=0.95,           # 95% confidence interval
    mcmc_samples=300,              # Bayesian sampling (ধীরে কিন্তু সঠিক)
    uncertainty_samples=1000
)
model_unc.fit(df)
forecast_unc = model_unc.predict(future)

# Uncertainty visualization
plt.figure(figsize=(14, 6))
plt.plot(df['ds'], df['y'], 'k.', markersize=3, alpha=0.3, label='Data')
plt.plot(forecast_unc['ds'], forecast_unc['yhat'], 'b-', linewidth=2, label='Forecast')
plt.fill_between(forecast_unc['ds'],
                 forecast_unc['yhat_lower'],
                 forecast_unc['yhat_upper'],
                 alpha=0.3, color='blue', label='95% Interval')
plt.legend()
plt.title('Prophet: Uncertainty Intervals')
plt.show()

# Trend uncertainty আলাদাভাবে
plt.figure(figsize=(14, 4))
plt.plot(forecast_unc['ds'], forecast_unc['trend'], 'r-', linewidth=2)
plt.fill_between(forecast_unc['ds'],
                 forecast_unc['trend_lower'],
                 forecast_unc['trend_upper'],
                 alpha=0.3, color='red')
plt.title('Trend Uncertainty')
plt.show()</code></pre>

    <h3>৯. Regressor যোগ করা (External Variables)</h3>
    <p>
      Prophet-এ external predictor variables (regressors) যোগ করা যায়। উদাহরণ: তাপমাত্রা দিয়ে বিদ্যুৎ খরচ predict করা।
    </p>
    <pre><code># External regressor সহ Prophet
# উদাহরণ: temperature দিয়ে বিদ্যুৎ ব্যবহার predict
np.random.seed(42)
temperature = 25 + 10 * np.sin(2 * np.pi * np.arange(730) / 365) + np.random.normal(0, 2, 730)
electricity = df['y'] + 2 * temperature + np.random.normal(0, 5, 730)

df_reg = pd.DataFrame({
    'ds': dates,
    'y': electricity,
    'temperature': temperature
})

model_reg = Prophet()
model_reg.add_regressor('temperature', prior_scale=10, mode='additive')
model_reg.fit(df_reg)

# Future dataframe-এ regressor মান দিতে হবে
future_reg = model_reg.make_future_dataframe(periods=90, freq='D')
# Test-এর জন্য temperature forecast প্রয়োজন
future_reg['temperature'] = 25 + 10 * np.sin(
    2 * np.pi * np.arange(len(future_reg)) / 365
)

forecast_reg = model_reg.predict(future_reg)
print("Regressor সহ forecast:")
print(forecast_reg[['ds', 'yhat', 'temperature']].tail(5))</code></pre>

    <h3>১০. Cross-Validation ও Performance Metrics</h3>
    <pre><code">from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# Prophet-এর built-in cross-validation
df_cv = cross_validation(
    model,
    initial='365 days',    # প্রথম training period
    period='90 days',      # কতদিন পরপর re-train
    horizon='90 days',     # কত দিনের পূর্বাভাস evaluate করবে
    parallel="processes"
)

print("Cross-Validation ফলাফল:")
print(df_cv.head(10))

# Performance metrics
df_p = performance_metrics(df_cv)
print("\nPerformance Metrics:")
print(df_p[['horizon', 'mse', 'rmse', 'mae', 'mape', 'coverage']].head(10))

# MAPE plot
fig = plot_cross_validation_metric(df_cv, metric='mape')
plt.title('MAPE vs Forecast Horizon')
plt.show()</code></pre>

    <h3>১১. সারসংক্ষেপ</h3>
    <p>Prophet-এর মূল সুবিধা ও সীমাবদ্ধতা:</p>
    <ul>
      <li><strong>সুবিধা:</strong> সহজ ব্যবহার, interpretable components, holiday support, missing data robust</li>
      <li><strong>সীমাবদ্ধতা:</strong> শুধু univariate, বহু-পদক্ষেপের পূর্বাভাসে সীমিত, non-business data-এ কম কার্যকর</li>
    </ul>
    <p>
      পরবর্তী পর্বে আমরা LSTM — একটি deep learning approach শিখব যা জটিল non-linear প্যাটার্ন ধরতে পারে।
    </p>
  `
};
