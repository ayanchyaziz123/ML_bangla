export const timeseries_4_lstm = {
  slug: 'timeseries-4-lstm',
  title: 'LSTM দিয়ে টাইম সিরিজ পূর্বাভাস',
  description: 'দীর্ঘমেয়াদী নির্ভরতা ধরতে LSTM নেটওয়ার্ক ব্যবহার করে টাইম সিরিজ ভবিষ্যদ্বাণী শিখুন',
  date: 'মে ২০২৫',
  category: 'টাইম সিরিজ',
  readTime: 14,
  content: `
<h3>কেন LSTM টাইম সিরিজের জন্য?</h3>
<p>ঐতিহ্যগত পদ্ধতি (ARIMA) স্বল্পমেয়াদী নির্ভরতা ধরতে পারে। কিন্তু বিদ্যুৎ খরচ বা স্টক মূল্য যেখানে সাপ্তাহিক বা মাসিক প্যাটার্ন আছে, LSTM অনেক কার্যকর।</p>

<h3>স্লাইডিং উইন্ডো পদ্ধতি</h3>

<pre><code class="language-python">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# শক্তি খরচ ডেটা লোড
df = pd.read_csv('energy_consumption.csv', parse_dates=['timestamp'],
                 index_col='timestamp')
series = df['consumption'].values.reshape(-1, 1)

# স্কেলিং (LSTM-এর জন্য 0-1 রেঞ্জ ভালো)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

def create_sequences(data, lookback=60):
    """স্লাইডিং উইন্ডো দিয়ে X, y তৈরি"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

lookback = 60  # ৬০ ধাপ পেছনে দেখব
X, y = create_sequences(series_scaled, lookback)

# LSTM-এর জন্য আকার: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ট্রেন-টেস্ট বিভাজন (সময়-ক্রমানুসারে)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"ট্রেন আকার: {X_train.shape}")
print(f"টেস্ট আকার: {X_test.shape}")
</code></pre>

<h3>LSTM মডেল তৈরি</h3>

<pre><code class="language-python">def build_lstm(lookback, units=50):
    model = keras.Sequential([
        keras.layers.LSTM(units, return_sequences=True,
                          input_shape=(lookback, 1)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(25),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm(lookback)
model.summary()

# কলব্যাক
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

# ট্রেনিং
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
</code></pre>

<h4>ট্রেনিং পর্যবেক্ষণ</h4>

<pre><code class="language-python">plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='ট্রেন লস')
plt.plot(history.history['val_loss'], label='ভ্যাল লস')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('LSTM ট্রেনিং ইতিহাস')
plt.show()
</code></pre>

<h3>পূর্বাভাস ও মূল্যায়ন</h3>

<pre><code class="language-python"># পূর্বাভাস
y_pred_scaled = model.predict(X_test)

# স্কেল ফিরিয়ে আনা
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# মেট্রিক
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# ভিজুয়ালাইজেশন
plt.figure(figsize=(15, 5))
plt.plot(y_actual[:200], label='প্রকৃত', color='blue')
plt.plot(y_pred[:200], label='পূর্বাভাস', color='red', alpha=0.8)
plt.xlabel('সময়')
plt.ylabel('শক্তি খরচ (kWh)')
plt.legend()
plt.title('LSTM টাইম সিরিজ পূর্বাভাস')
plt.show()
</code></pre>

<h3>মাল্টি-স্টেপ পূর্বাভাস</h3>

<pre><code class="language-python">def multi_step_forecast(model, last_sequence, n_steps, scaler):
    """ভবিষ্যতের একাধিক ধাপ পূর্বাভাস"""
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
        predictions.append(pred[0, 0])
        # উইন্ডো আপডেট
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred[0, 0]

    # স্কেল ফিরিয়ে আনা
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

# শেষ উইন্ডো দিয়ে ৩০ ধাপ ভবিষ্যদ্বাণী
last_60 = series_scaled[-lookback:, 0]
future_30 = multi_step_forecast(model, last_60, 30, scaler)
print("পরবর্তী ৩০ দিনের পূর্বাভাস:")
for i, val in enumerate(future_30[:5]):
    print(f"  দিন {i+1}: {val[0]:.2f} kWh")
</code></pre>

<h4>ARIMA vs LSTM তুলনা</h4>
<table>
<tr><th>দিক</th><th>ARIMA</th><th>LSTM</th></tr>
<tr><td>ডেটার আকার</td><td>ছোট (~100s)</td><td>বড় (1000s+)</td></tr>
<tr><td>স্থিতিশীলতা প্রয়োজন</td><td>হ্যাঁ</td><td>না</td></tr>
<tr><td>ব্যাখ্যাযোগ্যতা</td><td>উচ্চ</td><td>কম</td></tr>
<tr><td>দীর্ঘমেয়াদী নির্ভরতা</td><td>সীমিত</td><td>চমৎকার</td></tr>
<tr><td>ট্রেনিং সময়</td><td>দ্রুত</td><td>ধীর (GPU লাগে)</td></tr>
</table>
`
};
