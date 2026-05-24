export const nn_8_rnn_lstm = {
  title: "RNN ও LSTM: Sequential Data ও Time Series",
  description: "Recurrent Neural Network-এর hidden state, vanishing gradient সমস্যা, LSTM-এর cell state ও gate mechanism এবং Keras দিয়ে time series prediction বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "nn-rnn-lstm",
  content: `
    <h3>১. Sequential Data ও RNN-এর দরকার</h3>
    <p>Standard neural network প্রতিটি input independently process করে — আগের input মনে রাখে না। কিন্তু text, time series, speech — এ ধরনের data-এ <strong>ক্রম (sequence) গুরুত্বপূর্ণ</strong>।</p>
    <pre><code># সমস্যা: "আজকের তাপমাত্রা কত?" বোঝার জন্য আগের কথা জানা দরকার
# বা: [10, 20, 30, 40, ?] — পরের সংখ্যা predict করতে আগেরগুলো দরকার

# RNN solution: hidden state h_t আগের time step-এর information বহন করে
#
# h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
# y_t = W_y * h_t + b_y
#
# x_t  = time t-এর input
# h_t  = time t-এর hidden state ("memory")
# h_{t-1} = আগের hidden state</code></pre>
    <table>
      <thead><tr><th>Data Type</th><th>উদাহরণ</th><th>Sequence Length</th></tr></thead>
      <tbody>
        <tr><td>Time Series</td><td>Stock price, weather, sensor data</td><td>দিন/ঘণ্টা/মিনিট</td></tr>
        <tr><td>Text (NLP)</td><td>Sentiment analysis, translation</td><td>Word/character sequence</td></tr>
        <tr><td>Audio</td><td>Speech recognition</td><td>Time frames</td></tr>
        <tr><td>Video</td><td>Action recognition</td><td>Frame sequence</td></tr>
      </tbody>
    </table>

    <h3>২. Vanishing Gradient — RNN-এর সমস্যা</h3>
    <pre><code># দীর্ঘ sequence-এ backpropagation করলে gradient ছোট হতে হতে শূন্যের কাছে যায়
# ফলে দূরের time step-এর information শেখা যায় না

# উদাহরণ:
# "The cat that sat on the mat was ___" — 'was' কী হবে জানতে 'cat' মনে রাখতে হবে
# কিন্তু RNN t=1 (cat) থেকে t=8 (was) পর্যন্ত gradient ক্রমশ ছোট হয়

# ∂L/∂W = ∂L/∂h_T * ∂h_T/∂h_{T-1} * ... * ∂h_2/∂h_1 * ∂h_1/∂W
# প্রতিটি ∂h_t/∂h_{t-1} < 1 হলে গুণফল → 0 (vanishing gradient)
# প্রতিটি > 1 হলে গুণফল → ∞ (exploding gradient)

# Solution: LSTM — designed to remember long-term dependencies</code></pre>

    <h3>৩. LSTM — Long Short-Term Memory</h3>
    <p>LSTM-এ দুটো state আছে: <strong>cell state (c_t)</strong> = দীর্ঘমেয়াদী মেমরি, <strong>hidden state (h_t)</strong> = স্বল্পমেয়াদী মেমরি। তিনটি gate এই flow নিয়ন্ত্রণ করে।</p>
    <pre><code># ১. Forget Gate: কোন পুরনো information ভুলবো?
# f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   → 0=ভুলে যাও, 1=মনে রাখো

# ২. Input Gate: নতুন কী তথ্য যোগ করবো?
# i_t   = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
# c̃_t  = tanh(W_c * [h_{t-1}, x_t] + b_c)    → candidate values
# c_t   = f_t * c_{t-1} + i_t * c̃_t           → cell state update

# ৩. Output Gate: cell state থেকে কী বের করবো?
# o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
# h_t = o_t * tanh(c_t)                        → hidden state</code></pre>
    <table>
      <thead><tr><th>Model</th><th>Gates</th><th>States</th><th>Speed</th><th>Long-term Memory</th></tr></thead>
      <tbody>
        <tr><td>RNN</td><td>없음</td><td>h_t</td><td>빠름</td><td>দুর্বল</td></tr>
        <tr><td>LSTM</td><td>3 (forget, input, output)</td><td>h_t, c_t</td><td>ধীর</td><td>강함</td></tr>
        <tr><td>GRU</td><td>2 (reset, update)</td><td>h_t</td><td>মাঝারি</td><td>ভালো</td></tr>
      </tbody>
    </table>

    <h3>৪. Keras দিয়ে Time Series Prediction</h3>
    <pre><code">import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Synthetic sine wave time series তৈরি
t     = np.linspace(0, 100, 1000)
series = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)

# Sliding window দিয়ে sequences তৈরি
def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 20
X, y = create_sequences(series, seq_len)
X    = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")

# LSTM মডেল
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30, batch_size=32,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1,
)

# Prediction
y_pred = model.predict(X_test).flatten()
mse    = np.mean((y_test - y_pred)**2)
print(f"Test MSE: {mse:.4f}")

plt.figure(figsize=(12, 4))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted', ls='--')
plt.title('LSTM Time Series Prediction'); plt.legend(); plt.show()</code></pre>

    <h3>৫. GRU — Simplified LSTM</h3>
    <pre><code">from tensorflow import keras

# GRU: LSTM-এর মতোই কিন্তু ২টি gate (reset + update) → দ্রুত, প্রায় সমান কার্যকর

# Reset Gate: কতটুকু পুরনো hidden state ভুলবো
# r_t = sigmoid(W_r * [h_{t-1}, x_t])

# Update Gate: পুরনো ও নতুন কতটুকু মিলাবো
# z_t = sigmoid(W_z * [h_{t-1}, x_t])
# h̃_t = tanh(W * [r_t * h_{t-1}, x_t])
# h_t = (1-z_t)*h_{t-1} + z_t*h̃_t

gru_model = keras.Sequential([
    keras.layers.GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
    keras.layers.GRU(32),
    keras.layers.Dense(1),
])
gru_model.compile(optimizer='adam', loss='mse')

# Simple RNN
rnn_model = keras.Sequential([
    keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(seq_len, 1)),
    keras.layers.SimpleRNN(32),
    keras.layers.Dense(1),
])
rnn_model.compile(optimizer='adam', loss='mse')

print("LSTM params:", model.count_params())
print("GRU params: ", gru_model.count_params())
print("RNN params: ", rnn_model.count_params())</code></pre>

    <h3>৬. Sentiment Analysis — Text Classification</h3>
    <pre><code">from tensorflow import keras
import numpy as np

# IMDB movie review sentiment (built-in dataset)
max_words = 10000
max_len   = 200

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=max_words)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test  = keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=max_len)

model_text = keras.Sequential([
    keras.layers.Embedding(max_words, 64, input_length=max_len),  # word → vector
    keras.layers.LSTM(64, dropout=0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_text.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_text.fit(
    X_train, y_train,
    epochs=5, batch_size=128,
    validation_split=0.2,
    verbose=1,
)
loss, acc = model_text.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>RNN</td><td>Sequential data-র জন্য — hidden state দিয়ে আগের information বহন</td></tr>
        <tr><td>Vanishing Gradient</td><td>দীর্ঘ sequence-এ RNN ব্যর্থ — LSTM/GRU দিয়ে সমাধান</td></tr>
        <tr><td>LSTM</td><td>Cell state + 3 gates — long-term dependency শেখার জন্য সেরা</td></tr>
        <tr><td>GRU</td><td>2 gates — LSTM-এর simplified version, দ্রুত, প্রায় সমান কার্যকর</td></tr>
        <tr><td>return_sequences=True</td><td>Stacked LSTM-এ প্রথম layer-এ ব্যবহার করো</td></tr>
        <tr><td>Embedding layer</td><td>Text classification-এ word → dense vector mapping</td></tr>
      </tbody>
    </table>
  `,
};
