export const nn_8_rnn_lstm = {
  title: "RNN ও LSTM: Sequential Data ও Time Series",
  description: "Recurrent Neural Network-এর hidden state, vanishing gradient সমস্যা, LSTM-এর cell state ও gate mechanism এবং PyTorch দিয়ে time series prediction বাংলায়।",
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

    <h3>৪. PyTorch দিয়ে Time Series Prediction</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Synthetic sine wave time series তৈরি
t      = np.linspace(0, 100, 1000)
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
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # (samples, timesteps, features)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")

# LSTM মডেল — explicit nn.Module class
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)           # পুরো sequence পরের LSTM-এ পাঠানো হয় (return_sequences=True-এর মতো)
        _, (h_n, _) = self.lstm2(x)    # এখন শুধু শেষ hidden state দরকার
        x = self.relu(self.fc1(h_n[-1]))
        return self.fc2(x)

model = LSTMForecaster()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

n_val = int(len(X_train) * 0.2)
X_val, y_val = X_train[-n_val:], y_train[-n_val:]
best_val_loss, patience, patience_counter = float('inf'), 5, 0

for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    if val_loss &lt; best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:    # EarlyStopping(patience=5)
            break

model.load_state_dict(best_state)           # restore_best_weights=True

# Prediction
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze(-1).numpy()
mse = np.mean((y_test.numpy().flatten() - y_pred)**2)
print(f"Test MSE: {mse:.4f}")

plt.figure(figsize=(12, 4))
plt.plot(y_test.numpy().flatten()[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted', ls='--')
plt.title('LSTM Time Series Prediction'); plt.legend(); plt.show()</code></pre>

    <h3>৫. GRU — Simplified LSTM</h3>
    <pre><code>import torch.nn as nn

# GRU: LSTM-এর মতোই কিন্তু ২টি gate (reset + update) → দ্রুত, প্রায় সমান কার্যকর

# Reset Gate: কতটুকু পুরনো hidden state ভুলবো
# r_t = sigmoid(W_r * [h_{t-1}, x_t])

# Update Gate: পুরনো ও নতুন কতটুকু মিলাবো
# z_t = sigmoid(W_z * [h_{t-1}, x_t])
# h̃_t = tanh(W * [r_t * h_{t-1}, x_t])
# h_t = (1-z_t)*h_{t-1} + z_t*h̃_t

class GRUForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        _, h_n = self.gru2(x)
        return self.fc(h_n[-1])

gru_model = GRUForecaster()

# Simple RNN
class SimpleRNNForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=1, hidden_size=64, batch_first=True)
        self.rnn2 = nn.RNN(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        _, h_n = self.rnn2(x)
        return self.fc(h_n[-1])

rnn_model = SimpleRNNForecaster()

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("LSTM params:", count_params(model))
print("GRU params: ", count_params(gru_model))
print("RNN params: ", count_params(rnn_model))</code></pre>

    <h3>৬. Sentiment Analysis — Text Classification</h3>
    <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# IMDB movie review sentiment
tokenizer = get_tokenizer('basic_english')
max_len, vocab_size = 200, 10000

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(IMDB(split='train')),
                                   max_tokens=vocab_size, specials=['&lt;pad&gt;', '&lt;unk&gt;'])
vocab.set_default_index(vocab['&lt;unk&gt;'])

def encode(text):
    ids = vocab(tokenizer(text))[:max_len]
    return ids + [vocab['&lt;pad&gt;']] * (max_len - len(ids))    # pad_sequences-এর মতো

def collate_batch(batch):
    labels = torch.tensor([1.0 if label == 'pos' else 0.0 for label, _ in batch]).unsqueeze(1)
    texts  = torch.tensor([encode(text) for _, text in batch])
    return texts, labels

train_loader = DataLoader(list(IMDB(split='train')), batch_size=128, shuffle=True,  collate_fn=collate_batch)
test_loader  = DataLoader(list(IMDB(split='test')),  batch_size=128, shuffle=False, collate_fn=collate_batch)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)   # word → vector
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(self.relu(self.fc1(h_n[-1])))
        return self.fc2(x)          # raw logit — BCEWithLogitsLoss নিজেই sigmoid করে

model_text = SentimentLSTM(vocab_size=len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_text.parameters(), lr=0.001)

for epoch in range(5):
    model_text.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_text(xb), yb)
        loss.backward()
        optimizer.step()

model_text.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = (torch.sigmoid(model_text(xb)) >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct/total:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>RNN</td><td>Sequential data-র জন্য — hidden state দিয়ে আগের information বহন</td></tr>
        <tr><td>Vanishing Gradient</td><td>দীর্ঘ sequence-এ RNN ব্যর্থ — LSTM/GRU দিয়ে সমাধান</td></tr>
        <tr><td>LSTM</td><td>Cell state + 3 gates — long-term dependency শেখার জন্য সেরা</td></tr>
        <tr><td>GRU</td><td>2 gates — LSTM-এর simplified version, দ্রুত, প্রায় সমান কার্যকর</td></tr>
        <tr><td>Stacking nn.LSTM layers</td><td>এক LSTM-এর পুরো sequence output পরের LSTM-এ পাঠাও — return_sequences flag লাগে না</td></tr>
        <tr><td>Embedding layer</td><td>Text classification-এ word → dense vector mapping</td></tr>
      </tbody>
    </table>
  `,
};
