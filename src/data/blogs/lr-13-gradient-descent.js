export const lr_13_gradient_descent = {
  title: "Gradient Descent: মডেল কীভাবে শেখে?",
  description: "Gradient Descent-এর গণিত, Learning Rate কী করে, Batch vs Stochastic vs Mini-batch — লিনিয়ার রিগ্রেশনের ইঞ্জিন বাংলায় বোঝো।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 12,
  slug: "lr-gradient-descent",
  content: `
    <h3>১. মডেল কীভাবে শেখে?</h3>
    <p>লিনিয়ার রিগ্রেশনে মডেল শেখে মানে coefficient (m) এবং intercept (b) এর সঠিক মান খোঁজা। কিন্তু সে কীভাবে খোঁজে? উত্তর: <strong>Gradient Descent</strong>।</p>
    <p>ধরো তুমি পাহাড়ের চূড়ায় আছো কিন্তু কুয়াশায় কিছু দেখতে পাচ্ছো না। নিচে নামতে হবে — তুমি পা দিয়ে অনুভব করো কোন দিকে ঢাল বেশি, সেদিকে এক পা দাও। আবার অনুভব করো, আবার এক পা। এভাবে একটু একটু করে সমতলে (সবচেয়ে নিচে) পৌঁছাও। এটাই Gradient Descent।</p>

    <h3>২. Loss Function — কী কমাতে চাই?</h3>
    <p>মডেল যা কমাতে চায় তাকে বলে <strong>Loss Function</strong>। লিনিয়ার রিগ্রেশনে এটি হলো MSE:</p>
    <pre><code>MSE = (1/n) × Σ(yᵢ − ŷᵢ)²
    = (1/n) × Σ(yᵢ − (m×xᵢ + b))²

যেখানে:
  yᵢ  = আসল মান
  ŷᵢ  = মডেলের অনুমান
  m   = slope (coefficient)
  b   = intercept
  n   = ডেটা পয়েন্টের সংখ্যা</code></pre>

    <h3>৩. Gradient কী?</h3>
    <p>Gradient হলো Loss Function-এর <strong>partial derivative</strong> — বলে দেয় m বা b একটু বাড়ালে Loss কতটুকু বাড়ে বা কমে।</p>
    <pre><code># MSE-র derivative:
∂MSE/∂m = (-2/n) × Σ xᵢ(yᵢ − ŷᵢ)   ← m-এর দিকে ঢাল
∂MSE/∂b = (-2/n) × Σ (yᵢ − ŷᵢ)     ← b-এর দিকে ঢাল

# Update rule:
m = m − α × ∂MSE/∂m
b = b − α × ∂MSE/∂b

যেখানে α (alpha) = Learning Rate</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Gradient Descent — Loss কমানোর পথ</text>
        <path d="M 40 120 Q 150 30 240 110 Q 310 170 380 60 Q 420 30 440 50" stroke="#e5e7eb" stroke-width="2" fill="none"/>
        <path d="M 40 120 Q 100 50 180 90 Q 230 115 260 108" stroke="#1e40af" stroke-width="2.5" fill="none"/>
        <circle cx="40" cy="120" r="5" fill="#dc2626"/>
        <text x="40" y="135" text-anchor="middle" font-size="9" fill="#dc2626">শুরু</text>
        <circle cx="260" cy="108" r="5" fill="#16a34a"/>
        <text x="260" y="123" text-anchor="middle" font-size="9" fill="#16a34a">minimum</text>
        <text x="90" y="75" font-size="8" fill="#6b7280">প্রতিটি পদক্ষেপ = α × gradient</text>
        <line x1="60" y1="115" x2="100" y2="95" stroke="#1e40af" stroke-width="1.5" marker-end="url(#arr)"/>
        <line x1="100" y1="95" x2="140" y2="92" stroke="#1e40af" stroke-width="1.5"/>
        <line x1="140" y1="92" x2="180" y2="90" stroke="#1e40af" stroke-width="1.5"/>
        <line x1="180" y1="90" x2="220" y2="105" stroke="#1e40af" stroke-width="1.5"/>
        <line x1="220" y1="105" x2="255" y2="108" stroke="#1e40af" stroke-width="1.5"/>
      </svg>
    </div>

    <h3>৪. Python দিয়ে Gradient Descent (স্ক্র্যাচ থেকে)</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# ডেটা
np.random.seed(42)
X = 2 * np.random.rand(100)
y = 4 + 3 * X + np.random.randn(100)

# Gradient Descent
m, b = 0.0, 0.0          # শুরুর মান
alpha = 0.1               # learning rate
epochs = 1000             # কতবার update করবো
n = len(X)
loss_history = []

for epoch in range(epochs):
    y_pred = m * X + b

    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # Update
    m = m - alpha * dm
    b = b - alpha * db

    # Loss track
    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

print(f"m = {m:.4f}")   # ≈ 3.0
print(f"b = {b:.4f}")   # ≈ 4.0
print(f"Final MSE = {loss_history[-1]:.4f}")</code></pre>

    <h3>৫. Learning Rate — সবচেয়ে গুরুত্বপূর্ণ hyperparameter</h3>
    <table>
      <thead><tr><th>Learning Rate</th><th>কী হয়</th><th>সমস্যা</th></tr></thead>
      <tbody>
        <tr><td>অনেক বড় (α = 1.0)</td><td>বড় বড় পদক্ষেপ</td><td>minimum পেরিয়ে যায়, diverge করে</td></tr>
        <tr><td>অনেক ছোট (α = 0.0001)</td><td>ছোট ছোট পদক্ষেপ</td><td>অনেক সময় লাগে, আটকে যেতে পারে</td></tr>
        <tr><td>ঠিকঠাক (α = 0.01–0.1)</td><td>ভারসাম্যপূর্ণ</td><td>দ্রুত এবং সঠিকভাবে converge করে</td></tr>
      </tbody>
    </table>
    <pre><code># Learning rate effect দেখা
for alpha in [0.001, 0.01, 0.1, 0.5]:
    m, b = 0.0, 0.0
    losses = []
    for _ in range(200):
        y_pred = m * X + b
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        m -= alpha * dm
        b -= alpha * db
        losses.append(np.mean((y - y_pred)**2))
    print(f"α={alpha}: Final Loss = {losses[-1]:.3f}")</code></pre>

    <h3>৬. তিন ধরনের Gradient Descent</h3>
    <table>
      <thead><tr><th>ধরন</th><th>প্রতিটি update-এ কতটুকু ডেটা?</th><th>সুবিধা</th><th>অসুবিধা</th></tr></thead>
      <tbody>
        <tr><td><strong>Batch GD</strong></td><td>পুরো ডেটাসেট</td><td>স্থিতিশীল, smooth convergence</td><td>বড় ডেটায় ধীর</td></tr>
        <tr><td><strong>Stochastic GD (SGD)</strong></td><td>একটি মাত্র sample</td><td>দ্রুত, online learning</td><td>noisy, অস্থির</td></tr>
        <tr><td><strong>Mini-batch GD</strong></td><td>ছোট batch (32–256)</td><td>দুটোর মাঝামাঝি সুবিধা</td><td>batch size বেছে নিতে হয়</td></tr>
      </tbody>
    </table>
    <pre><code># Mini-batch Gradient Descent
batch_size = 32
m, b = 0.0, 0.0
alpha = 0.01

for epoch in range(100):
    # ডেটা shuffle করো
    indices = np.random.permutation(n)
    X_s, y_s = X[indices], y[indices]

    for start in range(0, n, batch_size):
        Xb = X_s[start:start + batch_size]
        yb = y_s[start:start + batch_size]
        nb = len(Xb)

        y_pred = m * Xb + b
        dm = (-2/nb) * np.sum(Xb * (yb - y_pred))
        db = (-2/nb) * np.sum(yb - y_pred)
        m -= alpha * dm
        b -= alpha * db

print(f"m={m:.3f}, b={b:.3f}")</code></pre>

    <h3>৭. sklearn কীভাবে করে?</h3>
    <pre><code>from sklearn.linear_model import LinearRegression, SGDRegressor

# sklearn LinearRegression — closed-form (OLS) ব্যবহার করে, GD নয়
lr = LinearRegression()
lr.fit(X.reshape(-1,1), y)
print(f"sklearn: m={lr.coef_[0]:.3f}, b={lr.intercept_:.3f}")

# SGDRegressor — Stochastic Gradient Descent ব্যবহার করে
sgd = SGDRegressor(learning_rate='constant', eta0=0.01,
                   max_iter=1000, random_state=42)
sgd.fit(X.reshape(-1,1), y)
print(f"SGD: m={sgd.coef_[0]:.3f}, b={sgd.intercept_[0]:.3f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Gradient Descent</td><td>Loss কমাতে coefficient পর্যায়ক্রমে update করে</td></tr>
        <tr><td>Gradient</td><td>Loss-এর derivative — কোন দিকে কমবে তা বলে</td></tr>
        <tr><td>Learning Rate (α)</td><td>পদক্ষেপের আকার — না বেশি বড়, না বেশি ছোট</td></tr>
        <tr><td>Epoch</td><td>পুরো ডেটা একবার দেখা</td></tr>
        <tr><td>Mini-batch</td><td>Speed + stability-র ভারসাম্য</td></tr>
      </tbody>
    </table>
  `,
};
