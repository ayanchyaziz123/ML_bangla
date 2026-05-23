export const nn_2_backprop = {
  title: "ব্যাকপ্রপাগেশন ও গ্রেডিয়েন্ট ডিসেন্ট",
  description: "Forward pass, loss function, chain rule দিয়ে backpropagation, gradient descent variants এবং Adam optimizer সহ নিউরাল নেটওয়ার্কের শেখার প্রক্রিয়া বিস্তারিত।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-backpropagation",
  content: `
    <h3>১. ফরওয়ার্ড পাস: ইনপুট থেকে প্রেডিকশন</h3>
    <p>নিউরাল নেটওয়ার্ক শেখার প্রক্রিয়া দুটি ধাপে হয়: Forward Pass এবং Backward Pass। Forward Pass-এ আমরা ইনপুট থেকে prediction বের করি।</p>
    <pre><code>import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(X, W1, b1, W2, b2):
    """
    2-layer neural network forward pass
    Layer 1: relu activation (hidden)
    Layer 2: sigmoid activation (output)
    """
    # Layer 1: hidden
    Z1 = np.dot(X, W1) + b1        # (n, h)
    A1 = np.maximum(0, Z1)          # ReLU activation

    # Layer 2: output
    Z2 = np.dot(A1, W2) + b2        # (n, 1)
    A2 = sigmoid(Z2)                # Sigmoid for binary classification

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

# উদাহরণ
np.random.seed(0)
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # XOR inputs
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))

A2, cache = forward_pass(X, W1, b1, W2, b2)
print("Predictions:", A2.flatten())
</code></pre>

    <h3>২. লস ফাংশন: ভুল পরিমাপ করা</h3>
    <p>Loss function বলে দেয় আমাদের prediction কতটা ভুল। এই ভুল কমানোই training-এর লক্ষ্য।</p>
    <p><strong>MSE (Regression-এর জন্য):</strong></p>
    <p>L = (1/n) Σ (yᵢ - ŷᵢ)²</p>
    <p><strong>Binary Cross-Entropy (Binary Classification-এর জন্য):</strong></p>
    <p>L = -(1/n) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]</p>
    <p><strong>Categorical Cross-Entropy (Multi-class-এর জন্য):</strong></p>
    <p>L = -(1/n) Σᵢ Σⱼ yᵢⱼ·log(ŷᵢⱼ)</p>
    <pre><code>def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss
    epsilon: numerical stability এর জন্য
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# উদাহরণ
y_true = np.array([[0], [1], [1], [0]])
y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])
print("BCE Loss:", binary_crossentropy(y_true, y_pred))  # ~0.14
print("MSE Loss:", mse_loss(y_true, y_pred))              # ~0.025
</code></pre>

    <h3>৩. ব্যাকপ্রপাগেশন: চেইন রুল দিয়ে গ্রেডিয়েন্ট হিসাব</h3>
    <p>Backpropagation হলো chain rule ব্যবহার করে প্রতিটি weight-এর জন্য dL/dw হিসাব করার প্রক্রিয়া।</p>
    <p>Chain rule: <strong>dL/dw₁ = dL/dA₂ · dA₂/dZ₂ · dZ₂/dA₁ · dA₁/dZ₁ · dZ₁/dw₁</strong></p>
    <p>Output layer gradient (sigmoid + BCE):</p>
    <p><strong>dL/dZ₂ = A₂ - y</strong> (এটি একটি সুন্দর সরলীকরণ)</p>
    <pre><code>def backward_pass(X, y, W1, W2, cache):
    """
    Compute gradients using backpropagation
    """
    n = X.shape[0]
    Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']

    # Output layer gradient
    # dL/dZ2 = A2 - y  (BCE loss + sigmoid এর combined derivative)
    dZ2 = A2 - y                          # (n, 1)
    dW2 = np.dot(A1.T, dZ2) / n          # (h, 1)
    db2 = np.mean(dZ2, axis=0, keepdims=True)  # (1, 1)

    # Hidden layer gradient
    dA1 = np.dot(dZ2, W2.T)              # (n, h)
    dZ1 = dA1 * (Z1 > 0)                 # ReLU derivative: 1 if z>0 else 0
    dW1 = np.dot(X.T, dZ1) / n           # (features, h)
    db1 = np.mean(dZ1, axis=0, keepdims=True)  # (1, h)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

# Chain rule step by step:
# dL/dA2 = -(y/A2) + (1-y)/(1-A2)
# dA2/dZ2 = A2*(1-A2)  [sigmoid derivative]
# dL/dZ2 = dL/dA2 * dA2/dZ2 = A2 - y  [simplified]
# dZ2/dW2 = A1
# dL/dW2 = dL/dZ2 * dZ2/dW2 = (A2-y) * A1
</code></pre>

    <h3>৪. ওজন আপডেট: গ্রেডিয়েন্ট ডিসেন্ট</h3>
    <p>Gradient descent নিয়ম: <strong>w = w - α · dL/dw</strong></p>
    <p>এখানে α হলো learning rate। বড় α = দ্রুত কিন্তু unstable; ছোট α = ধীর কিন্তু stable।</p>
    <pre><code>def update_weights(W1, b1, W2, b2, grads, learning_rate=0.01):
    """Gradient descent weight update"""
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    return W1, b1, W2, b2

# Complete training loop
def train(X, y, W1, b1, W2, b2, epochs=1000, lr=0.1):
    losses = []
    for epoch in range(epochs):
        # Forward pass
        A2, cache = forward_pass(X, W1, b1, W2, b2)

        # Compute loss
        loss = binary_crossentropy(y, A2)
        losses.append(loss)

        # Backward pass
        grads = backward_pass(X, y, W1, W2, cache)

        # Weight update
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, grads, lr)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, b1, W2, b2, losses

y_xor = np.array([[0],[1],[1],[0]])
W1, b1, W2, b2, losses = train(X, y_xor, W1, b1, W2, b2, epochs=2000, lr=0.5)
A2_final, _ = forward_pass(X, W1, b1, W2, b2)
print("Final predictions:", np.round(A2_final.flatten()))  # [0, 1, 1, 0]
</code></pre>

    <h3>৫. গ্রেডিয়েন্ট ডিসেন্টের ধরন</h3>
    <table>
      <thead><tr><th>ধরন</th><th>প্রতি step-এ data</th><th>সুবিধা</th><th>অসুবিধা</th></tr></thead>
      <tbody>
        <tr><td>Batch GD</td><td>সব data</td><td>Stable convergence</td><td>বড় dataset-এ ধীর, memory বেশি</td></tr>
        <tr><td>Stochastic GD (SGD)</td><td>১টি sample</td><td>দ্রুত update, কম memory</td><td>Noisy, অস্থির convergence</td></tr>
        <tr><td>Mini-batch GD</td><td>32-256 sample</td><td>Balance: দ্রুত + stable</td><td>Batch size tuning দরকার</td></tr>
      </tbody>
    </table>
    <p>Mini-batch GD সবচেয়ে বেশি ব্যবহৃত। GPU parallelism-এর সুবিধা নেয়।</p>

    <h3>৬. অ্যাডভান্সড অপটিমাইজার ও Vanishing Gradient</h3>
    <p><strong>Vanishing Gradient সমস্যা:</strong> Deep network-এ backprop করার সময় gradients layer-by-layer ছোট হতে থাকে। প্রথম দিকের layers প্রায় শিখতে পারে না।</p>
    <p>সমাধান: ReLU activation, Batch Normalization, Residual connections (ResNet)।</p>
    <pre><code>import numpy as np

# Adam Optimizer implementation
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # first moment (mean)
        self.v = {}  # second moment (variance)
        self.t = 0   # timestep

    def update(self, params, grads):
        self.t += 1
        updated = {}
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Momentum update
            self.m[key] = self.beta1 * self.m[key] + (1-self.beta1) * grads['d'+key]
            self.v[key] = self.beta2 * self.v[key] + (1-self.beta2) * grads['d'+key]**2

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Parameter update
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated

# Optimizer তুলনা
# SGD:     w = w - lr * gradient
# Momentum: velocity = beta*velocity + lr*gradient; w = w - velocity
# RMSprop: cache = beta*cache + (1-beta)*gradient^2; w = w - lr*gradient/sqrt(cache)
# Adam:    RMSprop + Momentum combined (সাধারণত সেরা)
print("Adam: lr=0.001, beta1=0.9, beta2=0.999 — default ভালো কাজ করে")
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম:</p>
    <ul>
      <li>Forward pass: input → hidden → output → loss</li>
      <li>Loss functions: MSE (regression), Binary Cross-Entropy (binary), Categorical CE (multi-class)</li>
      <li>Backpropagation: chain rule দিয়ে প্রতিটি weight-এর gradient বের করা</li>
      <li>Weight update: w = w - α · dL/dw</li>
      <li>Gradient descent variants: Batch, SGD, Mini-batch</li>
      <li>Adam optimizer: momentum + adaptive learning rate</li>
      <li>Vanishing gradient: deep network-এর চ্যালেঞ্জ</li>
    </ul>
    <p>পরবর্তী ব্লগে আমরা NumPy দিয়ে সম্পূর্ণ neural network scratch থেকে implement করব।</p>
  `,
};
