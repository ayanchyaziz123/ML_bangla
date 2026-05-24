export const nnEn = [
  {
    title: "Perceptron & Neural Network Math",
    description: "From biological neurons to the perceptron, activation functions, the XOR problem, and the foundations of Multi-Layer Perceptrons explained with math and code.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 12,
    slug: "nn-perceptron-math",
    content: `
      <h3>1. From Biological to Artificial Neurons</h3>
      <p>The human brain contains roughly 86 billion neurons. Each neuron receives signals from other neurons, processes them, and sends an output. Artificial neural networks mathematically mimic this process.</p>
      <p>Comparison of biological and artificial neurons:</p>
      <table>
        <thead><tr><th>Biological Neuron</th><th>Artificial Neuron</th><th>Role</th></tr></thead>
        <tbody>
          <tr><td>Dendrite</td><td>Inputs (x₁, x₂, ...)</td><td>Receives signals</td></tr>
          <tr><td>Synapse</td><td>Weights (w₁, w₂, ...)</td><td>Determines signal strength</td></tr>
          <tr><td>Cell body</td><td>Summation + Activation</td><td>Processes signals</td></tr>
          <tr><td>Axon</td><td>Output (ŷ)</td><td>Sends result</td></tr>
        </tbody>
      </table>
      <p>An artificial neuron works in two steps: first compute the weighted sum, then apply an activation function.</p>

      <h3>2. Perceptron Math</h3>
      <p>The perceptron is the simplest artificial neuron. Its formula:</p>
      <p><strong>z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b</strong></p>
      <p><strong>output = activation(z)</strong></p>
      <ul>
        <li><strong>x₁ ... xₙ</strong> — input features</li>
        <li><strong>w₁ ... wₙ</strong> — weights for each input</li>
        <li><strong>b</strong> — bias (shifts the threshold)</li>
        <li><strong>z</strong> — weighted sum / pre-activation</li>
        <li><strong>activation(z)</strong> — adds non-linearity</li>
      </ul>
      <p>In matrix form: <strong>z = Xᵀw + b</strong>.</p>
      <pre><code>import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.01):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z >= 0, 1, 0)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                errors += int(error != 0)
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break
        return self

# AND gate example
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.train(X, y, epochs=100)
print("Predictions:", p.predict(X))  # [0, 0, 0, 1]
</code></pre>

      <h3>3. Activation Functions</h3>
      <p>Activation functions add non-linearity to the network. Without them, no matter how many layers, the whole network is just one linear transformation.</p>
      <pre><code>def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def tanh(z):
    return np.tanh(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("Softmax:", probs)       # [0.659, 0.242, 0.099]
print("Sum:", np.sum(probs))   # 1.0
</code></pre>
      <table>
        <thead><tr><th>Function</th><th>Formula</th><th>Range</th><th>Use Case</th><th>Problem</th></tr></thead>
        <tbody>
          <tr><td>Sigmoid</td><td>1/(1+e⁻ᶻ)</td><td>0 to 1</td><td>Binary output</td><td>Vanishing gradient</td></tr>
          <tr><td>ReLU</td><td>max(0,z)</td><td>0 to ∞</td><td>Hidden layers</td><td>Dying ReLU</td></tr>
          <tr><td>Tanh</td><td>(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)</td><td>-1 to 1</td><td>Hidden (RNN)</td><td>Vanishing gradient</td></tr>
          <tr><td>Softmax</td><td>eᶻⁱ/Σeᶻʲ</td><td>0 to 1</td><td>Multi-class output</td><td>—</td></tr>
          <tr><td>Leaky ReLU</td><td>max(αz,z)</td><td>-∞ to ∞</td><td>Hidden layers</td><td>α tuning</td></tr>
        </tbody>
      </table>

      <h3>4. The XOR Problem & Why We Need Multiple Layers</h3>
      <p>A single perceptron can only solve linearly separable problems. The XOR gate is not linearly separable, so a single perceptron cannot solve it.</p>
      <pre><code>X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.train(X_xor, y_xor, epochs=1000)
print("XOR Predictions:", p.predict(X_xor))
# Can never produce [0,1,1,0] — needs a hidden layer!
</code></pre>
      <p>To solve XOR we need a hidden layer that creates a non-linear decision boundary.</p>

      <h3>5. Multi-Layer Perceptron (MLP)</h3>
      <p>An MLP has three types of layers: Input Layer (no computation), Hidden Layer(s) (learn non-linear patterns, typically ReLU), and Output Layer (final prediction, task-dependent activation).</p>
      <pre><code>def mlp_forward(X, W1, b1, W2, b2):
    """
    2-layer MLP forward pass
    X: input (n_samples, n_features)
    """
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, A2

np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.01
b2 = np.zeros((1, 1))

X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
A1, A2 = mlp_forward(X_xor, W1, b1, W2, b2)
print("Output shape:", A2.shape)       # (4, 1)
print("Pre-train predictions:", A2.flatten())
</code></pre>

      <h3>6. Summary</h3>
      <p>In this post we covered:</p>
      <ul>
        <li>The relationship between biological and artificial neurons</li>
        <li>Perceptron math: z = Xᵀw + b, output = activation(z)</li>
        <li>Activation functions: Sigmoid, ReLU, Tanh, Softmax</li>
        <li>The XOR problem: single perceptron limitation</li>
        <li>MLP: solving complex problems with multiple layers</li>
      </ul>
      <p>The next post covers Backpropagation — how neural networks actually learn.</p>
    `,
  },
  {
    title: "Backpropagation & Gradient Descent",
    description: "Forward pass, loss functions, chain rule backpropagation, gradient descent variants, Adam optimizer, and the vanishing gradient problem explained with full NumPy code.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 12,
    slug: "nn-backpropagation",
    content: `
      <h3>1. Forward Pass: From Input to Prediction</h3>
      <p>Neural network learning happens in two phases: Forward Pass (compute prediction) and Backward Pass (compute gradients and update weights).</p>
      <pre><code>import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(X, W1, b1, W2, b2):
    """2-layer neural network forward pass"""
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)         # ReLU hidden
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)               # Sigmoid output (binary)
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

np.random.seed(0)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))
A2, cache = forward_pass(X, W1, b1, W2, b2)
print("Predictions:", A2.flatten())
</code></pre>

      <h3>2. Loss Functions: Measuring Error</h3>
      <p>The loss function measures how wrong our prediction is. Training aims to minimize this loss.</p>
      <p><strong>MSE (Regression):</strong> L = (1/n) Σ (yᵢ - ŷᵢ)²</p>
      <p><strong>Binary Cross-Entropy (Binary Classification):</strong> L = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]</p>
      <p><strong>Categorical Cross-Entropy (Multi-class):</strong> L = -(1/n) Σᵢ Σⱼ yᵢⱼ·log(ŷᵢⱼ)</p>
      <pre><code>def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

y_true = np.array([[0], [1], [1], [0]])
y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])
print("BCE Loss:", binary_crossentropy(y_true, y_pred))  # ~0.14
print("MSE Loss:", mse_loss(y_true, y_pred))              # ~0.025
</code></pre>

      <h3>3. Backpropagation: Chain Rule Gradients</h3>
      <p>Backpropagation uses the chain rule to compute dL/dw for every weight in the network.</p>
      <p>Chain rule: <strong>dL/dw₁ = dL/dA₂ · dA₂/dZ₂ · dZ₂/dA₁ · dA₁/dZ₁ · dZ₁/dw₁</strong></p>
      <p>Output layer gradient (sigmoid + BCE simplifies to): <strong>dL/dZ₂ = A₂ - y</strong></p>
      <pre><code>def backward_pass(X, y, W1, W2, cache):
    n = X.shape[0]
    Z1, A1, Z2, A2 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2']

    # Output layer
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / n
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    # Hidden layer (ReLU derivative)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / n
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
</code></pre>

      <h3>4. Weight Update: Gradient Descent</h3>
      <p>Gradient descent rule: <strong>w = w - α · dL/dw</strong></p>
      <p>Large α = fast but unstable; small α = slow but stable.</p>
      <pre><code>def update_weights(W1, b1, W2, b2, grads, lr=0.01):
    W1 -= lr * grads['dW1']
    b1 -= lr * grads['db1']
    W2 -= lr * grads['dW2']
    b2 -= lr * grads['db2']
    return W1, b1, W2, b2

def train(X, y, W1, b1, W2, b2, epochs=1000, lr=0.1):
    losses = []
    for epoch in range(epochs):
        A2, cache = forward_pass(X, W1, b1, W2, b2)
        loss = binary_crossentropy(y, A2)
        losses.append(loss)
        grads = backward_pass(X, y, W1, W2, cache)
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, grads, lr)
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2, losses

y_xor = np.array([[0],[1],[1],[0]])
W1, b1, W2, b2, losses = train(X, y_xor, W1, b1, W2, b2, epochs=2000, lr=0.5)
A2_final, _ = forward_pass(X, W1, b1, W2, b2)
print("Final predictions:", np.round(A2_final.flatten()))  # [0, 1, 1, 0]
</code></pre>

      <h3>5. Gradient Descent Variants</h3>
      <table>
        <thead><tr><th>Type</th><th>Data per step</th><th>Pros</th><th>Cons</th></tr></thead>
        <tbody>
          <tr><td>Batch GD</td><td>All data</td><td>Stable convergence</td><td>Slow on large datasets, high memory</td></tr>
          <tr><td>Stochastic GD (SGD)</td><td>1 sample</td><td>Fast updates, low memory</td><td>Noisy, unstable convergence</td></tr>
          <tr><td>Mini-batch GD</td><td>32-256 samples</td><td>Balance: fast + stable</td><td>Batch size tuning needed</td></tr>
        </tbody>
      </table>

      <h3>6. Advanced Optimizers & Vanishing Gradient</h3>
      <p><strong>Vanishing Gradient:</strong> In deep networks, gradients shrink layer by layer during backprop. Early layers barely learn. Solutions: ReLU activation, Batch Normalization, Residual connections.</p>
      <pre><code>class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2
        self.epsilon = epsilon; self.m = {}; self.v = {}; self.t = 0

    def update(self, params, grads):
        self.t += 1
        updated = {}
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads['d'+key]
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*grads['d'+key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated

# SGD:     w = w - lr * gradient
# Momentum: v = beta*v + lr*g; w = w - v
# RMSprop: cache = beta*cache + (1-beta)*g^2; w = w - lr*g/sqrt(cache)
# Adam:    Momentum + RMSprop combined (usually best default)
</code></pre>

      <h3>7. Summary</h3>
      <ul>
        <li>Forward pass: input → hidden → output → loss</li>
        <li>Loss functions: MSE (regression), Binary CE (binary), Categorical CE (multi-class)</li>
        <li>Backpropagation: chain rule to compute every weight's gradient</li>
        <li>Weight update: w = w - α · dL/dw</li>
        <li>Gradient descent variants: Batch, SGD, Mini-batch</li>
        <li>Adam optimizer: momentum + adaptive learning rate</li>
        <li>Vanishing gradient: the core challenge of deep networks</li>
      </ul>
    `,
  },
  {
    title: "Neural Network from Scratch with NumPy",
    description: "Build a complete class-based 2-layer neural network using only NumPy — forward pass, backward pass, mini-batch training, weight initialization, and decision boundary visualization.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 13,
    slug: "nn-numpy-from-scratch",
    content: `
      <h3>1. Why Build from Scratch?</h3>
      <p>TensorFlow and PyTorch automate everything. But without implementing from scratch you'll never truly understand why a model fails or how to debug it. In this post we use only NumPy — no ML frameworks.</p>
      <ul>
        <li>Object-oriented design: NeuralNetwork class</li>
        <li>Binary classification: make_circles dataset</li>
        <li>Complete pipeline: init → forward → loss → backward → train → predict</li>
      </ul>

      <h3>2. Dataset & Initialization</h3>
      <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)
</code></pre>

      <h3>3. NeuralNetwork Class</h3>
      <pre><code>class NeuralNetwork:
    """2-layer: input -> hidden (relu) -> output (sigmoid)"""

    def __init__(self, n_features, n_hidden, learning_rate=0.01, random_state=42):
        np.random.seed(random_state)
        self.lr = learning_rate
        # He initialization for ReLU
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2 / n_features)
        self.b1 = np.zeros((1, n_hidden))
        # Xavier initialization for sigmoid
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(1 / n_hidden)
        self.b2 = np.zeros((1, 1))
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y):
        n = X.shape[0]
        dZ2 = self.A2 - y
        self.dW2 = np.dot(self.A1.T, dZ2) / n
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / n
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        self.dW1 = np.dot(X.T, dZ1) / n
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / n

    def update_weights(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
</code></pre>

      <h3>4. Training Loop</h3>
      <pre><code>    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1000, batch_size=32, verbose=True):
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                y_pred = self.forward(X_shuffled[start:end])
                epoch_loss += self.compute_loss(y_shuffled[start:end], y_pred)
                self.backward(X_shuffled[start:end], y_shuffled[start:end])
                self.update_weights()

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                self.val_losses.append(self.compute_loss(y_val, val_pred))

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {self.accuracy(X_train, y_train):.3f}")
        return self
</code></pre>

      <h3>5. Train, Evaluate & Visualize</h3>
      <pre><code">model = NeuralNetwork(n_features=2, n_hidden=16, learning_rate=0.05)
model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=1000, batch_size=32)

print(f"Train Accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test Accuracy:  {model.accuracy(X_test, y_test):.4f}")

# Loss curve
plt.figure(figsize=(8, 4))
plt.plot(model.train_losses, label='Train Loss')
plt.plot(model.val_losses,   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
plt.title('Training & Validation Loss')
plt.legend(); plt.grid(True)
plt.savefig("loss_curve.png"); plt.show()
</code></pre>

      <h3>6. Weight Initialization Matters</h3>
      <pre><code>model_zeros = NeuralNetwork(n_features=2, n_hidden=8, learning_rate=0.05)
model_zeros.W1 = np.zeros((2, 8))   # BAD: symmetry problem
model_zeros.W2 = np.zeros((8, 1))
model_zeros.train(X_train, y_train, epochs=500, verbose=False)
print("Zero init accuracy:", model_zeros.accuracy(X_test, y_test))  # ~0.5

model_he = NeuralNetwork(n_features=2, n_hidden=8, learning_rate=0.05)
model_he.train(X_train, y_train, epochs=500, verbose=False)
print("He init accuracy:", model_he.accuracy(X_test, y_test))  # ~0.95+
# Zero init: all neurons receive identical gradients -> all learn the same thing
</code></pre>

      <h3>7. Summary</h3>
      <table>
        <thead><tr><th>Component</th><th>Function</th><th>Key Detail</th></tr></thead>
        <tbody>
          <tr><td>__init__</td><td>Weight initialization</td><td>He for ReLU, Xavier for sigmoid</td></tr>
          <tr><td>forward()</td><td>Z1→A1→Z2→A2</td><td>ReLU hidden, sigmoid output</td></tr>
          <tr><td>compute_loss()</td><td>Binary cross-entropy</td><td>clip for numerical stability</td></tr>
          <tr><td>backward()</td><td>Chain rule gradients</td><td>dZ2=A2-y, ReLU derivative</td></tr>
          <tr><td>update_weights()</td><td>Gradient descent</td><td>w -= lr * dw</td></tr>
          <tr><td>train()</td><td>Mini-batch loop</td><td>shuffle, batch, forward, backward, update</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Keras & TensorFlow Neural Networks",
    description: "TensorFlow/Keras Sequential API for building neural networks, MNIST handwritten digit classification, training callbacks, accuracy/loss curves, and model save/load.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 12,
    slug: "nn-keras-tensorflow",
    content: `
      <h3>1. Why Keras? From NumPy to Keras</h3>
      <p>The previous post needed 300+ lines to build a 2-layer network. Keras does the same in 10-15 lines. Benefits: automatic GPU/TPU support, automatic differentiation, built-in optimizers and losses, callbacks, and TensorBoard integration.</p>
      <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
tf.random.set_seed(42)
np.random.seed(42)
</code></pre>

      <h3>2. MNIST Dataset: Handwritten Digits</h3>
      <p>MNIST is the "Hello World" of ML — 70,000 handwritten digit images (0-9), each 28×28 grayscale pixels.</p>
      <pre><code>(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print("Training:", X_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print("Test:",     X_test.shape,  y_test.shape)    # (10000, 28, 28) (10000,)

# Preprocess
X_train = X_train.astype('float32') / 255.0   # Normalize 0-255 -> 0-1
X_test  = X_test.astype('float32') / 255.0
X_train_flat = X_train.reshape(-1, 784)        # Flatten 28x28 -> 784
X_test_flat  = X_test.reshape(-1, 784)

print("Label sample:", y_train[:10])  # [5 0 4 1 9 2 1 3 1 4]
</code></pre>

      <h3>3. Sequential Model</h3>
      <pre><code>model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 classes
], name="simple_mlp")

model.summary()
# dense:   100480 params
# dense_1:   8256 params
# dense_2:    650 params
# Total:  109,386

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # integer labels
    metrics=['accuracy']
)
</code></pre>

      <h3>4. Training: fit(), evaluate(), predict()</h3>
      <pre><code>early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)
model_checkpoint = callbacks.ModelCheckpoint(
    'best_mnist_model.keras', monitor='val_accuracy', save_best_only=True
)

history = model.fit(
    X_train_flat, y_train,
    epochs=30, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, model_checkpoint],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")   # ~0.98

y_pred_proba = model.predict(X_test_flat[:5])
y_pred = np.argmax(y_pred_proba, axis=1)
print("Predicted:", y_pred)
print("Actual:   ", y_test[:5])
</code></pre>

      <h3>5. Training Curves & Confusion Matrix</h3>
      <pre><code">def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='orange')
    axes[0].set_title('Model Accuracy'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history['loss'],     label='Train', color='blue')
    axes[1].plot(history.history['val_loss'], label='Val',   color='orange')
    axes[1].set_title('Model Loss'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

plot_training_history(history)

from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred_all = np.argmax(model.predict(X_test_flat), axis=1)
cm = confusion_matrix(y_test, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix — MNIST Test Set")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix.png"); plt.show()
</code></pre>

      <h3>6. Model Save & Load</h3>
      <pre><code>model.save('mnist_model.keras')                               # Keras format
model.save('mnist_model_savedmodel')                          # SavedModel format
model.save_weights('mnist_weights.weights.h5')                # Weights only

loaded_model = keras.models.load_model('mnist_model.keras')
test_loss, test_acc = loaded_model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Loaded model accuracy: {test_acc:.4f}")

single_image = X_test_flat[0:1]
prediction = loaded_model.predict(single_image)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class]
print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
</code></pre>
      <table>
        <thead><tr><th>API</th><th>Purpose</th><th>Returns</th></tr></thead>
        <tbody>
          <tr><td>model.fit()</td><td>Training</td><td>History object (loss, accuracy per epoch)</td></tr>
          <tr><td>model.evaluate()</td><td>Test set performance</td><td>[loss, metric1, ...]</td></tr>
          <tr><td>model.predict()</td><td>Raw predictions</td><td>NumPy array</td></tr>
          <tr><td>model.summary()</td><td>Architecture overview</td><td>Print to stdout</td></tr>
          <tr><td>model.save()</td><td>Persistence</td><td>File on disk</td></tr>
          <tr><td>keras.models.load_model()</td><td>Load saved model</td><td>Keras model object</td></tr>
        </tbody>
      </table>

      <h3>7. Summary</h3>
      <ul>
        <li>Sequential API: stack layers easily</li>
        <li>compile(): set optimizer, loss, metrics</li>
        <li>fit(): epochs, batch_size, validation_split, callbacks</li>
        <li>MNIST: normalize → flatten → Dense layers → softmax → ~98% accuracy</li>
        <li>EarlyStopping: prevents overfitting</li>
        <li>Model save/load: for production deployment</li>
      </ul>
    `,
  },
  {
    title: "CNN for Image Classification",
    description: "Convolutional Neural Network intuition, Conv2D filters, MaxPooling spatial downsampling, MNIST and CIFAR-10 with MLP vs CNN accuracy comparison, and Transfer Learning intro.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 13,
    slug: "nn-cnn-image-classification",
    content: `
      <h3>1. Why CNN? MLP Limitations on Images</h3>
      <p>MLPs have three key problems for images: too many parameters (224×224 RGB = 150K inputs), spatial information lost when flattened, and not translation-invariant (cat on left vs right = different features). CNNs solve these with local pattern sharing and spatial hierarchy.</p>
      <pre><code">import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# CNN intuition hierarchy:
# Layer 1 (Conv): simple edges (horizontal, vertical, diagonal)
# Layer 2 (Conv): corners, curves
# Layer 3 (Conv): eyes, wheels, textures
# Final layers:   "dog", "car", "face"
</code></pre>

      <h3>2. Convolution Operation</h3>
      <p>A filter/kernel (e.g., 3×3 matrix) slides over the image. At each position it does element-wise multiplication and sum — that is convolution.</p>
      <table>
        <thead><tr><th>Parameter</th><th>Meaning</th><th>Typical Value</th><th>Effect</th></tr></thead>
        <tbody>
          <tr><td>filters</td><td>Number of feature maps</td><td>32, 64, 128</td><td>More = more features learned</td></tr>
          <tr><td>kernel_size</td><td>Filter size</td><td>(3,3), (5,5)</td><td>Larger = wider receptive field</td></tr>
          <tr><td>padding</td><td>'valid' or 'same'</td><td>'same'</td><td>same = preserve spatial size</td></tr>
          <tr><td>strides</td><td>Filter step size</td><td>(1,1)</td><td>Larger stride = smaller output</td></tr>
          <tr><td>activation</td><td>Non-linearity</td><td>'relu'</td><td>Removes negative values</td></tr>
        </tbody>
      </table>
      <pre><code>import numpy as np

def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

image = np.array([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=float)
vertical_edge = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
feature_map = convolve2d(image, vertical_edge)
print("Feature map shape:", feature_map.shape)  # (3, 3)
</code></pre>

      <h3>3. MaxPooling: Spatial Downsampling</h3>
      <p>MaxPooling takes the maximum value from each window — reduces feature map size, provides translation invariance, and retains dominant features.</p>
      <pre><code">pool_input = np.array([[1,3,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]])
# 2x2 MaxPooling with stride 2:
# Top-left:     max(1,3,5,6) = 6
# Top-right:    max(2,4,7,8) = 8
# Bottom-left:  max(3,2,1,2) = 3
# Bottom-right: max(1,0,3,4) = 4
print(np.array([[6,8],[3,4]]))   # 4x4 -> 2x2
</code></pre>

      <h3>4. CNN Architecture for MNIST</h3>
      <pre><code">(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
X_train = X_train[..., np.newaxis]   # (60000, 28, 28, 1)
X_test  = X_test[..., np.newaxis]

cnn_model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),        # 28x28 -> 14x14

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),        # 14x14 -> 7x7

    layers.Flatten(),                  # 7*7*64 = 3136
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
], name="cnn_mnist")

cnn_model.summary()
cnn_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
</code></pre>

      <h3>5. Training & MLP vs CNN Comparison</h3>
      <pre><code">cnn_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=128,
                              validation_split=0.1, verbose=1)
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_acc:.4f}")   # ~0.993
</code></pre>
      <table>
        <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>MNIST Accuracy</th></tr></thead>
        <tbody>
          <tr><td>Simple MLP</td><td>784→128→64→10</td><td>~109K</td><td>~97.8%</td></tr>
          <tr><td>Deep MLP</td><td>784→256→128→64→10</td><td>~236K</td><td>~98.2%</td></tr>
          <tr><td>Simple CNN</td><td>Conv32→Pool→Conv64→Pool→128→10</td><td>~422K</td><td>~99.1%</td></tr>
          <tr><td>Deep CNN</td><td>2x(Conv→Pool)→Conv→128→10</td><td>~750K</td><td>~99.4%</td></tr>
        </tbody>
      </table>

      <h3>6. CIFAR-10 & Transfer Learning</h3>
      <pre><code">(X_cifar_train, y_cifar_train), (X_cifar_test, y_cifar_test) = keras.datasets.cifar10.load_data()
X_cifar_train = X_cifar_train.astype('float32') / 255.0

from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
base_model.trainable = False   # Freeze pre-trained weights

transfer_model = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print("Transfer learning: VGG16 pre-trained on 1.2M ImageNet images!")
</code></pre>

      <h3>7. Summary</h3>
      <ul>
        <li>CNN intuition: local patterns → hierarchical features</li>
        <li>Conv2D: filters, feature maps, padding, stride</li>
        <li>MaxPooling2D: spatial downsampling, translation invariance</li>
        <li>MNIST CNN: 99%+ accuracy vs MLP ~98%</li>
        <li>Transfer learning: fine-tune ImageNet pre-trained VGG16, ResNet, EfficientNet</li>
      </ul>
    `,
  },
  {
    title: "Regularization: Dropout, Batch Norm & Overfitting",
    description: "Detect and prevent overfitting in neural networks using Dropout, Batch Normalization, L1/L2 regularization, EarlyStopping, and Learning Rate Scheduling with comparison code.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 11,
    slug: "nn-regularization-dropout",
    content: `
      <h3>1. What is Overfitting & How to Detect It</h3>
      <p>Overfitting occurs when a model memorizes training data but fails on new data. Key sign: high training accuracy, low validation accuracy.</p>
      <pre><code">import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X = X.astype('float32')
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

def build_overfit_model():
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

baseline = build_overfit_model()
baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_overfit = baseline.fit(X_train, y_train, epochs=100, batch_size=32,
                                 validation_data=(X_val, y_val), verbose=0)
print(f"Train: {history_overfit.history['accuracy'][-1]:.3f}",
      f"Val: {history_overfit.history['val_accuracy'][-1]:.3f}")
# Train: 0.99, Val: 0.72 — OVERFITTING!
</code></pre>

      <h3>2. Dropout: Randomly Disabling Neurons</h3>
      <p>Dropout randomly zeros out neuron outputs during training. It forces neurons to learn independently without co-adapting. At inference all neurons are active but outputs are scaled by (1 - rate).</p>
      <pre><code">def build_dropout_model(dropout_rate=0.3):
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,)),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

dropout_model = build_dropout_model(0.3)
dropout_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_dropout = dropout_model.fit(X_train, y_train, epochs=100, batch_size=32,
                                     validation_data=(X_val, y_val), verbose=0)
print(f"Dropout — Train: {history_dropout.history['accuracy'][-1]:.3f}",
      f"Val: {history_dropout.history['val_accuracy'][-1]:.3f}")
# Train: 0.88, Val: 0.85 — gap much smaller!
</code></pre>

      <h3>3. Batch Normalization</h3>
      <p>Batch Normalization normalizes each layer's inputs (mean≈0, std≈1). It accelerates training, allows higher learning rates, and has a slight regularization effect.</p>
      <pre><code">def build_batchnorm_model():
    return keras.Sequential([
        layers.Dense(512, input_shape=(20,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(1, activation='sigmoid')
    ])
# BN before activation generally performs slightly better
</code></pre>

      <h3>4. L1 & L2 Regularization</h3>
      <p>L2 (Ridge): <strong>Total Loss = CE + λ·Σw²</strong> — keeps weights small</p>
      <p>L1 (Lasso): <strong>Total Loss = CE + λ·Σ|w|</strong> — encourages sparse weights</p>
      <pre><code">def build_l2_model(l2_lambda=0.001):
    return keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(20,),
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(1, activation='sigmoid')
    ])

# regularizers.l1(0.001)
# regularizers.l2(0.001)
# regularizers.l1_l2(l1=0.001, l2=0.001)
</code></pre>

      <h3>5. EarlyStopping & Learning Rate Scheduling</h3>
      <pre><code">early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10,
    restore_best_weights=True, min_delta=0.001, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Best-practice combined model
best_model = keras.Sequential([
    layers.Dense(256, input_shape=(20,), kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
    layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
best_model.compile(optimizer=keras.optimizers.Adam(0.001),
                    loss='binary_crossentropy', metrics=['accuracy'])
history_best = best_model.fit(X_train, y_train, epochs=200, batch_size=32,
                               validation_data=(X_val, y_val),
                               callbacks=[early_stop, reduce_lr], verbose=1)
</code></pre>
      <table>
        <thead><tr><th>Technique</th><th>Effect</th><th>When to Use</th><th>Key Parameter</th></tr></thead>
        <tbody>
          <tr><td>Dropout</td><td>Prevents co-adaptation</td><td>Fully connected layers</td><td>rate: 0.2–0.5</td></tr>
          <tr><td>Batch Norm</td><td>Normalizes + speeds training</td><td>Deep networks, Conv layers</td><td>momentum, epsilon</td></tr>
          <tr><td>L2 Reg</td><td>Penalizes large weights</td><td>Weight regularization</td><td>lambda: 0.001–0.01</td></tr>
          <tr><td>L1 Reg</td><td>Sparse weights</td><td>Feature selection</td><td>lambda: 0.001</td></tr>
          <tr><td>EarlyStopping</td><td>Stops at best epoch</td><td>Always</td><td>patience: 5–20</td></tr>
          <tr><td>ReduceLROnPlateau</td><td>Decreases lr at plateau</td><td>Learning stagnation</td><td>factor: 0.5, patience: 5</td></tr>
        </tbody>
      </table>

      <h3>6. Comparison Visualization</h3>
      <pre><code">import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_overfit.history['val_accuracy'], label='Baseline', color='red')
axes[0].plot(history_dropout.history['val_accuracy'], label='Dropout(0.3)', color='blue')
axes[0].set_title("Validation Accuracy"); axes[0].legend(); axes[0].grid(True)

gap_baseline = np.array(history_overfit.history['accuracy']) - np.array(history_overfit.history['val_accuracy'])
gap_dropout  = np.array(history_dropout.history['accuracy'])  - np.array(history_dropout.history['val_accuracy'])
axes[1].plot(gap_baseline, label='Baseline Gap', color='red')
axes[1].plot(gap_dropout,  label='Dropout Gap',  color='blue')
axes[1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
axes[1].set_title("Train-Val Gap"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.savefig("regularization_comparison.png"); plt.show()
</code></pre>

      <h3>7. Summary</h3>
      <ul>
        <li>Dropout: disables random neurons, prevents co-adaptation</li>
        <li>Batch Normalization: normalize + accelerate training</li>
        <li>L1/L2: weight penalties keep weights small</li>
        <li>EarlyStopping: stops at best epoch with restore_best_weights=True</li>
        <li>ReduceLROnPlateau: reduces learning rate when training plateaus</li>
      </ul>
    `,
  },
  {
    title: "Project: Fashion MNIST Clothing Classification",
    description: "End-to-end deep learning project on Fashion MNIST — data exploration, preprocessing, building 3 models (simple MLP, deep MLP, CNN), comparison, confusion matrix, and best practices.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 14,
    slug: "nn-project-fashion-mnist",
    content: `
      <h3>1. Fashion MNIST: Dataset Overview</h3>
      <p>Fashion MNIST is a harder drop-in replacement for MNIST. 70,000 grayscale images (28×28) across 10 clothing categories. Far more challenging than handwritten digits — a real-world image classification challenge.</p>
      <p>Classes: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Sandal (5), Shirt (6), Sneaker (7), Bag (8), Ankle boot (9).</p>
      <pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

tf.random.set_seed(42); np.random.seed(42)

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
print("Full training set:", X_train_full.shape)
print("Test set:", X_test.shape)
</code></pre>

      <h3>2. Data Exploration & Visualization</h3>
      <pre><code">fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_full[i], cmap='gray')
    ax.set_title(CLASS_NAMES[y_train_full[i]], fontsize=9)
    ax.axis('off')
plt.suptitle("Fashion MNIST Sample Images")
plt.tight_layout(); plt.savefig("fashion_samples.png"); plt.show()

class_counts = np.bincount(y_train_full)
print("Class distribution (all balanced ~6000 each):")
for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {i}: {name:<15} — {count} samples")
</code></pre>

      <h3>3. Preprocessing</h3>
      <pre><code>X_train_full = X_train_full.astype('float32') / 255.0
X_test       = X_test.astype('float32') / 255.0

val_size = 5000
X_val, y_val   = X_train_full[:val_size], y_train_full[:val_size]
X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train_flat = X_train.reshape(-1, 784)
X_val_flat   = X_val.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

X_train_cnn = X_train[..., np.newaxis]
X_val_cnn   = X_val[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        callbacks.ModelCheckpoint(f'{model_name}_best.keras', monitor='val_accuracy', save_best_only=True)
    ]
</code></pre>

      <h3>4. Build & Train Three Models</h3>
      <pre><code">def build_simple_mlp():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ], name='simple_mlp')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deep_mlp():
    model = keras.Sequential([
        layers.Dense(512, input_shape=(784,), kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.3),
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ], name='deep_mlp')
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ], name='cnn')
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

results = {}
for name, build_fn, X_tr, X_v, X_te in [
    ('simple_mlp', build_simple_mlp, X_train_flat, X_val_flat, X_test_flat),
    ('deep_mlp',   build_deep_mlp,   X_train_flat, X_val_flat, X_test_flat),
    ('cnn',        build_cnn,        X_train_cnn,  X_val_cnn,  X_test_cnn),
]:
    model = build_fn(); start = time.time()
    history = model.fit(X_tr, y_train, epochs=50, batch_size=64,
                        validation_data=(X_v, y_val),
                        callbacks=get_callbacks(name), verbose=0)
    t = time.time() - start
    loss, acc = model.evaluate(X_te, y_test, verbose=0)
    results[name] = {'model': model, 'history': history, 'test_acc': acc,
                     'n_params': model.count_params(), 'train_time': t}
    print(f"{name}: Params={model.count_params():,} | Acc={acc:.4f} | Time={t:.1f}s")
</code></pre>

      <h3>5. Comparison & Confusion Matrix</h3>
      <pre><code">print("\n" + "="*60)
print(f"{'Model':<15} {'Params':>10} {'Test Acc':>10} {'Time':>8}")
print("="*60)
for name, r in results.items():
    print(f"{name:<15} {r['n_params']:>10,} {r['test_acc']:>10.4f} {r['train_time']:>7.1f}s")

best_model = results['cnn']['model']
y_pred = np.argmax(best_model.predict(X_test_cnn), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("CNN Confusion Matrix — Fashion MNIST Test Set")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.savefig("fashion_confusion_matrix.png"); plt.show()
</code></pre>
      <table>
        <thead><tr><th>Model</th><th>Architecture</th><th>Parameters</th><th>Test Accuracy</th><th>Training Time</th></tr></thead>
        <tbody>
          <tr><td>Simple MLP</td><td>784→256→128→10 + Dropout</td><td>~234K</td><td>~88.5%</td><td>~30s</td></tr>
          <tr><td>Deep MLP + BN</td><td>784→512→256→128→10 + BN + Dropout + L2</td><td>~530K</td><td>~89.8%</td><td>~60s</td></tr>
          <tr><td>CNN</td><td>Conv32→Conv32→Pool→Conv64→Conv64→Pool→256→10</td><td>~620K</td><td>~92.5%</td><td>~180s</td></tr>
        </tbody>
      </table>

      <h3>6. Per-Class Accuracy & Error Analysis</h3>
      <pre><code">print("\nPer-Class Accuracy (CNN):")
for i, class_name in enumerate(CLASS_NAMES):
    mask = (y_test == i)
    acc = np.mean(y_pred[mask] == y_test[mask])
    print(f"  {i}: {class_name:<15} — {acc:.3f} ({acc*100:.1f}%)")

print("\nMost Confused Pairs:")
cm_no_diag = cm.copy(); np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    print(f"  True: {CLASS_NAMES[idx[0]]:<15} -> Pred: {CLASS_NAMES[idx[1]]:<15} ({cm_no_diag[idx]} errors)")
    cm_no_diag[idx] = 0

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
</code></pre>

      <h3>7. Best Practices Summary</h3>
      <table>
        <thead><tr><th>Best Practice</th><th>Why It Matters</th><th>Default Choice</th></tr></thead>
        <tbody>
          <tr><td>Normalize inputs</td><td>Stable gradients, faster convergence</td><td>0–1 or StandardScaler</td></tr>
          <tr><td>He/Xavier init</td><td>Symmetry breaking, gradient flow</td><td>Keras default (glorot_uniform)</td></tr>
          <tr><td>Adam optimizer</td><td>Adaptive lr + momentum</td><td>lr=0.001</td></tr>
          <tr><td>Batch Normalization</td><td>Faster training, higher lr possible</td><td>Always in deep networks</td></tr>
          <tr><td>EarlyStopping</td><td>Prevents overfitting, saves compute</td><td>patience=10, restore_best=True</td></tr>
          <tr><td>Validation set</td><td>Unbiased performance estimate</td><td>10–20% of training data</td></tr>
          <tr><td>Confusion matrix</td><td>Class-level insights</td><td>Always in final evaluation</td></tr>
        </tbody>
      </table>
      <p>This series covered the full journey from perceptron to a complete deep learning pipeline. The next series will explore Recurrent Neural Networks (RNN/LSTM) and Natural Language Processing.</p>
    `,
  },
  {
    title: "RNN & LSTM: Sequential Data and Time Series",
    description: "Recurrent Neural Networks for sequence data — hidden state mechanics, vanishing gradient problem, LSTM gates (forget/input/output), GRU, Keras time series prediction and IMDB sentiment analysis.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 13,
    slug: "nn-rnn-lstm",
    content: `
      <h3>1. Why Standard NNs Fail on Sequential Data</h3>
      <p>Standard neural networks process each input independently — they have no memory of previous inputs. Text, time series, audio — these require understanding <strong>order and context</strong>.</p>
      <pre><code># RNN solution: hidden state h_t carries information across time steps
#
# h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
# y_t = W_y * h_t + b_y
#
# x_t      = input at time t
# h_t      = hidden state at time t (the "memory")
# h_{t-1}  = previous hidden state</code></pre>
      <table>
        <thead><tr><th>Data Type</th><th>Examples</th><th>Sequence</th></tr></thead>
        <tbody>
          <tr><td>Time Series</td><td>Stock price, weather, sensors</td><td>Hours/days/minutes</td></tr>
          <tr><td>Text (NLP)</td><td>Sentiment, translation</td><td>Word/character sequence</td></tr>
          <tr><td>Audio</td><td>Speech recognition</td><td>Time frames</td></tr>
        </tbody>
      </table>

      <h3>2. Vanishing Gradient Problem</h3>
      <pre><code># In long sequences, gradients shrink toward zero during backpropagation
# Early time steps get almost no gradient signal — can't learn long-range dependencies
#
# dL/dW = dL/dh_T * dh_T/dh_{T-1} * ... * dh_2/dh_1 * dh_1/dW
# Each dh_t/dh_{t-1} < 1  →  product approaches 0 (vanishing)
# Each dh_t/dh_{t-1} > 1  →  product approaches inf (exploding)
#
# Solution: LSTM — designed specifically to remember long-term dependencies</code></pre>

      <h3>3. LSTM Gates</h3>
      <pre><code># LSTM has two states: cell state c_t (long-term) + hidden state h_t (short-term)
# Three gates control the information flow:
#
# 1. Forget Gate: what to discard from cell state?
#    f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   0=forget, 1=keep
#
# 2. Input Gate: what new info to store?
#    i_t  = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
#    c̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
#    c_t  = f_t * c_{t-1} + i_t * c̃_t
#
# 3. Output Gate: what to output?
#    o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
#    h_t = o_t * tanh(c_t)</code></pre>
      <table>
        <thead><tr><th>Model</th><th>Gates</th><th>Long-term Memory</th><th>Speed</th></tr></thead>
        <tbody>
          <tr><td>RNN</td><td>None</td><td>Weak</td><td>Fast</td></tr>
          <tr><td>LSTM</td><td>3 (forget, input, output)</td><td>Strong</td><td>Slow</td></tr>
          <tr><td>GRU</td><td>2 (reset, update)</td><td>Good</td><td>Medium</td></tr>
        </tbody>
      </table>

      <h3>4. Keras LSTM — Time Series Prediction</h3>
      <pre><code">import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

t      = np.linspace(0, 100, 1000)
series = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len]); y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y   = create_sequences(series, 20)
X      = X.reshape(*X.shape, 1)
split  = int(len(X) * 0.8)
X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(20, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_tr, y_tr, epochs=30, batch_size=32, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

y_pred = model.predict(X_te).flatten()
print(f"Test MSE: {np.mean((y_te - y_pred)**2):.4f}")</code></pre>

      <h3>5. Text Classification with LSTM</h3>
      <pre><code">from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test  = keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=200)

model = keras.Sequential([
    keras.layers.Embedding(10000, 64, input_length=200),
    keras.layers.LSTM(64, dropout=0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")</code></pre>

      <h3>Summary</h3>
      <table>
        <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
        <tbody>
          <tr><td>RNN</td><td>Hidden state carries sequential memory — but struggles with long sequences</td></tr>
          <tr><td>Vanishing Gradient</td><td>Gradients fade over long sequences — solved by LSTM/GRU</td></tr>
          <tr><td>LSTM</td><td>Cell state + 3 gates = strong long-term dependency learning</td></tr>
          <tr><td>GRU</td><td>2 gates, faster than LSTM, nearly equal performance</td></tr>
          <tr><td>return_sequences=True</td><td>Required in first layer when stacking LSTM layers</td></tr>
          <tr><td>Embedding layer</td><td>Maps words to dense vectors for text classification</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Transfer Learning: Fast Image Classification with Pre-trained Models",
    description: "Use ImageNet pre-trained VGG16 and MobileNetV2 for feature extraction and fine-tuning — achieve high accuracy with minimal data using Keras applications and data augmentation.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 12,
    slug: "nn-transfer-learning",
    content: `
      <h3>1. What is Transfer Learning & Why Use It</h3>
      <p>A model pre-trained on ImageNet (1.2M images, 1000 classes) already knows edges, textures, and shapes. Transfer learning <strong>reuses those learned features</strong> for a new task — no need to learn from scratch.</p>
      <pre><code># Two approaches:
#
# 1. Feature Extraction (Frozen Base):
#    - Freeze all pre-trained weights
#    - Train only new top layers
#    - Best for small datasets
#
# 2. Fine-Tuning (Partial Unfreeze):
#    - Unfreeze last few layers of base model
#    - Re-train entire model at very low learning rate
#    - Better accuracy when you have more data</code></pre>
      <table>
        <thead><tr><th>Model</th><th>ImageNet Acc</th><th>Parameters</th><th>Speed</th><th>Best For</th></tr></thead>
        <tbody>
          <tr><td>VGG16</td><td>92.7%</td><td>138M</td><td>Slow</td><td>Research, feature extraction</td></tr>
          <tr><td>ResNet50</td><td>93.0%</td><td>25.6M</td><td>Medium</td><td>General purpose</td></tr>
          <tr><td>MobileNetV2</td><td>91.0%</td><td>3.4M</td><td>Fast</td><td>Mobile, edge devices</td></tr>
          <tr><td>EfficientNetB0</td><td>93.3%</td><td>5.3M</td><td>Fast</td><td>Best efficiency</td></tr>
        </tbody>
      </table>

      <h3>2. Feature Extraction — Frozen Base</h3>
      <pre><code">import tensorflow as tf
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

base_model = keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,    # drop ImageNet classifier head
    weights='imagenet',   # pre-trained weights
)
base_model.trainable = False   # freeze all layers

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),  # feature map → vector
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax'),  # 10 CIFAR classes
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history1 = model.fit(X_train, y_train, epochs=5, batch_size=64,
                     validation_data=(X_test, y_test))
print(f"Feature extraction accuracy: {max(history1.history['val_accuracy']):.4f}")</code></pre>

      <h3>3. Fine-Tuning</h3>
      <pre><code">base_model.trainable = True
# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Very low learning rate — don't destroy pre-trained features
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    X_train, y_train, epochs=10, batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)
print(f"Fine-tuned accuracy: {max(history2.history['val_accuracy']):.4f}")</code></pre>

      <h3>4. Data Augmentation</h3>
      <pre><code">data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
])

# Add augmentation as first layer in the model
aug_model = keras.Sequential([
    data_augmentation,
    keras.applications.MobileNetV2(input_shape=(32,32,3),
                                   include_top=False, weights='imagenet'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation='softmax'),
])
aug_model.layers[1].trainable = False  # freeze base</code></pre>

      <h3>5. Scratch CNN vs Transfer Learning</h3>
      <pre><code">scratch = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h_scratch = scratch.fit(X_train, y_train, epochs=10, batch_size=64,
                        validation_data=(X_test, y_test), verbose=0)

print(f"Scratch CNN:          {max(h_scratch.history['val_accuracy']):.4f}")
print(f"Feature Extraction:   {max(history1.history['val_accuracy']):.4f}")
print(f"Fine-Tuned:           {max(history2.history['val_accuracy']):.4f}")</code></pre>

      <h3>Summary</h3>
      <table>
        <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
        <tbody>
          <tr><td>Feature Extraction</td><td>Freeze base, train only top layers — fast, works with little data</td></tr>
          <tr><td>Fine-Tuning</td><td>Partially unfreeze base at low lr — better accuracy, needs more data</td></tr>
          <tr><td>include_top=False</td><td>Drop the 1000-class head, add your own output layer</td></tr>
          <tr><td>GlobalAveragePooling2D</td><td>Better than Flatten — less overfitting</td></tr>
          <tr><td>Data Augmentation</td><td>Essential with small datasets — adds variation without new images</td></tr>
          <tr><td>MobileNetV2</td><td>Best default choice for production/mobile — small, fast, accurate</td></tr>
        </tbody>
      </table>
    `,
  },
];
