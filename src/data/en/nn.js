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
    title: "PyTorch Neural Networks",
    description: "PyTorch fundamentals for building neural networks with nn.Module, MNIST handwritten digit classification, manual training loops, accuracy/loss curves, and model save/load.",
    date: "২৩ মে, ২০২৬",
    category: "Neural Network",
    readTime: 12,
    slug: "nn-keras-tensorflow",
    content: `
      <h3>1. Why PyTorch? From NumPy to PyTorch</h3>
      <p>The previous post needed 300+ lines to build a 2-layer network. PyTorch does the same in far fewer lines while still giving you full control over the training loop. Benefits: automatic GPU support via <code>.to(device)</code>, automatic differentiation (autograd), built-in optimizers and losses, and a training loop that is just plain Python — easy to read and debug.</p>
      <pre><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

torch.manual_seed(42)
np.random.seed(42)
</code></pre>

      <h3>2. MNIST Dataset: Handwritten Digits</h3>
      <p>MNIST is the "Hello World" of ML — 70,000 handwritten digit images (0-9), each 28×28 grayscale pixels.</p>
      <pre><code>transform = transforms.Compose([transforms.ToTensor()])   # scales pixels to 0-1

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Training:", len(train_dataset))  # 60000
print("Test:",     len(test_dataset))   # 10000

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)      # (128, 1, 28, 28)
print("Label sample:", labels[:10].tolist())
</code></pre>

      <h3>3. Defining the Model with nn.Module</h3>
      <p>In PyTorch, a model is a Python class: layers are declared in <code>__init__</code> and the forward computation is written explicitly in <code>forward()</code>.</p>
      <pre><code>class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)    # 10 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)              # raw logits — softmax is inside the loss

model = SimpleMLP().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print("Total params:", total_params)    # 109,386
# fc1: 784*128 + 128 = 100480
# fc2: 128*64 + 64   = 8256
# fc3: 64*10 + 10    = 650

criterion = nn.CrossEntropyLoss()       # combines softmax + NLL, takes integer labels directly
optimizer = optim.Adam(model.parameters(), lr=0.001)
</code></pre>

      <h3>4. Training: the Manual Training Loop</h3>
      <p>PyTorch has no <code>.fit()</code> — instead you write the loop yourself: forward pass, compute loss, backward pass, update weights.</p>
      <pre><code>def evaluate(model, loader):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total

best_val_loss = float('inf')
patience, patience_counter = 5, 0
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_loss, train_acc = running_loss / total, correct / total
    val_loss, val_acc = evaluate(model, test_loader)
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss &lt; best_val_loss:               # like ModelCheckpoint(save_best_only=True)
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_mnist_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:       # like EarlyStopping(patience=5)
            print(f"Early stopping at epoch {epoch+1}")
            break

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")   # ~0.98

model.eval()
sample_x, sample_y = next(iter(test_loader))
with torch.no_grad():
    logits = model(sample_x[:5].to(device))
    y_pred = logits.argmax(1).cpu()
print("Predicted:", y_pred.tolist())
print("Actual:   ", sample_y[:5].tolist())
</code></pre>

      <h3>5. Training Curves & Confusion Matrix</h3>
      <pre><code>def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_acc'], label='Train', color='blue')
    axes[0].plot(history['val_acc'],   label='Val',   color='orange')
    axes[0].set_title('Model Accuracy'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_loss'], label='Train', color='blue')
    axes[1].plot(history['val_loss'],   label='Val',   color='orange')
    axes[1].set_title('Model Loss'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

plot_training_history(history)

from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu()
        all_preds.append(preds); all_labels.append(yb)
all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix — MNIST Test Set")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.savefig("confusion_matrix.png"); plt.show()
</code></pre>

      <h3>6. Model Save & Load</h3>
      <pre><code>torch.save(model.state_dict(), 'mnist_model.pt')     # weights only (recommended)
torch.save(model, 'mnist_model_full.pt')             # entire model object (less portable)

loaded_model = SimpleMLP().to(device)
loaded_model.load_state_dict(torch.load('mnist_model.pt'))
loaded_model.eval()

test_loss, test_acc = evaluate(loaded_model, test_loader)
print(f"Loaded model accuracy: {test_acc:.4f}")

print(loaded_model)   # architecture overview

single_image, _ = test_dataset[0]
single_image = single_image.unsqueeze(0).to(device)   # add batch dim: (1, 1, 28, 28)
with torch.no_grad():
    logits = loaded_model(single_image)
    probs = torch.softmax(logits, dim=1)
    predicted_class = probs.argmax(1).item()
    confidence = probs[0][predicted_class].item()
print(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
</code></pre>
      <table>
        <thead><tr><th>Concept</th><th>Purpose</th><th>Returns</th></tr></thead>
        <tbody>
          <tr><td>training loop</td><td>Training</td><td>history dict (loss, accuracy per epoch) — built manually</td></tr>
          <tr><td>evaluate() (custom)</td><td>Test set performance</td><td>(loss, accuracy)</td></tr>
          <tr><td>model(x) under torch.no_grad()</td><td>Raw predictions</td><td>logits tensor</td></tr>
          <tr><td>print(model)</td><td>Architecture overview</td><td>Print to stdout</td></tr>
          <tr><td>torch.save(model.state_dict(), path)</td><td>Persistence</td><td>File on disk</td></tr>
          <tr><td>model.load_state_dict(torch.load(path))</td><td>Load saved weights</td><td>Populates model in place</td></tr>
        </tbody>
      </table>

      <h3>7. Summary</h3>
      <ul>
        <li>nn.Module: define layers in __init__, computation in forward()</li>
        <li>criterion + optimizer: replace compile()'s loss and optimizer arguments</li>
        <li>Manual training loop: zero_grad() → forward → loss → backward() → step()</li>
        <li>MNIST: ToTensor → DataLoader → Linear layers → CrossEntropyLoss → ~98% accuracy</li>
        <li>Manual early stopping: track best val_loss, save checkpoint, stop after patience runs out</li>
        <li>Model save/load: state_dict for production deployment</li>
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
      <pre><code>import torch
import torch.nn as nn
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
      <pre><code>from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])   # (0-255) -> (0-1), shape (1,28,28)
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

class CNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool  = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)     # 7*7*64 = 3136
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))     # 14x14 -> 7x7
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)                          # raw logits

cnn_model = CNNMnist()
print(cnn_model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
</code></pre>

      <h3>5. Training & MLP vs CNN Comparison</h3>
      <pre><code>device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model.to(device)

for epoch in range(10):
    cnn_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn_model(xb), yb)
        loss.backward()
        optimizer.step()

cnn_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = cnn_model(xb).argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
cnn_acc = correct / total
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
      <pre><code>from torchvision import datasets as tv_datasets, transforms as tv_transforms
import torchvision.models as models

cifar_transform = tv_transforms.Compose([tv_transforms.ToTensor()])
cifar_train = tv_datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
base_model = vgg16.features            # convolutional feature extractor only
for p in base_model.parameters():
    p.requires_grad = False            # Freeze pre-trained weights

class TransferModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)  # VGG16 features on 32x32 input -> (512, 1, 1)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

transfer_model = TransferModel(base_model)
print("Transfer learning: VGG16 pre-trained on 1.2M ImageNet images!")
</code></pre>

      <h3>7. Summary</h3>
      <ul>
        <li>CNN intuition: local patterns → hierarchical features</li>
        <li>nn.Conv2d: filters, feature maps, padding, stride</li>
        <li>nn.MaxPool2d: spatial downsampling, translation invariance</li>
        <li>MNIST CNN: 99%+ accuracy vs MLP ~98%</li>
        <li>Transfer learning: fine-tune ImageNet pre-trained VGG16, ResNet, EfficientNet from torchvision.models</li>
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
      <pre><code>import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)   # raw logit — BCEWithLogitsLoss applies sigmoid internally
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

def build_overfit_model():
    return OverfitModel()

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                 lr=0.001, weight_decay=0.0):
    """Manual training loop — the PyTorch equivalent of model.fit()"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'accuracy': [], 'val_accuracy': []}
    n = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = ((torch.sigmoid(model(X_train)) >= 0.5).float() == y_train).float().mean().item()
            val_acc   = ((torch.sigmoid(model(X_val))   >= 0.5).float() == y_val).float().mean().item()
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

    return history

baseline = build_overfit_model()
history_overfit = train_model(baseline, X_train, y_train, X_val, y_val, epochs=100)
print(f"Train: {history_overfit['accuracy'][-1]:.3f}",
      f"Val: {history_overfit['val_accuracy'][-1]:.3f}")
# Train: 0.99, Val: 0.72 — OVERFITTING!
</code></pre>

      <h3>2. Dropout: Randomly Disabling Neurons</h3>
      <p>Dropout randomly zeros out neuron outputs during training. It forces neurons to learn independently without co-adapting. At inference all neurons are active but outputs are scaled by (1 - rate).</p>
      <pre><code>class DropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.fc5(x)

def build_dropout_model(dropout_rate=0.3):
    return DropoutModel(dropout_rate)

dropout_model = build_dropout_model(dropout_rate=0.3)
history_dropout = train_model(dropout_model, X_train, y_train, X_val, y_val, epochs=100)
print(f"Dropout — Train: {history_dropout['accuracy'][-1]:.3f}",
      f"Val: {history_dropout['val_accuracy'][-1]:.3f}")
# Train: 0.88, Val: 0.85 — gap much smaller!
</code></pre>

      <h3>3. Batch Normalization</h3>
      <p>Batch Normalization normalizes each layer's inputs (mean≈0, std≈1). It accelerates training, allows higher learning rates, and has a slight regularization effect.</p>
      <pre><code>class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

def build_batchnorm_model():
    return BatchNormModel()
# BN before activation generally performs slightly better
</code></pre>

      <h3>4. L1 & L2 Regularization</h3>
      <p>L2 (Ridge): <strong>Total Loss = CE + λ·Σw²</strong> — keeps weights small</p>
      <p>L1 (Lasso): <strong>Total Loss = CE + λ·Σ|w|</strong> — encourages sparse weights</p>
      <pre><code>class L2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def build_l2_model():
    return L2Model()

# L2: PyTorch applies it through the optimizer's weight_decay — no per-layer regularizer needed
# l2_model = build_l2_model()
# optimizer = torch.optim.Adam(l2_model.parameters(), lr=0.001, weight_decay=0.001)

# L1: PyTorch has no built-in flag — add the penalty manually inside the loss
def l1_penalty(model, l1_lambda=0.001):
    return l1_lambda * sum(p.abs().sum() for p in model.parameters())
# loss = criterion(model(xb), yb) + l1_penalty(model)   # add this inside the training loop
</code></pre>

      <h3>5. EarlyStopping & Learning Rate Scheduling</h3>
      <pre><code># Best-practice combined model
class BestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

best_model = BestModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)   # ReduceLROnPlateau

best_val_loss, patience, patience_counter = float('inf'), 10, 0
best_state = None

for epoch in range(200):
    best_model.train()
    perm = torch.randperm(X_train.shape[0])
    for i in range(0, X_train.shape[0], 32):
        idx = perm[i:i+32]
        optimizer.zero_grad()
        loss = criterion(best_model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

    best_model.eval()
    with torch.no_grad():
        val_loss = criterion(best_model(X_val), y_val).item()
    scheduler.step(val_loss)

    if val_loss &lt; best_val_loss - 0.001:      # min_delta=0.001
        best_val_loss = val_loss
        best_state = best_model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:      # EarlyStopping(patience=10)
            print(f"Early stopping at epoch {epoch+1}")
            break

best_model.load_state_dict(best_state)        # restore_best_weights=True
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
      <pre><code>import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_overfit['val_accuracy'], label='Baseline', color='red')
axes[0].plot(history_dropout['val_accuracy'], label='Dropout(0.3)', color='blue')
axes[0].set_title("Validation Accuracy"); axes[0].legend(); axes[0].grid(True)

gap_baseline = np.array(history_overfit['accuracy']) - np.array(history_overfit['val_accuracy'])
gap_dropout  = np.array(history_dropout['accuracy'])  - np.array(history_dropout['val_accuracy'])
axes[1].plot(gap_baseline, label='Baseline Gap', color='red')
axes[1].plot(gap_dropout,  label='Dropout Gap',  color='blue')
axes[1].axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')
axes[1].set_title("Train-Val Gap"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.savefig("regularization_comparison.png"); plt.show()
</code></pre>

      <h3>7. Summary</h3>
      <ul>
        <li>Dropout (nn.Dropout): disables random neurons, prevents co-adaptation</li>
        <li>Batch Normalization (nn.BatchNorm1d): normalize + accelerate training</li>
        <li>L2 via optimizer's weight_decay; L1 added manually to the loss — both keep weights small</li>
        <li>Manual early stopping: track best val_loss, save state_dict, restore it at the end</li>
        <li>torch.optim.lr_scheduler.ReduceLROnPlateau: reduces learning rate when training plateaus</li>
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
      <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

torch.manual_seed(42); np.random.seed(42)

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

transform = transforms.Compose([transforms.ToTensor()])
train_full = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
test_set   = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("Full training set:", len(train_full))
print("Test set:", len(test_set))
</code></pre>

      <h3>2. Data Exploration & Visualization</h3>
      <pre><code>fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_full[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(CLASS_NAMES[label], fontsize=9)
    ax.axis('off')
plt.suptitle("Fashion MNIST Sample Images")
plt.tight_layout(); plt.savefig("fashion_samples.png"); plt.show()

train_labels = train_full.targets.numpy()
class_counts = np.bincount(train_labels)
print("Class distribution (all balanced ~6000 each):")
for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {i}: {name:<15} — {count} samples")
</code></pre>

      <h3>3. Preprocessing</h3>
      <pre><code>X_train_full = train_full.data.float() / 255.0    # (60000, 28, 28)
y_train_full = train_full.targets
X_test       = test_set.data.float() / 255.0
y_test       = test_set.targets

val_size = 5000
X_val, y_val     = X_train_full[:val_size], y_train_full[:val_size]
X_train, y_train = X_train_full[val_size:], y_train_full[val_size:]
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train_flat = X_train.reshape(-1, 784)
X_val_flat   = X_val.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

X_train_cnn = X_train.unsqueeze(1)   # (55000, 1, 28, 28)
X_val_cnn   = X_val.unsqueeze(1)
X_test_cnn  = X_test.unsqueeze(1)

class EarlyStopper:
    """Tracks best val_loss and checkpoints the model — replaces EarlyStopping + ModelCheckpoint"""
    def __init__(self, model_name, patience=8):
        self.model_name = model_name
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss &lt; self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), f'{self.model_name}_best.pt')
            return False       # keep training
        self.counter += 1
        return self.counter >= self.patience   # True -> stop
</code></pre>

      <h3>4. Build & Train Three Models</h3>
      <pre><code>class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        return self.fc3(x)

def build_simple_mlp():
    return SimpleMLP()

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

def build_deep_mlp():
    return DeepMLP()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = self.flatten(x)
        x = self.dropout3(self.relu(self.bn3(self.fc1(x))))
        return self.fc2(x)

def build_cnn():
    return CNN()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate(name, build_fn, X_tr, X_v, X_te, weight_decay=0.0, epochs=50, batch_size=64):
    model = build_fn()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6)
    stopper = EarlyStopper(name, patience=8)

    train_loader = DataLoader(TensorDataset(X_tr, y_train), batch_size=batch_size, shuffle=True)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_val).item()
        scheduler.step(val_loss)
        if stopper.step(val_loss, model):
            break

    model.load_state_dict(torch.load(f'{name}_best.pt'))   # restore_best_weights
    t = time.time() - start

    model.eval()
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_test).float().mean().item()

    return {'model': model, 'test_acc': acc, 'n_params': count_params(model), 'train_time': t}

results = {}
for name, build_fn, X_tr, X_v, X_te, wd in [
    ('simple_mlp', build_simple_mlp, X_train_flat, X_val_flat, X_test_flat, 0.0),
    ('deep_mlp',   build_deep_mlp,   X_train_flat, X_val_flat, X_test_flat, 0.001),
    ('cnn',        build_cnn,        X_train_cnn,  X_val_cnn,  X_test_cnn,  0.0),
]:
    r = train_and_evaluate(name, build_fn, X_tr, X_v, X_te, weight_decay=wd)
    results[name] = r
    print(f"{name}: Params={r['n_params']:,} | Acc={r['test_acc']:.4f} | Time={r['train_time']:.1f}s")
</code></pre>

      <h3>5. Comparison & Confusion Matrix</h3>
      <pre><code>print("\n" + "="*60)
print(f"{'Model':<15} {'Params':>10} {'Test Acc':>10} {'Time':>8}")
print("="*60)
for name, r in results.items():
    print(f"{name:<15} {r['n_params']:>10,} {r['test_acc']:>10.4f} {r['train_time']:>7.1f}s")

best_model = results['cnn']['model']
best_model.eval()
with torch.no_grad():
    y_pred = best_model(X_test_cnn).argmax(1).numpy()
y_test_np = y_test.numpy()
cm = confusion_matrix(y_test_np, y_pred)

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
      <pre><code>print("\nPer-Class Accuracy (CNN):")
for i, class_name in enumerate(CLASS_NAMES):
    mask = (y_test_np == i)
    acc = np.mean(y_pred[mask] == y_test_np[mask])
    print(f"  {i}: {class_name:<15} — {acc:.3f} ({acc*100:.1f}%)")

print("\nMost Confused Pairs:")
cm_no_diag = cm.copy(); np.fill_diagonal(cm_no_diag, 0)
for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    print(f"  True: {CLASS_NAMES[idx[0]]:<15} -> Pred: {CLASS_NAMES[idx[1]]:<15} ({cm_no_diag[idx]} errors)")
    cm_no_diag[idx] = 0

print("\nClassification Report:")
print(classification_report(y_test_np, y_pred, target_names=CLASS_NAMES))
</code></pre>

      <h3>7. Best Practices Summary</h3>
      <table>
        <thead><tr><th>Best Practice</th><th>Why It Matters</th><th>Default Choice</th></tr></thead>
        <tbody>
          <tr><td>Normalize inputs</td><td>Stable gradients, faster convergence</td><td>0–1 or StandardScaler</td></tr>
          <tr><td>He/Xavier init</td><td>Symmetry breaking, gradient flow</td><td>PyTorch default (Kaiming uniform)</td></tr>
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
    description: "Recurrent Neural Networks for sequence data — hidden state mechanics, vanishing gradient problem, LSTM gates (forget/input/output), GRU, PyTorch time series prediction and IMDB sentiment analysis.",
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
# Each dh_t/dh_{t-1} &lt; 1  →  product approaches 0 (vanishing)
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

      <h3>4. PyTorch LSTM — Time Series Prediction</h3>
      <pre><code>import numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

t      = np.linspace(0, 100, 1000)
series = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len]); y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y  = create_sequences(series, 20)
X     = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)   # (samples, 20, 1)
y     = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
split = int(len(X) * 0.8)
X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)          # feed the full sequence into the next LSTM (like return_sequences=True)
        _, (h_n, _) = self.lstm2(x)   # only need the final hidden state now
        x = self.relu(self.fc1(h_n[-1]))
        return self.fc2(x)

model = LSTMForecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

n_val = int(len(X_tr) * 0.2)
X_val_split, y_val_split = X_tr[-n_val:], y_tr[-n_val:]
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
        val_loss = criterion(model(X_val_split), y_val_split).item()
    if val_loss &lt; best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:    # EarlyStopping(patience=5)
            break

model.load_state_dict(best_state)           # restore_best_weights=True

model.eval()
with torch.no_grad():
    y_pred = model(X_te).squeeze(-1).numpy()
print(f"Test MSE: {np.mean((y_te.numpy().flatten() - y_pred)**2):.4f}")</code></pre>

      <h3>5. Text Classification with LSTM</h3>
      <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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
    return ids + [vocab['&lt;pad&gt;']] * (max_len - len(ids))    # pad_sequences equivalent

def collate_batch(batch):
    labels = torch.tensor([1.0 if label == 'pos' else 0.0 for label, _ in batch]).unsqueeze(1)
    texts  = torch.tensor([encode(text) for _, text in batch])
    return texts, labels

train_loader = DataLoader(list(IMDB(split='train')), batch_size=128, shuffle=True,  collate_fn=collate_batch)
test_loader  = DataLoader(list(IMDB(split='test')),  batch_size=128, shuffle=False, collate_fn=collate_batch)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.dropout(self.relu(self.fc1(h_n[-1])))
        return self.fc2(x)              # raw logit — BCEWithLogitsLoss applies sigmoid internally

model = SentimentLSTM(vocab_size=len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = (torch.sigmoid(model(xb)) >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(f"Test Accuracy: {correct/total:.4f}")</code></pre>

      <h3>Summary</h3>
      <table>
        <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
        <tbody>
          <tr><td>RNN</td><td>Hidden state carries sequential memory — but struggles with long sequences</td></tr>
          <tr><td>Vanishing Gradient</td><td>Gradients fade over long sequences — solved by LSTM/GRU</td></tr>
          <tr><td>LSTM</td><td>Cell state + 3 gates = strong long-term dependency learning</td></tr>
          <tr><td>GRU</td><td>2 gates, faster than LSTM, nearly equal performance</td></tr>
          <tr><td>Stacking nn.LSTM layers</td><td>Feed the full sequence output of one LSTM into the next — no return_sequences flag needed</td></tr>
          <tr><td>Embedding layer</td><td>Maps words to dense vectors for text classification</td></tr>
        </tbody>
      </table>
    `,
  },
  {
    title: "Transfer Learning: Fast Image Classification with Pre-trained Models",
    description: "Use ImageNet pre-trained VGG16 and MobileNetV2 for feature extraction and fine-tuning — achieve high accuracy with minimal data using torchvision models and data augmentation.",
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
      <pre><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
base_model = base_model.features        # drop the ImageNet classifier head
for p in base_model.parameters():
    p.requires_grad = False             # freeze all layers

class TransferModel(nn.Module):
    def __init__(self, base, dropout_rate=0.3):
        super().__init__()
        self.base = base
        self.pool = nn.AdaptiveAvgPool2d(1)   # GlobalAveragePooling2D equivalent
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 256)       # MobileNetV2 features output 1280 channels
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)         # 10 CIFAR classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        x = self.flatten(self.pool(x))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = TransferModel(base_model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total

val_accs = []
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    val_accs.append(evaluate(model, test_loader))

print(f"Feature extraction accuracy: {max(val_accs):.4f}")</code></pre>

      <h3>3. Fine-Tuning</h3>
      <pre><code>for p in base_model.parameters():
    p.requires_grad = True              # unfreeze everything first

base_layers = list(base_model.children())
for layer in base_layers[:-30]:         # re-freeze all but the last 30 sub-modules
    for p in layer.parameters():
        p.requires_grad = False

# Very low learning rate — don't destroy pre-trained features
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

best_val_acc, patience, patience_counter, best_state = 0.0, 3, 0, None
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    val_acc = evaluate(model, test_loader)
    if val_acc > best_val_acc:
        best_val_acc, patience_counter = val_acc, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:   # EarlyStopping(patience=3)
            break

model.load_state_dict(best_state)          # restore_best_weights=True
print(f"Fine-tuned accuracy: {best_val_acc:.4f}")</code></pre>

      <h3>4. Data Augmentation</h3>
      <pre><code>train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),                     # RandomFlip("horizontal")
    transforms.RandomRotation(10),                          # RandomRotation(0.1)
    transforms.RandomResizedCrop(32, scale=(0.9, 1.0)),      # RandomZoom(0.1)
    transforms.ColorJitter(contrast=0.1),                    # RandomContrast(0.1)
    transforms.ToTensor(),
])

# Augmentation lives in the Dataset's transform — applied on-the-fly each epoch
aug_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
aug_train_loader = DataLoader(aug_train_set, batch_size=64, shuffle=True)

aug_base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
for p in aug_base.parameters():
    p.requires_grad = False    # freeze base

aug_model = TransferModel(aug_base, dropout_rate=0.4)</code></pre>

      <h3>5. Scratch CNN vs Transfer Learning</h3>
      <pre><code>class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 6 * 6, 128)   # 32->30->15->13->6 through conv/pool (no padding)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

scratch = ScratchCNN()
scratch_optimizer = torch.optim.Adam(scratch.parameters(), lr=0.001)

scratch_val_accs = []
for epoch in range(10):
    scratch.train()
    for xb, yb in train_loader:
        scratch_optimizer.zero_grad()
        loss = criterion(scratch(xb), yb)
        loss.backward()
        scratch_optimizer.step()
    scratch_val_accs.append(evaluate(scratch, test_loader))

print(f"Scratch CNN:          {max(scratch_val_accs):.4f}")
print(f"Feature Extraction:   {max(val_accs):.4f}")
print(f"Fine-Tuned:           {best_val_acc:.4f}")</code></pre>

      <h3>Summary</h3>
      <table>
        <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
        <tbody>
          <tr><td>Feature Extraction</td><td>Freeze base (requires_grad = False), train only top layers — fast, works with little data</td></tr>
          <tr><td>Fine-Tuning</td><td>Partially unfreeze base at low lr — better accuracy, needs more data</td></tr>
          <tr><td>base_model.features</td><td>Drop the 1000-class head, add your own output layer</td></tr>
          <tr><td>nn.AdaptiveAvgPool2d(1)</td><td>Better than Flatten — less overfitting (GlobalAveragePooling2D equivalent)</td></tr>
          <tr><td>Data Augmentation</td><td>Essential with small datasets — transforms in the Dataset add variation without new images</td></tr>
          <tr><td>MobileNetV2 (torchvision.models)</td><td>Best default choice for production/mobile — small, fast, accurate</td></tr>
        </tbody>
      </table>
    `,
  },
];
