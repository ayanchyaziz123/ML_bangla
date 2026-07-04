export const nn_3_numpy = {
  title: "NumPy দিয়ে নিউরাল নেটওয়ার্ক Scratch থেকে",
  description: "শুধু NumPy ব্যবহার করে class-based 2-layer neural network সম্পূর্ণরূপে implement করা — forward pass, backward pass, training loop এবং XOR সমস্যা সমাধান।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 13,
  slug: "nn-numpy-from-scratch",
  content: `
    <h3>১. কেন Scratch থেকে শেখা দরকার?</h3>
    <p>TensorFlow বা PyTorch সব কিছু automatic করে দেয়। কিন্তু scratch থেকে implement না করলে কখনো বুঝবেন না কেন model কাজ করছে না বা কীভাবে debug করতে হয়।</p>
    <p>এই ব্লগে আমরা:</p>
    <ul>
      <li>শুধু NumPy ব্যবহার করব — কোনো ML framework নেই</li>
      <li>Object-oriented design: NeuralNetwork class</li>
      <li>Binary classification: make_circles dataset</li>
      <li>Complete: init → forward → loss → backward → train → predict</li>
    </ul>

    <h3>২. Dataset তৈরি ও Initialization</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Dataset: 2D circles (non-linearly separable)
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
y = y.reshape(-1, 1)  # (500,) -> (500, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape, y_train.shape)  # (400, 2) (400, 1)
print("Test set:", X_test.shape, y_test.shape)         # (100, 2) (100, 1)
print("Class distribution:", np.bincount(y_train.flatten().astype(int)))

# Data visualization
plt.figure(figsize=(6, 5))
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='red', label='Class 1', alpha=0.6)
plt.title("Make Circles Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("circles_dataset.png")
plt.show()
</code></pre>

    <h3>৩. NeuralNetwork Class Implementation</h3>
    <pre><code>class NeuralNetwork:
    """
    2-layer neural network: input -> hidden (relu) -> output (sigmoid)
    Architecture: n_features -> n_hidden -> 1
    """

    def __init__(self, n_features, n_hidden, learning_rate=0.01, random_state=42):
        np.random.seed(random_state)
        self.lr = learning_rate
        self.n_features = n_features
        self.n_hidden = n_hidden

        # He initialization: relu activation এর জন্য উপযুক্ত
        # std = sqrt(2 / n_inputs)
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2 / n_features)
        self.b1 = np.zeros((1, n_hidden))

        # Xavier initialization: sigmoid/tanh activation এর জন্য
        # std = sqrt(1 / n_inputs)
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(1 / n_hidden)
        self.b2 = np.zeros((1, 1))

        # Training history
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid"""
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    @staticmethod
    def sigmoid_derivative(a):
        """Sigmoid derivative in terms of activation output"""
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1    # (n, n_hidden)
        self.A1 = self.relu(self.Z1)               # (n, n_hidden)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (n, 1)
        self.A2 = self.sigmoid(self.Z2)                # (n, 1)

        return self.A2

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        n = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, X, y):
        """Backpropagation — compute gradients"""
        n = X.shape[0]

        # Output layer: dL/dZ2 = A2 - y (BCE + sigmoid combined)
        dZ2 = self.A2 - y                                  # (n, 1)
        self.dW2 = np.dot(self.A1.T, dZ2) / n             # (n_hidden, 1)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / n  # (1, 1)

        # Hidden layer: chain rule through relu
        dA1 = np.dot(dZ2, self.W2.T)                          # (n, n_hidden)
        dZ1 = dA1 * self.relu_derivative(self.Z1)              # (n, n_hidden)
        self.dW1 = np.dot(X.T, dZ1) / n                       # (n_features, n_hidden)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / n     # (1, n_hidden)

    def update_weights(self):
        """Gradient descent update"""
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
</code></pre>

    <h3>৪. Training Loop</h3>
    <pre><code>    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1000, batch_size=32, verbose=True):
        """
        Mini-batch gradient descent training
        """
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute batch loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss

                # Backward pass
                self.backward(X_batch, y_batch)

                # Weight update
                self.update_weights()

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            # Validation loss
            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_losses.append(val_loss)

            if verbose and epoch % 100 == 0:
                train_acc = self.accuracy(X_train, y_train)
                val_acc = self.accuracy(X_val, y_val) if X_val is not None else None
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.3f}" +
                      (f" | Val Acc: {val_acc:.3f}" if val_acc else ""))

        return self
</code></pre>

    <h3>৫. মডেল Train ও Evaluate করা</h3>
    <pre><code># Model তৈরি ও training
model = NeuralNetwork(
    n_features=2,
    n_hidden=16,
    learning_rate=0.05,
    random_state=42
)

model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=1000,
    batch_size=32,
    verbose=True
)

# Evaluation
train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)
print(f"\nFinal Training Accuracy: {train_acc:.4f}")
print(f"Final Test Accuracy:     {test_acc:.4f}")

# Loss curve plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(model.train_losses, label='Train Loss', color='blue')
plt.plot(model.val_losses, label='Val Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# Decision boundary visualization
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdBu', edgecolors='k', s=20)
plt.title("Learned Decision Boundary")

plt.tight_layout()
plt.savefig("nn_results.png")
plt.show()
</code></pre>

    <h3>৬. Weight Initialization-এর গুরুত্ব</h3>
    <pre><code># ভুল initialization (সব zero)
model_zeros = NeuralNetwork(n_features=2, n_hidden=8, learning_rate=0.05)
model_zeros.W1 = np.zeros((2, 8))  # BAD: symmetry problem
model_zeros.W2 = np.zeros((8, 1))
model_zeros.train(X_train, y_train, epochs=500, verbose=False)
print("Zero init accuracy:", model_zeros.accuracy(X_test, y_test))  # ~0.5 (random!)

# সঠিক initialization (He init)
model_he = NeuralNetwork(n_features=2, n_hidden=8, learning_rate=0.05)
# default He init
model_he.train(X_train, y_train, epochs=500, verbose=False)
print("He init accuracy:", model_he.accuracy(X_test, y_test))  # ~0.95+

# Symmetry problem: সব weight same হলে সব neurons same gradient পাবে
# ফলে সব neurons same জিনিস শিখবে — hidden layer useless হয়ে যাবে
</code></pre>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা NumPy দিয়ে সম্পূর্ণ neural network implement করলাম:</p>
    <table>
      <thead><tr><th>Component</th><th>Function</th><th>Key Detail</th></tr></thead>
      <tbody>
        <tr><td>__init__</td><td>Weight initialization</td><td>He init for ReLU, Xavier for sigmoid</td></tr>
        <tr><td>forward()</td><td>Z1→A1→Z2→A2</td><td>ReLU hidden, sigmoid output</td></tr>
        <tr><td>compute_loss()</td><td>Binary cross-entropy</td><td>clip for numerical stability</td></tr>
        <tr><td>backward()</td><td>Chain rule gradients</td><td>dZ2=A2-y, chain through ReLU</td></tr>
        <tr><td>update_weights()</td><td>Gradient descent</td><td>w -= lr * dw</td></tr>
        <tr><td>train()</td><td>Mini-batch loop</td><td>shuffle, batch, forward, backward, update</td></tr>
      </tbody>
    </table>
    <p>পরবর্তী ব্লগে আমরা PyTorch দিয়ে একই কাজ অনেক সহজে করব এবং MNIST dataset-এ handwritten digit classification করব।</p>
  `,
};
