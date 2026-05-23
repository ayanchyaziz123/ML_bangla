export const nn_1_perceptron = {
  title: "পার্সেপট্রন ও নিউরাল নেটওয়ার্কের গণিত",
  description: "জৈবিক নিউরন থেকে পার্সেপট্রন, অ্যাক্টিভেশন ফাংশন, XOR সমস্যা এবং মাল্টি-লেয়ার পার্সেপট্রনের মূল ধারণা বিস্তারিতভাবে আলোচনা।",
  date: "২৩ মে, ২০২৬",
  category: "নিউরাল নেটওয়ার্ক",
  readTime: 12,
  slug: "nn-perceptron-math",
  content: `
    <h3>১. জৈবিক নিউরন থেকে কৃত্রিম নিউরন</h3>
    <p>মানুষের মস্তিষ্কে প্রায় ৮৬ বিলিয়ন নিউরন আছে। প্রতিটি নিউরন অন্য নিউরন থেকে সংকেত গ্রহণ করে, প্রক্রিয়া করে এবং আউটপুট পাঠায়। কৃত্রিম নিউরাল নেটওয়ার্ক এই প্রক্রিয়াটিকেই গাণিতিকভাবে অনুকরণ করে।</p>
    <p>জৈবিক নিউরনের সাথে কৃত্রিম নিউরনের তুলনা:</p>
    <table>
      <thead><tr><th>জৈবিক নিউরন</th><th>কৃত্রিম নিউরন</th><th>ভূমিকা</th></tr></thead>
      <tbody>
        <tr><td>ডেনড্রাইট (Dendrite)</td><td>ইনপুট (x₁, x₂, ...)</td><td>সংকেত গ্রহণ করে</td></tr>
        <tr><td>সিন্যাপস (Synapse)</td><td>ওজন (w₁, w₂, ...)</td><td>সংকেতের শক্তি নির্ধারণ করে</td></tr>
        <tr><td>সেল বডি (Cell body)</td><td>যোগফল + অ্যাক্টিভেশন</td><td>সংকেত প্রক্রিয়া করে</td></tr>
        <tr><td>অ্যাক্সন (Axon)</td><td>আউটপুট (ŷ)</td><td>ফলাফল পাঠায়</td></tr>
      </tbody>
    </table>
    <p>একটি কৃত্রিম নিউরন মূলত দুটি ধাপে কাজ করে: প্রথমে weighted sum হিসাব করে, তারপর একটি অ্যাক্টিভেশন ফাংশন প্রয়োগ করে।</p>

    <h3>২. পার্সেপট্রনের গণিত</h3>
    <p>পার্সেপট্রন হলো সবচেয়ে সহজ কৃত্রিম নিউরন। এর গাণিতিক সূত্র:</p>
    <p><strong>z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b</strong></p>
    <p><strong>output = activation(z)</strong></p>
    <p>এখানে:</p>
    <ul>
      <li><strong>x₁, x₂, ..., xₙ</strong> — ইনপুট ফিচার</li>
      <li><strong>w₁, w₂, ..., wₙ</strong> — প্রতিটি ইনপুটের ওজন (weight)</li>
      <li><strong>b</strong> — বায়াস (bias) — থ্রেশহোল্ড শিফট করতে</li>
      <li><strong>z</strong> — weighted sum বা pre-activation</li>
      <li><strong>activation(z)</strong> — নন-লিনিয়ারিটি যোগ করে</li>
    </ul>
    <p>ম্যাট্রিক্স আকারে: <strong>z = Xᵀw + b</strong> যেখানে X এবং w হলো ভেক্টর।</p>
    <pre><code>import numpy as np

# একটি সহজ পার্সেপট্রন
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.01):
        # Random weight initialization
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, X):
        # Weighted sum: z = X·w + b
        z = np.dot(X, self.weights) + self.bias
        # Step activation function
        return np.where(z >= 0, 1, 0)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                # Weight update rule: w = w + lr * error * x
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                errors += int(error != 0)
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break
        return self

# AND gate উদাহরণ
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.train(X, y, epochs=100)
print("Predictions:", p.predict(X))  # [0, 0, 0, 1]
</code></pre>

    <h3>৩. অ্যাক্টিভেশন ফাংশন</h3>
    <p>অ্যাক্টিভেশন ফাংশন নেটওয়ার্কে নন-লিনিয়ারিটি যোগ করে। এটি ছাড়া যত লেয়ারই থাকুক, পুরো নেটওয়ার্ক একটি linear transformation হয়ে যাবে।</p>

    <h3>৩.১ Sigmoid</h3>
    <p>সূত্র: <strong>σ(z) = 1 / (1 + e⁻ᶻ)</strong></p>
    <p>আউটপুট রেঞ্জ: 0 থেকে 1। Binary classification-এর output layer-এ ব্যবহৃত হয়।</p>
    <pre><code>def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

z_values = np.linspace(-6, 6, 100)
print("sigmoid(0) =", sigmoid(0))   # 0.5
print("sigmoid(2) =", sigmoid(2))   # ~0.88
print("sigmoid(-2) =", sigmoid(-2)) # ~0.12
</code></pre>

    <h3>৩.২ ReLU (Rectified Linear Unit)</h3>
    <p>সূত্র: <strong>ReLU(z) = max(0, z)</strong></p>
    <p>Hidden layer-এ সবচেয়ে বেশি ব্যবহৃত। Vanishing gradient সমস্যা সমাধান করে। Computationally সস্তা।</p>
    <pre><code>def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Leaky ReLU: negative-এর জন্য small slope
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
</code></pre>

    <h3>৩.৩ Tanh ও Softmax</h3>
    <p>Tanh সূত্র: <strong>tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)</strong> — আউটপুট -1 থেকে 1।</p>
    <p>Softmax সূত্র: <strong>softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ</strong> — multi-class output layer-এ ব্যবহৃত।</p>
    <pre><code>def tanh(z):
    return np.tanh(z)

def softmax(z):
    # Numerically stable version
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# উদাহরণ: 3-class classification
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("Softmax:", probs)         # [0.659, 0.242, 0.099]
print("Sum:", np.sum(probs))     # 1.0
</code></pre>
    <table>
      <thead><tr><th>ফাংশন</th><th>সূত্র</th><th>রেঞ্জ</th><th>ব্যবহার</th><th>সমস্যা</th></tr></thead>
      <tbody>
        <tr><td>Sigmoid</td><td>1/(1+e⁻ᶻ)</td><td>0 to 1</td><td>Binary output</td><td>Vanishing gradient</td></tr>
        <tr><td>ReLU</td><td>max(0,z)</td><td>0 to ∞</td><td>Hidden layers</td><td>Dying ReLU</td></tr>
        <tr><td>Tanh</td><td>(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)</td><td>-1 to 1</td><td>Hidden (RNN)</td><td>Vanishing gradient</td></tr>
        <tr><td>Softmax</td><td>eᶻⁱ/Σeᶻʲ</td><td>0 to 1</td><td>Multi-class output</td><td>—</td></tr>
        <tr><td>Leaky ReLU</td><td>max(αz,z)</td><td>-∞ to ∞</td><td>Hidden layers</td><td>α tuning</td></tr>
      </tbody>
    </table>

    <h3>৪. XOR সমস্যা ও MLP-র প্রয়োজনীয়তা</h3>
    <p>একটি single perceptron শুধু linearly separable সমস্যা সমাধান করতে পারে। XOR gate linearly separable নয়, তাই single perceptron XOR সমাধান করতে পারে না।</p>
    <pre><code># XOR: single perceptron কাজ করে না
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])  # XOR output

p = Perceptron(n_inputs=2, learning_rate=0.1)
p.train(X_xor, y_xor, epochs=1000)
print("XOR Predictions:", p.predict(X_xor))
# কখনো [0,1,1,0] পাওয়া সম্ভব না!

# Solution: Multi-Layer Perceptron (MLP)
# Hidden layer দিয়ে XOR solve করা যায়
# AND, OR, NOT gate combine করলে XOR হয়
</code></pre>
    <p>XOR সমস্যা সমাধানের জন্য আমাদের একটি hidden layer দরকার। Hidden layer নন-লিনিয়ার decision boundary তৈরি করতে পারে।</p>

    <h3>৫. মাল্টি-লেয়ার পার্সেপট্রন (MLP)</h3>
    <p>MLP-তে তিন ধরনের লেয়ার থাকে:</p>
    <ul>
      <li><strong>Input Layer</strong>: ফিচার গ্রহণ করে, কোনো computation নেই</li>
      <li><strong>Hidden Layer(s)</strong>: নন-লিনিয়ার pattern শিখে, ReLU সাধারণত ব্যবহৃত হয়</li>
      <li><strong>Output Layer</strong>: চূড়ান্ত prediction, task অনুযায়ী activation</li>
    </ul>
    <pre><code>import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    """
    2-layer MLP forward pass
    X: input (n_samples, n_features)
    W1: (n_features, n_hidden)
    W2: (n_hidden, n_output)
    """
    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  # binary classification

    return A1, A2

# Architecture: 2 inputs -> 4 hidden -> 1 output
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.01
b2 = np.zeros((1, 1))

X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
A1, A2 = mlp_forward(X_xor, W1, b1, W2, b2)
print("Output shape:", A2.shape)  # (4, 1)
print("Predictions (before training):", A2.flatten())
</code></pre>
    <p>এই কোডে ওজনগুলো random, তাই training ছাড়া correct output পাওয়া যাবে না। পরের ব্লগে আমরা backpropagation দিয়ে ওজন আপডেট করা শিখব।</p>

    <h3>৬. সারসংক্ষেপ</h3>
    <p>এই ব্লগে আমরা শিখলাম:</p>
    <ul>
      <li>জৈবিক নিউরন এবং কৃত্রিম নিউরনের মধ্যে সম্পর্ক</li>
      <li>পার্সেপট্রনের গণিত: z = Xᵀw + b, output = activation(z)</li>
      <li>Sigmoid, ReLU, Tanh, Softmax অ্যাক্টিভেশন ফাংশন</li>
      <li>XOR সমস্যা: single perceptron-এর সীমাবদ্ধতা</li>
      <li>MLP: multiple layer দিয়ে জটিল সমস্যা সমাধান</li>
    </ul>
    <p>পরবর্তী ব্লগে আমরা Backpropagation — নিউরাল নেটওয়ার্কের শেখার প্রক্রিয়া — বিস্তারিত দেখব।</p>
  `,
};
