export const stats_3_distributions = {
  slug: 'stats-3-distributions',
  title: 'Probability Distributions: ML-এ কোথায় কোনটি ব্যবহার হয়',
  description: 'Normal, Binomial, Poisson, Exponential, Beta, Dirichlet distributions — Central Limit Theorem, MLE, MAP estimation এবং প্রতিটি distribution-এর ML application।',
  date: 'মে ২০২৬',
  category: 'এমএল-এর জন্য পরিসংখ্যান',
  readTime: 13,
  content: `
    <h3>১. Normal (Gaussian) Distribution</h3>
    <p><strong>N(μ, σ²)</strong> — সবচেয়ে গুরুত্বপূর্ণ distribution। ML-এ everywhere:</p>
    <ul>
      <li>Linear Regression residuals assume normal distribution</li>
      <li>Neural network weight initialization (Xavier/He)</li>
      <li>Gaussian Naive Bayes</li>
      <li>Anomaly detection (z-score based)</li>
    </ul>
    <pre><code class="language-python">from scipy import stats
import numpy as np

# Normal distribution properties
mu, sigma = 170, 10  # Height in cm
dist = stats.norm(mu, sigma)

# Probabilities
print(f"P(X > 180) = {1 - dist.cdf(180):.4f}")
print(f"P(160 < X < 180) = {dist.cdf(180) - dist.cdf(160):.4f}")
print(f"68-95-99.7 rule: {dist.cdf(mu+sigma) - dist.cdf(mu-sigma):.4f}")

# MLE: Estimate μ and σ from data
data = np.random.normal(mu, sigma, 1000)
mu_mle, sigma_mle = stats.norm.fit(data)
print(f"\\nMLE estimates: μ={mu_mle:.2f}, σ={sigma_mle:.2f}")
</code></pre>

    <h3>২. Central Limit Theorem</h3>
    <p>ML-এ CLT অত্যন্ত গুরুত্বপূর্ণ: যেকোনো distribution থেকে sample mean normal হয় (large n-এ)। এটাই confidence intervals ও hypothesis tests-এর ভিত্তি।</p>
    <pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sample_sizes = [1, 5, 30]

for i, n in enumerate(sample_sizes):
    sample_means = [np.random.exponential(1, n).mean() for _ in range(5000)]
    for j, dist_name in enumerate(['Exponential', 'Uniform']):
        ax = axes[j][i]
        if dist_name == 'Exponential':
            sample_means = [np.random.exponential(1, n).mean() for _ in range(5000)]
        else:
            sample_means = [np.random.uniform(0, 1, n).mean() for _ in range(5000)]
        ax.hist(sample_means, bins=50, density=True, alpha=0.7)
        ax.set_title(f'{dist_name}, n={n}')

plt.suptitle('Central Limit Theorem: Any Distribution → Normal (as n grows)')
plt.tight_layout(); plt.show()
</code></pre>

    <h3>৩. MLE vs MAP Estimation</h3>
    <p><strong>MLE (Maximum Likelihood Estimation):</strong> Data-কে সবচেয়ে ভালো explain করে এমন parameters খোঁজা।<br/>
    <strong>MAP (Maximum A Posteriori):</strong> Prior knowledge সহ MLE — regularization হিসেবে কাজ করে।</p>
    <pre><code class="language-python">import numpy as np
from scipy.optimize import minimize

# Logistic Regression as MLE
def sigmoid(x): return 1 / (1 + np.exp(-x))

def mle_loss(w, X, y):
    """Negative log-likelihood (Cross-entropy)"""
    pred = sigmoid(X @ w)
    pred = np.clip(pred, 1e-10, 1-1e-10)
    return -np.mean(y * np.log(pred) + (1-y) * np.log(1-pred))

def map_loss(w, X, y, lambda_reg=0.01):
    """MLE + L2 regularization (MAP with Gaussian prior)"""
    return mle_loss(w, X, y) + lambda_reg * np.sum(w**2)

np.random.seed(42)
X = np.random.randn(200, 3)
true_w = np.array([1.5, -2.0, 0.5])
y = (sigmoid(X @ true_w) > 0.5).astype(float)
X = np.hstack([np.ones((200, 1)), X])  # Add bias

w0 = np.zeros(4)
mle_result = minimize(mle_loss, w0, args=(X, y), method='L-BFGS-B')
map_result = minimize(map_loss, w0, args=(X, y, 0.1), method='L-BFGS-B')

print(f"MLE weights: {mle_result.x.round(3)}")
print(f"MAP weights: {map_result.x.round(3)}")
print("MAP weights are smaller/regularized — equivalent to L2 regularization!")
</code></pre>

    <h3>৪. Key Distributions in ML</h3>
    <table>
      <thead><tr><th>Distribution</th><th>Use Case in ML</th><th>Parameters</th></tr></thead>
      <tbody>
        <tr><td>Normal</td><td>Regression errors, weight init</td><td>μ, σ²</td></tr>
        <tr><td>Bernoulli</td><td>Binary classification output</td><td>p</td></tr>
        <tr><td>Binomial</td><td>Count of successes</td><td>n, p</td></tr>
        <tr><td>Categorical</td><td>Multi-class softmax output</td><td>p₁...pₖ</td></tr>
        <tr><td>Poisson</td><td>Count data (rare events)</td><td>λ</td></tr>
        <tr><td>Beta</td><td>Prior for probability (Bayesian)</td><td>α, β</td></tr>
        <tr><td>Dirichlet</td><td>Prior for topic models (LDA)</td><td>α₁...αₖ</td></tr>
        <tr><td>Exponential</td><td>Time between events</td><td>λ</td></tr>
      </tbody>
    </table>
  `
};
