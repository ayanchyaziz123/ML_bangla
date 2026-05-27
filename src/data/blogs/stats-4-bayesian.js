export const stats_4_bayesian = {
  slug: 'stats-4-bayesian',
  title: 'Bayesian Statistics: Uncertainty Quantification',
  description: 'Prior, Likelihood, Posterior, Conjugate Priors, Credible Intervals, MCMC basics, এবং Bayesian Linear Regression — ML-এ uncertainty estimate করার সঠিক পদ্ধতি।',
  date: 'মে ২০২৬',
  category: 'এমএল-এর জন্য পরিসংখ্যান',
  readTime: 15,
  content: `
    <h3>১. Frequentist vs Bayesian</h3>
    <p>দুটি মৌলিকভাবে ভিন্ন দৃষ্টিভঙ্গি:</p>
    <table>
      <thead><tr><th>Aspect</th><th>Frequentist</th><th>Bayesian</th></tr></thead>
      <tbody>
        <tr><td>Probability</td><td>Long-run frequency</td><td>Degree of belief</td></tr>
        <tr><td>Parameters</td><td>Fixed unknown constants</td><td>Random variables with distributions</td></tr>
        <tr><td>Uncertainty</td><td>Confidence intervals</td><td>Credible intervals (direct probability)</td></tr>
        <tr><td>Prior knowledge</td><td>Not used</td><td>Explicitly modeled</td></tr>
      </tbody>
    </table>
    <p><strong>Bayesian update rule:</strong> Posterior ∝ Likelihood × Prior</p>
    <p>P(θ|data) ∝ P(data|θ) × P(θ)</p>

    <h3>২. Coin Flip: Bayesian Update</h3>
    <pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Prior: Beta distribution (conjugate prior for Bernoulli)
# Beta(α, β) → α = prior heads, β = prior tails

def bayesian_coin_update(data, prior_alpha=1, prior_beta=1):
    heads = sum(data)
    tails = len(data) - heads
    posterior_alpha = prior_alpha + heads
    posterior_beta  = prior_beta + tails
    return posterior_alpha, posterior_beta

# Simulate coin flips (true p=0.7)
np.random.seed(42)
true_p = 0.7
data = np.random.binomial(1, true_p, 100)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
p = np.linspace(0, 1, 200)

checkpoints = [0, 5, 20, 100]
for ax, n in zip(axes, checkpoints):
    a, b = bayesian_coin_update(data[:n], 1, 1)
    posterior = stats.beta(a, b)
    ax.plot(p, posterior.pdf(p), 'b-', linewidth=2)
    ax.axvline(true_p, color='r', linestyle='--', label='True p')
    ci_low, ci_high = posterior.ppf(0.025), posterior.ppf(0.975)
    ax.fill_between(p, posterior.pdf(p),
                    where=(p >= ci_low) & (p <= ci_high), alpha=0.3)
    ax.set_title(f'n={n}: 95% CI [{ci_low:.2f}, {ci_high:.2f}]')
    ax.legend()
plt.tight_layout(); plt.show()

# Final estimate
a, b = bayesian_coin_update(data, 1, 1)
posterior = stats.beta(a, b)
print(f"Posterior mean: {a/(a+b):.4f} (true: {true_p})")
print(f"95% Credible Interval: [{posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f}]")
</code></pre>

    <h3>৩. Bayesian Linear Regression</h3>
    <pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 30
x = np.linspace(0, 5, n)
true_slope, true_intercept = 2.0, 1.0
y = true_slope * x + true_intercept + np.random.normal(0, 1, n)

# Bayesian Linear Regression (conjugate)
# Prior: p(w) = N(0, alpha^{-1} I)
# Likelihood: y | x, w ~ N(w^T phi(x), beta^{-1})

alpha = 1.0   # Prior precision (inverse variance)
beta = 1.0    # Noise precision

X = np.column_stack([np.ones(n), x])  # Design matrix

# Posterior: N(m_N, S_N)
S_N_inv = alpha * np.eye(2) + beta * X.T @ X
S_N = np.linalg.inv(S_N_inv)
m_N = beta * S_N @ X.T @ y  # Posterior mean

print(f"Posterior mean (intercept): {m_N[0]:.4f} (true: {true_intercept})")
print(f"Posterior mean (slope):     {m_N[1]:.4f} (true: {true_slope})")
print(f"Posterior std (intercept):  {np.sqrt(S_N[0,0]):.4f}")
print(f"Posterior std (slope):      {np.sqrt(S_N[1,1]):.4f}")

# Predictive distribution
x_test = np.linspace(-1, 6, 100)
X_test = np.column_stack([np.ones(100), x_test])
pred_mean = X_test @ m_N
pred_var = 1/beta + np.diag(X_test @ S_N @ X_test.T)
pred_std = np.sqrt(pred_var)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.7, label='Data')
plt.plot(x_test, pred_mean, 'r-', label='Posterior mean')
plt.fill_between(x_test, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                 alpha=0.2, color='red', label='95% CI')
plt.legend(); plt.title('Bayesian Linear Regression')
plt.show()
</code></pre>

    <h3>৪. MCMC: Complex Posterior Sampling</h3>
    <pre><code class="language-python">import pymc as pm
import numpy as np

# When conjugate priors are unavailable, use MCMC
np.random.seed(42)
x = np.random.randn(100)
y = 2.5 * x + 1.0 + np.random.randn(100) * 0.5

with pm.Model() as model:
    # Priors
    slope     = pm.Normal('slope', mu=0, sigma=10)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma     = pm.HalfNormal('sigma', sigma=1)
    # Likelihood
    mu = slope * x + intercept
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=2, progressbar=False)

import arviz as az
print(az.summary(trace, var_names=['slope', 'intercept', 'sigma']))
az.plot_posterior(trace, var_names=['slope', 'intercept'])
</code></pre>
  `
};
