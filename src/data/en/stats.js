export const statsEn = [
  {
    slug: 'stats-1-probability',
    title: 'Probability Theory: The Math Behind ML',
    description: 'Random variables, Bayes theorem, joint/conditional/marginal probability — the statistical foundations every ML practitioner needs',
    category: 'Statistics for ML',
    content: `
<h3>Probability Fundamentals</h3>
<ul>
<li><strong>Joint:</strong> P(A ∩ B) — both A and B occur</li>
<li><strong>Marginal:</strong> P(A) — A regardless of B</li>
<li><strong>Conditional:</strong> P(A|B) = P(A∩B)/P(B) — A given B occurred</li>
</ul>

<h3>Bayes Theorem</h3>
<p><strong>P(A|B) = P(B|A) × P(A) / P(B)</strong></p>
<pre><code class="language-python">import numpy as np
# Medical test: sensitivity=99%, specificity=95%, prevalence=1%
p_disease = 0.01
p_pos_given_dis = 0.99
p_pos_given_no = 0.05
p_positive = p_pos_given_dis*p_disease + p_pos_given_no*(1-p_disease)
posterior = p_pos_given_dis * p_disease / p_positive
print(f"P(Disease | Positive test) = {posterior:.4f}")
# Only ~16.6% despite 99% sensitivity — base rate fallacy!
</code></pre>

<h3>Key Distributions in ML</h3>
<table>
<tr><th>Distribution</th><th>ML Application</th></tr>
<tr><td>Normal N(μ,σ²)</td><td>Regression errors, weight initialization</td></tr>
<tr><td>Bernoulli(p)</td><td>Binary classification output</td></tr>
<tr><td>Categorical</td><td>Softmax multi-class output</td></tr>
<tr><td>Beta(α,β)</td><td>Bayesian prior for probability</td></tr>
<tr><td>Dirichlet</td><td>Topic model (LDA) prior</td></tr>
</table>
`
  },
  {
    slug: 'stats-2-hypothesis',
    title: 'Hypothesis Testing: Comparing Models Rigorously',
    description: 'p-values, Type I/II errors, paired t-test for models, chi-square for features, Bonferroni and FDR correction',
    category: 'Statistics for ML',
    content: `
<h3>The Framework</h3>
<ul>
<li><strong>H₀:</strong> No difference (null hypothesis)</li>
<li><strong>p-value:</strong> P(data this extreme | H₀ is true)</li>
<li><strong>α = 0.05:</strong> reject H₀ if p &lt; α</li>
<li><strong>Type I (α):</strong> false positive — "detected" an effect that doesn't exist</li>
<li><strong>Type II (β):</strong> false negative — missed a real effect</li>
</ul>

<pre><code class="language-python">from scipy import stats
import numpy as np

model_a = np.array([0.82, 0.79, 0.85, 0.81, 0.83, 0.80, 0.84, 0.82, 0.78, 0.83])
model_b = np.array([0.85, 0.88, 0.87, 0.86, 0.89, 0.87, 0.85, 0.88, 0.86, 0.90])

# Paired t-test (same CV folds = paired!)
t, p = stats.ttest_rel(model_a, model_b)
print(f"Paired t-test: t={t:.4f}, p={p:.4f}")
print(f"Significant: {p < 0.05}")

# Cohen's d effect size
d = (model_b.mean() - model_a.mean()) / np.sqrt((model_a.std()**2+model_b.std()**2)/2)
print(f"Cohen's d = {d:.4f}  (small=0.2, medium=0.5, large=0.8)")
</code></pre>

<h3>Multiple Testing Correction</h3>
<pre><code class="language-python">from statsmodels.stats.multitest import multipletests
import numpy as np

p_values = np.random.uniform(0, 1, 100)
p_values[:10] = np.random.uniform(0, 0.001, 10)  # 10 true effects

bh, bh_pvals, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
print(f"Naive: {(p_values<0.05).sum()} significant")
print(f"Benjamini-Hochberg (FDR 5%): {bh.sum()} significant")
</code></pre>
`
  },
  {
    slug: 'stats-3-distributions',
    title: 'Probability Distributions: Where Each Shows Up in ML',
    description: 'Normal, Binomial, Poisson, Beta distributions, Central Limit Theorem, MLE vs MAP estimation',
    category: 'Statistics for ML',
    content: `
<h3>Central Limit Theorem</h3>
<p>The sample mean of any distribution approaches Normal as n → ∞. Foundation of confidence intervals, hypothesis tests, and SGD convergence analysis.</p>

<h3>MLE vs MAP</h3>
<p><strong>MLE:</strong> argmax P(data|θ) — find parameters that make data most likely.<br/>
<strong>MAP:</strong> argmax P(data|θ)P(θ) — MLE with a prior. L2 regularization = MAP with Gaussian prior.</p>
<pre><code class="language-python">import numpy as np
from scipy.optimize import minimize

def sigmoid(x): return 1/(1+np.exp(-x))

def mle(w, X, y):
    p = np.clip(sigmoid(X@w), 1e-10, 1-1e-10)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def map_loss(w, X, y, lam=0.01):
    return mle(w, X, y) + lam * np.sum(w**2)  # L2 regularization = Gaussian prior

np.random.seed(42)
X = np.hstack([np.ones((200,1)), np.random.randn(200,3)])
y = (sigmoid(X @ [0,1.5,-2,0.5]) > 0.5).astype(float)

mle_w = minimize(mle, np.zeros(4), args=(X,y)).x
map_w = minimize(map_loss, np.zeros(4), args=(X,y,0.1)).x
print(f"MLE: {mle_w.round(3)}\nMAP: {map_w.round(3)}")
</code></pre>

<h3>Distributions Quick Reference</h3>
<pre><code class="language-python">from scipy import stats
import numpy as np

# Fit a normal distribution to data
data = np.random.normal(100, 15, 1000)
mu, sigma = stats.norm.fit(data)
print(f"Fitted: μ={mu:.2f}, σ={sigma:.2f}")
print(f"P(X > 130) = {1 - stats.norm.cdf(130, mu, sigma):.4f}")

# Poisson: count data
lam = 3.5  # average events per hour
print(f"P(exactly 5 events) = {stats.poisson.pmf(5, lam):.4f}")
print(f"P(>= 5 events) = {1 - stats.poisson.cdf(4, lam):.4f}")
</code></pre>
`
  },
  {
    slug: 'stats-4-bayesian',
    title: 'Bayesian Inference: Quantifying Uncertainty',
    description: 'Prior, likelihood, posterior, Beta-Binomial conjugate, Bayesian linear regression, credible intervals, intro to MCMC',
    category: 'Statistics for ML',
    content: `
<h3>Frequentist vs Bayesian</h3>
<table>
<tr><th>Aspect</th><th>Frequentist</th><th>Bayesian</th></tr>
<tr><td>Parameters</td><td>Fixed unknown constants</td><td>Random variables with distributions</td></tr>
<tr><td>Uncertainty</td><td>Confidence intervals</td><td>Credible intervals (direct probability)</td></tr>
<tr><td>Prior knowledge</td><td>Not used</td><td>Explicitly modeled</td></tr>
</table>

<h3>Beta-Binomial Conjugate (Coin Flipping)</h3>
<pre><code class="language-python">import numpy as np
from scipy import stats

# Prior: Beta(1,1) = Uniform
# After observing h heads, t tails: posterior = Beta(1+h, 1+t)
np.random.seed(42)
flips = np.random.binomial(1, 0.7, 100)  # true p=0.7
h, t = flips.sum(), len(flips) - flips.sum()

posterior = stats.beta(1+h, 1+t)
print(f"Posterior mean: {posterior.mean():.4f}")
print(f"95% Credible Interval: [{posterior.ppf(0.025):.4f}, {posterior.ppf(0.975):.4f}]")
# "There is 95% probability that the true p is in this interval" — direct interpretation!
</code></pre>

<h3>Bayesian Linear Regression</h3>
<pre><code class="language-python">import numpy as np

np.random.seed(42)
x = np.linspace(0, 5, 30)
y = 2*x + 1 + np.random.normal(0, 1, 30)
X = np.column_stack([np.ones(30), x])

alpha, beta = 1.0, 1.0  # prior and noise precision
S_N = np.linalg.inv(alpha*np.eye(2) + beta*X.T@X)  # posterior covariance
m_N = beta * S_N @ X.T @ y                           # posterior mean

print(f"Posterior: intercept={m_N[0]:.4f}±{np.sqrt(S_N[0,0]):.4f}, slope={m_N[1]:.4f}±{np.sqrt(S_N[1,1]):.4f}")
</code></pre>
`
  },
  {
    slug: 'stats-5-project',
    title: 'Statistics Project: Complete A/B Testing Framework',
    description: 'Power analysis for sample size, z-test for proportions, bootstrap CI, multiple testing pitfalls, guardrail metrics',
    category: 'Statistics for ML',
    content: `
<h3>Complete A/B Testing Pipeline</h3>
<pre><code class="language-python">from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
import numpy as np

# Step 1: Power Analysis (before the experiment)
effect = proportion_effectsize(0.12, 0.10)  # detect 10%→12% conversion
n = TTestIndPower().solve_power(effect_size=effect, power=0.80, alpha=0.05)
print(f"Required per group: {int(np.ceil(n))}")

# Step 2: Run experiment
np.random.seed(42)
control   = np.random.binomial(1, 0.10, int(n))
treatment = np.random.binomial(1, 0.125, int(n))

# Step 3: Test
z, p = proportions_ztest([treatment.sum(), control.sum()], [len(treatment), len(control)], alternative='larger')
print(f"p-value: {p:.4f}, Significant: {p < 0.05}")

# Step 4: Effect size and CI
lift = (treatment.mean()-control.mean()) / control.mean() * 100
ci = proportion_confint(treatment.sum(), len(treatment))
print(f"Lift: {lift:.1f}%, Treatment CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
</code></pre>

<h3>Bootstrap Confidence Interval</h3>
<pre><code class="language-python">def bootstrap_ci(a, b, n=10000, ci=0.95):
    diffs = [np.random.choice(b,len(b),replace=True).mean() -
             np.random.choice(a,len(a),replace=True).mean() for _ in range(n)]
    alpha = (1-ci)/2
    return np.percentile(diffs, [alpha*100, (1-alpha)*100])

lo, hi = bootstrap_ci(control, treatment)
print(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
print(f"Contains zero (no effect): {lo <= 0 <= hi}")
</code></pre>

<h3>Common Pitfalls</h3>
<ul>
<li><strong>Peeking:</strong> Stopping early when p &lt; 0.05 inflates Type I error. Pre-register duration.</li>
<li><strong>Multiple variants:</strong> 5 variants at α=0.05 → P(false positive) = 23%. Use Bonferroni.</li>
<li><strong>Novelty effect:</strong> Wait 2+ weeks for behavior to stabilize.</li>
<li><strong>Missing guardrails:</strong> Always check revenue, latency, and error rate alongside primary metric.</li>
</ul>
`
  },
];
