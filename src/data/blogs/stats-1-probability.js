export const stats_1_probability = {
  slug: 'stats-1-probability',
  title: 'সম্ভাবনা তত্ত্ব: ML-এর গাণিতিক ভিত্তি',
  description: 'Random Variables, Probability Distributions, Bayes Theorem, Joint/Marginal/Conditional Probability — Machine Learning বুঝতে যে Statistics অপরিহার্য।',
  date: 'মে ২০২৬',
  category: 'এমএল-এর জন্য পরিসংখ্যান',
  readTime: 14,
  content: `
    <h3>১. কেন Statistics ML-এর জন্য গুরুত্বপূর্ণ?</h3>
    <p>Machine Learning আসলে applied statistics। ML-এর প্রতিটি পদ্ধতির নিচে statistical foundations আছে:</p>
    <ul>
      <li>Linear Regression = Ordinary Least Squares (statistics)</li>
      <li>Logistic Regression = Maximum Likelihood Estimation</li>
      <li>Naive Bayes = Bayes Theorem + conditional independence</li>
      <li>Neural Networks = Function approximation + gradient descent optimization</li>
    </ul>

    <h3>২. Probability Basics</h3>
    <p><strong>Event:</strong> কোনো outcome যার probability আছে। P(A) = event A-এর probability, 0 ≤ P(A) ≤ 1।</p>
    <ul>
      <li><strong>Joint Probability:</strong> P(A ∩ B) = A এবং B উভয়ই ঘটার probability</li>
      <li><strong>Marginal Probability:</strong> P(A) = B সম্পর্কে জানা ছাড়াই A-এর probability</li>
      <li><strong>Conditional Probability:</strong> P(A|B) = B ঘটেছে জেনে A ঘটার probability = P(A∩B)/P(B)</li>
    </ul>

    <pre><code class="language-python">import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dice probability simulation
np.random.seed(42)
n_rolls = 100000
rolls = np.random.randint(1, 7, n_rolls)

# Marginal probability P(X = 6)
p_six = (rolls == 6).mean()
print(f"P(X=6) = {p_six:.4f} (expected: {1/6:.4f})")

# Joint probability P(roll1=6 AND roll2=6)
rolls1 = np.random.randint(1, 7, n_rolls)
rolls2 = np.random.randint(1, 7, n_rolls)
p_both_six = ((rolls1 == 6) & (rolls2 == 6)).mean()
print(f"P(both=6) = {p_both_six:.4f} (expected: {1/36:.4f})")

# Conditional: P(sum=7 | first=3)
p_sum7_given_3 = ((rolls1 + rolls2 == 7) & (rolls1 == 3)).sum() / (rolls1 == 3).sum()
print(f"P(sum=7 | first=3) = {p_sum7_given_3:.4f} (expected: {1/6:.4f})")
</code></pre>

    <h3>৩. Bayes Theorem</h3>
    <p><strong>P(A|B) = P(B|A) × P(A) / P(B)</strong></p>
    <p>ML-এ Bayes Theorem-এর ব্যাপক ব্যবহার। Naive Bayes classifier সরাসরি এটি ব্যবহার করে।</p>

    <pre><code class="language-python"># Medical test example
# Disease prevalence: P(Disease) = 0.01
# Test sensitivity: P(Positive|Disease) = 0.99
# Test specificity: P(Negative|No Disease) = 0.95
# What is P(Disease|Positive)?

p_disease = 0.01
p_pos_given_disease = 0.99
p_neg_given_no_disease = 0.95
p_pos_given_no_disease = 1 - p_neg_given_no_disease

# P(Positive) = P(Pos|Dis)*P(Dis) + P(Pos|NoDis)*P(NoDis)
p_positive = (p_pos_given_disease * p_disease +
              p_pos_given_no_disease * (1 - p_disease))

# Bayes: P(Disease|Positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
print(f"P(Disease | Positive test) = {p_disease_given_pos:.4f}")
# Only ~16.5% despite 99% sensitivity! This is the base rate fallacy.

# Simulation verification
np.random.seed(42)
n = 1000000
has_disease = np.random.random(n) < p_disease
test_positive = np.where(
    has_disease,
    np.random.random(n) < p_pos_given_disease,
    np.random.random(n) < p_pos_given_no_disease
)
print(f"Simulated P(Disease | Positive) = {has_disease[test_positive].mean():.4f}")
</code></pre>

    <h3>৪. Random Variables ও Probability Distributions</h3>
    <pre><code class="language-python">from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
x = np.linspace(-4, 4, 200)

# Normal Distribution
dist = stats.norm(0, 1)
axes[0,0].plot(x, dist.pdf(x))
axes[0,0].set_title('Normal (μ=0, σ=1)')
axes[0,0].fill_between(x[(x>=-1)&(x<=1)], dist.pdf(x[(x>=-1)&(x<=1)]), alpha=0.3, label='68%')

# Binomial
k = np.arange(0, 21)
axes[0,1].bar(k, stats.binom.pmf(k, 20, 0.5), alpha=0.7)
axes[0,1].set_title('Binomial (n=20, p=0.5)')

# Poisson
axes[0,2].bar(k, stats.poisson.pmf(k, 5), alpha=0.7, color='green')
axes[0,2].set_title('Poisson (λ=5)')

# Exponential
x_pos = np.linspace(0, 5, 200)
axes[1,0].plot(x_pos, stats.expon.pdf(x_pos, scale=1))
axes[1,0].set_title('Exponential (λ=1)')

# Beta
x01 = np.linspace(0, 1, 200)
for a, b, c in [(2,2,'blue'), (5,2,'red'), (2,5,'green')]:
    axes[1,1].plot(x01, stats.beta.pdf(x01, a, b), color=c, label=f'α={a},β={b}')
axes[1,1].legend(); axes[1,1].set_title('Beta Distribution')

# t-distribution vs Normal
for df, c in [(1,'red'), (5,'orange'), (30,'green')]:
    axes[1,2].plot(x, stats.t.pdf(x, df), color=c, label=f'df={df}')
axes[1,2].plot(x, stats.norm.pdf(x), 'b--', label='Normal')
axes[1,2].legend(); axes[1,2].set_title('t vs Normal')

plt.tight_layout(); plt.show()
print("Key: Normal → linear regression; Binomial → classification; Poisson → count data")
</code></pre>
  `
};
