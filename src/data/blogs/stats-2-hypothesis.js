export const stats_2_hypothesis = {
  slug: 'stats-2-hypothesis',
  title: 'Hypothesis Testing: ML Experiment সঠিকভাবে মূল্যায়ন',
  description: 'Null Hypothesis, p-value, Type I/II Error, t-test, chi-square test, A/B testing এবং Multiple Testing Correction — ML model comparison-এর statistical foundation।',
  date: 'মে ২০২৬',
  category: 'এমএল-এর জন্য পরিসংখ্যান',
  readTime: 13,
  content: `
    <h3>১. Hypothesis Testing কী?</h3>
    <p>দুটি ML model-এর accuracy তুলনা করার সময় প্রশ্ন ওঠে: পার্থক্যটা real নাকি শুধু random chance? <strong>Hypothesis Testing</strong> এই প্রশ্নের statistical উত্তর দেয়।</p>
    <ul>
      <li><strong>Null Hypothesis (H₀):</strong> কোনো real পার্থক্য নেই (default assumption)</li>
      <li><strong>Alternative Hypothesis (H₁):</strong> পার্থক্য আছে</li>
      <li><strong>p-value:</strong> H₀ সত্য ধরে নিলে observed data পাওয়ার probability</li>
      <li><strong>α (significance level):</strong> সাধারণত 0.05 — এর নিচে p-value হলে H₀ reject করি</li>
    </ul>
    <table>
      <thead><tr><th></th><th>H₀ সত্য</th><th>H₀ মিথ্যা</th></tr></thead>
      <tbody>
        <tr><td>H₀ reject করলে</td><td>Type I Error (False Positive) α</td><td>Correct (Power = 1-β)</td></tr>
        <tr><td>H₀ reject না করলে</td><td>Correct</td><td>Type II Error (False Negative) β</td></tr>
      </tbody>
    </table>

    <h3>২. t-test: দুটি Model-এর Accuracy তুলনা</h3>
    <pre><code class="language-python">from scipy import stats
import numpy as np

np.random.seed(42)
# Cross-validation scores for two models
model_a = np.array([0.82, 0.79, 0.85, 0.81, 0.83, 0.80, 0.84, 0.82, 0.78, 0.83])
model_b = np.array([0.85, 0.88, 0.87, 0.86, 0.89, 0.87, 0.85, 0.88, 0.86, 0.90])

print(f"Model A: {model_a.mean():.4f} ± {model_a.std():.4f}")
print(f"Model B: {model_b.mean():.4f} ± {model_b.std():.4f}")

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(model_a, model_b)
print(f"\\nIndependent t-test: t={t_stat:.4f}, p={p_value:.4f}")
print(f"Significant (p < 0.05)? {p_value < 0.05}")

# Paired t-test (same CV folds - BETTER for model comparison)
t_stat_p, p_value_p = stats.ttest_rel(model_a, model_b)
print(f"\\nPaired t-test: t={t_stat_p:.4f}, p={p_value_p:.4f}")
print(f"Significant (p < 0.05)? {p_value_p < 0.05}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((model_a.std()**2 + model_b.std()**2) / 2)
cohens_d = (model_b.mean() - model_a.mean()) / pooled_std
print(f"\\nCohen's d = {cohens_d:.4f}")
print("Effect size: small=0.2, medium=0.5, large=0.8")
</code></pre>

    <h3>৩. Chi-Square Test: Feature Selection</h3>
    <pre><code class="language-python">from scipy.stats import chi2_contingency, chi2
import pandas as pd, numpy as np

# Chi-square test for feature independence
# Example: is 'Gender' independent of 'Purchased'?
np.random.seed(42)
n = 1000
gender = np.random.choice(['Male', 'Female'], n)
# Introduce slight dependency
purchased = np.where(gender == 'Female',
                     np.random.choice([0, 1], n, p=[0.4, 0.6]),
                     np.random.choice([0, 1], n, p=[0.55, 0.45]))

contingency = pd.crosstab(gender, purchased, rownames=['Gender'], colnames=['Purchased'])
print(contingency)

chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
print(f"\\nChi-square: {chi2_stat:.4f}, p-value: {p_val:.4f}, df: {dof}")
print(f"Features are {'dependent' if p_val < 0.05 else 'independent'} (p < 0.05 threshold)")

# sklearn chi2 for feature selection in NLP
from sklearn.feature_selection import chi2 as sk_chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train', categories=['sci.med', 'talk.politics.guns'])
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data.data)
y = data.target

chi2_scores, p_values = sk_chi2(X, y)
# Top features most associated with class
top_idx = chi2_scores.argsort()[-10:][::-1]
features = vectorizer.get_feature_names_out()
print("\\nTop discriminative features:")
for idx in top_idx:
    print(f"  {features[idx]}: chi2={chi2_scores[idx]:.1f}, p={p_values[idx]:.2e}")
</code></pre>

    <h3>৪. Multiple Testing Problem</h3>
    <pre><code class="language-python">import numpy as np
from statsmodels.stats.multitest import multipletests

# Problem: test 100 features, α=0.05
# Expected false positives = 100 × 0.05 = 5 features flagged by chance!
np.random.seed(42)
n_tests = 100
# Simulate: only 10 are truly significant
p_values = np.concatenate([
    np.random.uniform(0, 0.001, 10),   # True positives
    np.random.uniform(0, 1, 90)        # Nulls
])
np.random.shuffle(p_values)

naive_significant = (p_values < 0.05).sum()
print(f"Naive (α=0.05): {naive_significant} significant features")

# Bonferroni correction: α / n_tests (very conservative)
bonferroni = multipletests(p_values, method='bonferroni')[0]
print(f"Bonferroni: {bonferroni.sum()} significant features")

# Benjamini-Hochberg (FDR control): less conservative, better for ML
bh_results, bh_pvals, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
print(f"Benjamini-Hochberg (FDR 5%): {bh_results.sum()} significant features")
# BH is preferred for feature selection in ML
</code></pre>
  `
};
