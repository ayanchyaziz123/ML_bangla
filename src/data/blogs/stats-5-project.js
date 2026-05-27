export const stats_5_project = {
  slug: 'stats-5-project',
  title: 'Statistics Project: A/B Testing Pipeline',
  description: 'সম্পূর্ণ A/B testing framework — sample size calculation, experiment design, statistical significance testing, effect size, confidence intervals এবং common pitfalls।',
  date: 'মে ২০২৬',
  category: 'এমএল-এর জন্য পরিসংখ্যান',
  readTime: 16,
  content: `
    <h3>A/B Testing কী?</h3>
    <p>A/B testing হলো দুটি version (control A ও treatment B) এর মধ্যে randomized experiment। ML-এ নতুন model কে production-এ deploy করার আগে A/B test করা হয়।</p>

    <h3>১. Sample Size Calculation (Power Analysis)</h3>
    <pre><code class="language-python">from statsmodels.stats.power import TTestIndPower
import numpy as np

# Minimum Detectable Effect (MDE): কত % improvement detect করতে চাই?
baseline_conversion = 0.10   # 10% baseline
mde = 0.02                   # Want to detect 2% absolute improvement
target_conversion = baseline_conversion + mde

# Effect size (Cohen's h for proportions)
from statsmodels.stats.proportion import proportion_effectsize
effect_size = proportion_effectsize(target_conversion, baseline_conversion)

# Power analysis
analysis = TTestIndPower()
sample_size = analysis.solve_power(
    effect_size=effect_size,
    power=0.80,     # 80% chance of detecting a real effect
    alpha=0.05,     # 5% false positive rate
    ratio=1.0       # Equal group sizes
)
print(f"Required sample size per group: {int(np.ceil(sample_size))}")
print(f"Total sample size: {int(np.ceil(sample_size)) * 2}")

# How long will the experiment run?
daily_traffic = 5000
days_needed = int(np.ceil(sample_size * 2 / daily_traffic))
print(f"Experiment duration: {days_needed} days (at {daily_traffic} visitors/day)")
</code></pre>

    <h3>২. Running the Experiment</h3>
    <pre><code class="language-python">import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
n_per_group = 2000

# Simulate experiment data
control = np.random.binomial(1, 0.10, n_per_group)    # 10% conversion
treatment = np.random.binomial(1, 0.125, n_per_group)  # 12.5% conversion

# Basic stats
print(f"Control:   n={len(control)}, conversions={control.sum()}, rate={control.mean():.4f}")
print(f"Treatment: n={len(treatment)}, conversions={treatment.sum()}, rate={treatment.mean():.4f}")

# Z-test for proportions
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

counts = np.array([treatment.sum(), control.sum()])
nobs   = np.array([len(treatment), len(control)])
z_stat, p_value = proportions_ztest(counts, nobs, alternative='larger')

print(f"\\nZ-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05? {p_value < 0.05}")

# Confidence intervals
ci_control   = proportion_confint(control.sum(),   len(control),   alpha=0.05)
ci_treatment = proportion_confint(treatment.sum(), len(treatment), alpha=0.05)
print(f"\\nControl 95% CI:   [{ci_control[0]:.4f}, {ci_control[1]:.4f}]")
print(f"Treatment 95% CI: [{ci_treatment[0]:.4f}, {ci_treatment[1]:.4f}]")

# Relative lift
lift = (treatment.mean() - control.mean()) / control.mean() * 100
print(f"\\nRelative lift: {lift:.1f}%")
</code></pre>

    <h3>৩. Bootstrap Confidence Intervals</h3>
    <pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

def bootstrap_diff(control, treatment, n_bootstrap=10000, ci=0.95):
    diffs = []
    for _ in range(n_bootstrap):
        c_sample = np.random.choice(control, len(control), replace=True)
        t_sample = np.random.choice(treatment, len(treatment), replace=True)
        diffs.append(t_sample.mean() - c_sample.mean())
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(diffs, [alpha*100, (1-alpha)*100])
    return np.array(diffs), lower, upper

diffs, lower, upper = bootstrap_diff(control, treatment)
print(f"Bootstrap 95% CI for difference: [{lower:.4f}, {upper:.4f}]")
print(f"Zero in CI? {lower <= 0 <= upper}")  # If False, significant

plt.figure(figsize=(10, 4))
plt.hist(diffs, bins=50, density=True, alpha=0.7, color='steelblue')
plt.axvline(0, color='red', linestyle='--', label='No effect')
plt.axvline(lower, color='orange', linestyle=':', label=f'95% CI [{lower:.4f}, {upper:.4f}]')
plt.axvline(upper, color='orange', linestyle=':')
plt.xlabel('Conversion rate difference (Treatment - Control)')
plt.legend(); plt.title('Bootstrap Distribution of Difference')
plt.show()
</code></pre>

    <h3>৪. Common A/B Testing Pitfalls</h3>
    <pre><code class="language-python"># Pitfall 1: Peeking (stopping early when p < 0.05)
np.random.seed(42)
n = 3000
control_stream   = np.random.binomial(1, 0.10, n)
treatment_stream = np.random.binomial(1, 0.10, n)  # SAME rate — null is true!

p_values = []
for i in range(100, n+1, 50):
    c, t = control_stream[:i], treatment_stream[:i]
    _, p = stats.ttest_ind(c, t)
    p_values.append(p)

false_sig = sum(1 for p in p_values if p < 0.05)
print(f"Peeking: {false_sig}/{len(p_values)} checks showed p < 0.05 (null is TRUE!)")
print("Solution: Pre-register experiment duration. Don't peek!")

# Pitfall 2: Multiple comparisons
n_variants = 5
# Testing 5 variants vs control = 5 tests, expected false positives = 5*0.05 = 0.25
print(f"\\nFor {n_variants} variants at α=0.05:")
print(f"P(at least one false positive) = {1 - (1-0.05)**n_variants:.4f}")
print("Solution: Bonferroni correction → α_adjusted = 0.05 / 5 = 0.01")

# Pitfall 3: Not checking guardrail metrics
print("\\nAlways monitor guardrail metrics:")
print("- Revenue per user (primary)")
print("- Page load time (guardrail - shouldn't increase)")
print("- Error rate (guardrail)")
print("- User retention (long-term)")
</code></pre>
  `
};
