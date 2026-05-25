export const mlops_4_monitoring = {
  slug: 'mlops-4-monitoring',
  title: 'মডেল মনিটরিং ও ড্রিফট ডিটেকশন',
  description: 'প্রোডাকশনে ডেটা ড্রিফট ও কনসেপ্ট ড্রিফট শনাক্ত করে মডেলের মান বজায় রাখুন',
  date: 'মে ২০২৫',
  category: 'এমএলঅপস',
  readTime: 12,
  content: `
<h3>কেন মনিটরিং দরকার?</h3>
<p>একটি মডেল ট্রেনিংয়ে ভালো করলেই প্রোডাকশনে চিরকাল ভালো থাকে না। বাস্তব ডেটা সময়ের সাথে পরিবর্তিত হয়।</p>

<h4>দুই ধরনের ড্রিফট</h4>
<ul>
<li><strong>ডেটা ড্রিফট (Input Drift):</strong> ইনপুট ফিচারের বিতরণ পরিবর্তন। যেমন: ব্যবহারকারীর বয়স বিতরণ বদলে গেছে।</li>
<li><strong>কনসেপ্ট ড্রিফট:</strong> ইনপুট-আউটপুট সম্পর্ক পরিবর্তন। যেমন: করোনার পরে ক্রেতার আচরণ বদলে গেছে।</li>
</ul>

<h3>Evidently AI দিয়ে ড্রিফট শনাক্তকরণ</h3>

<pre><code class="language-python">import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ModelPerformancePreset
from evidently.metrics import *

# ট্রেনিং ডেটা (রেফারেন্স)
np.random.seed(42)
reference_data = pd.DataFrame({
    'বয়স': np.random.normal(35, 10, 1000),
    'আয়': np.random.normal(60000, 15000, 1000),
    'ক্রেডিট_স্কোর': np.random.normal(680, 80, 1000),
    'ঋণ_পরিমাণ': np.random.normal(200000, 50000, 1000),
    'prediction': np.random.binomial(1, 0.7, 1000),
    'target': np.random.binomial(1, 0.7, 1000)
})

# প্রোডাকশন ডেটা (ড্রিফটেড)
current_data = pd.DataFrame({
    'বয়স': np.random.normal(42, 12, 500),  # বয়স বেড়েছে
    'আয়': np.random.normal(55000, 20000, 500),  # আয় কমেছে
    'ক্রেডিট_স্কোর': np.random.normal(650, 100, 500),  # ক্রেডিট স্কোর কমেছে
    'ঋণ_পরিমাণ': np.random.normal(250000, 70000, 500),
    'prediction': np.random.binomial(1, 0.6, 500),
    'target': np.random.binomial(1, 0.55, 500)  # মডেলের নির্ভুলতা কমেছে
})

# ডেটা ড্রিফট রিপোর্ট
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference_data, current_data=current_data)
data_drift_report.save_html("data_drift_report.html")

# পারফরম্যান্স রিপোর্ট
perf_report = Report(metrics=[
    BinaryClassificationPreset(),
])
perf_report.run(reference_data=reference_data, current_data=current_data)
perf_report.save_html("performance_report.html")
print("রিপোর্ট তৈরি হয়েছে!")
</code></pre>

<h3>স্ট্যাটিস্টিক্যাল ড্রিফট টেস্ট</h3>

<pre><code class="language-python">from scipy import stats

def detect_drift(reference, current, feature, alpha=0.05):
    """KS টেস্ট দিয়ে ড্রিফট শনাক্তকরণ"""
    ref_vals = reference[feature].dropna()
    cur_vals = current[feature].dropna()

    ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
    is_drifted = p_value < alpha

    print(f"\\nফিচার: {feature}")
    print(f"  KS স্ট্যাটিস্টিক: {ks_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  ড্রিফট সনাক্ত: {'✅ হ্যাঁ' if is_drifted else '❌ না'}")

    return is_drifted, ks_stat, p_value

# সব ফিচার পরীক্ষা
features = ['বয়স', 'আয়', 'ক্রেডিট_স্কোর', 'ঋণ_পরিমাণ']
drift_results = {}

for feat in features:
    drifted, ks, p = detect_drift(reference_data, current_data, feat)
    drift_results[feat] = {'drifted': drifted, 'ks': ks, 'p_value': p}

print("\\n--- ড্রিফট সারাংশ ---")
drifted_features = [f for f, r in drift_results.items() if r['drifted']]
print(f"ড্রিফটেড ফিচার: {drifted_features}")
</code></pre>

<h3>পারফরম্যান্স মনিটরিং ড্যাশবোর্ড</h3>

<pre><code class="language-python">import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def monitor_performance_over_time(model_logs):
    """সময়ের সাথে মডেল পারফরম্যান্স ট্র্যাক"""
    dates = [entry['date'] for entry in model_logs]
    accuracies = [entry['accuracy'] for entry in model_logs]
    f1_scores = [entry['f1'] for entry in model_logs]
    drift_scores = [entry['drift_score'] for entry in model_logs]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(dates, accuracies, 'b-o', label='নির্ভুলতা')
    axes[0].axhline(y=0.85, color='r', linestyle='--', label='ন্যূনতম থ্রেশহোল্ড')
    axes[0].set_title('মডেল নির্ভুলতা সময়ের সাথে')
    axes[0].legend()

    axes[1].plot(dates, f1_scores, 'g-o', label='F1 স্কোর')
    axes[1].set_title('F1 স্কোর সময়ের সাথে')
    axes[1].legend()

    axes[2].bar(dates, drift_scores, color=['red' if s > 0.05 else 'green' for s in drift_scores])
    axes[2].axhline(y=0.05, color='r', linestyle='--', label='ড্রিফট থ্রেশহোল্ড')
    axes[2].set_title('ড্রিফট স্কোর (p-value)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('monitoring_dashboard.png', dpi=150)
    plt.show()

# সিমুলেটেড লগ
model_logs = [
    {'date': '2025-01', 'accuracy': 0.92, 'f1': 0.90, 'drift_score': 0.45},
    {'date': '2025-02', 'accuracy': 0.91, 'f1': 0.89, 'drift_score': 0.30},
    {'date': '2025-03', 'accuracy': 0.88, 'f1': 0.85, 'drift_score': 0.08},
    {'date': '2025-04', 'accuracy': 0.83, 'f1': 0.79, 'drift_score': 0.02},  # ড্রিফট!
    {'date': '2025-05', 'accuracy': 0.79, 'f1': 0.74, 'drift_score': 0.01},  # রিট্রেইন দরকার
]
monitor_performance_over_time(model_logs)
</code></pre>

<h3>স্বয়ংক্রিয় পুনর্প্রশিক্ষণ ট্রিগার</h3>

<pre><code class="language-python">def should_retrain(current_accuracy, baseline_accuracy,
                   drift_score, thresholds):
    """পুনর্প্রশিক্ষণ দরকার কিনা নির্ধারণ"""
    reasons = []

    if current_accuracy < thresholds['min_accuracy']:
        reasons.append(f"নির্ভুলতা কম ({current_accuracy:.2%} < {thresholds['min_accuracy']:.2%})")

    drop = baseline_accuracy - current_accuracy
    if drop > thresholds['max_accuracy_drop']:
        reasons.append(f"নির্ভুলতা {drop:.2%} হ্রাস পেয়েছে")

    if drift_score < thresholds['drift_p_value']:
        reasons.append(f"ডেটা ড্রিফট সনাক্ত (p={drift_score:.4f})")

    if reasons:
        print("🔴 পুনর্প্রশিক্ষণ প্রয়োজন:")
        for r in reasons:
            print(f"  • {r}")
        return True
    else:
        print("🟢 মডেল ভালো আছে, পুনর্প্রশিক্ষণ দরকার নেই")
        return False

thresholds = {'min_accuracy': 0.85, 'max_accuracy_drop': 0.05, 'drift_p_value': 0.05}
should_retrain(0.79, 0.92, 0.01, thresholds)
</code></pre>
`
};
