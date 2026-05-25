export const anomaly_1_statistics = {
  slug: 'anomaly-1-statistics',
  title: 'অ্যানোমালি ডিটেকশন: পরিসংখ্যান পদ্ধতি',
  description: 'অ্যানোমালি কী, পয়েন্ট-কনটেক্সচুয়াল-কালেক্টিভ অ্যানোমালির পার্থক্য, Z-score, IQR, Grubbs Test এবং পরিসংখ্যান থ্রেশহোল্ড দিয়ে আউটলায়ার ডিটেকশনের সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'অ্যানোমালি ডিটেকশন',
  readTime: 12,
  content: `
    <h3>১. অ্যানোমালি ডিটেকশন কী?</h3>
    <p>
      ডেটার মধ্যে এমন কিছু পয়েন্ট থাকে যেগুলো বাকি ডেটা থেকে সম্পূর্ণ আলাদা — এগুলোই <strong>অ্যানোমালি</strong> বা <strong>আউটলায়ার</strong>। মেশিন লার্নিংয়ে অ্যানোমালি ডিটেকশন হলো এমন একটি প্রক্রিয়া যেখানে আমরা এই অস্বাভাবিক পয়েন্টগুলো খুঁজে বের করি।
    </p>
    <p>
      বাস্তব জীবনে অ্যানোমালি ডিটেকশনের প্রয়োগ অনেক:
    </p>
    <ul>
      <li><strong>ব্যাংকিং:</strong> ক্রেডিট কার্ড জালিয়াতি ধরা</li>
      <li><strong>নেটওয়ার্ক সিকিউরিটি:</strong> সাইবার আক্রমণ সনাক্ত করা</li>
      <li><strong>শিল্প:</strong> যন্ত্রপাতির ত্রুটি পূর্বানুমান করা</li>
      <li><strong>স্বাস্থ্যসেবা:</strong> রোগীর অস্বাভাবিক স্বাস্থ্য তথ্য চিহ্নিত করা</li>
      <li><strong>ই-কমার্স:</strong> বট ট্রাফিক ও ফেক রিভিউ ধরা</li>
    </ul>

    <h3>২. অ্যানোমালির তিনটি প্রকার</h3>

    <h4>২.১ পয়েন্ট অ্যানোমালি (Point Anomaly)</h4>
    <p>
      সবচেয়ে সাধারণ ধরন। একটি একক ডেটা পয়েন্ট যদি বাকি ডেটা থেকে অনেক আলাদা হয়, তাকে পয়েন্ট অ্যানোমালি বলে। উদাহরণ: কারো সাধারণ মাসিক ব্যয় ৫,০০০ টাকা, কিন্তু হঠাৎ একমাসে ৫০,০০০ টাকা লেনদেন।
    </p>

    <h4>২.২ কনটেক্সচুয়াল অ্যানোমালি (Contextual Anomaly)</h4>
    <p>
      একটি পয়েন্ট নির্দিষ্ট প্রেক্ষাপটে অ্যানোমালি, কিন্তু অন্য প্রেক্ষাপটে স্বাভাবিক। উদাহরণ: গ্রীষ্মে ৩৫°C তাপমাত্রা স্বাভাবিক, কিন্তু শীতে এই তাপমাত্রা অ্যানোমালি। সময়-সিরিজ ডেটায় এটি বেশি দেখা যায়।
    </p>

    <h4>২.৩ কালেক্টিভ অ্যানোমালি (Collective Anomaly)</h4>
    <p>
      এককভাবে প্রতিটি পয়েন্ট স্বাভাবিক, কিন্তু একসাথে দেখলে তারা অস্বাভাবিক প্যাটার্ন তৈরি করে। উদাহরণ: ব্যাংক অ্যাকাউন্টে প্রতিদিন ১০০ টাকা করে উত্তোলন স্বাভাবিক মনে হয়, কিন্তু ৩০ দিনে ৩,০০০ টাকা একইভাবে তোলা একটি প্যাটার্ন নির্দেশ করে।
    </p>

    <h3>৩. Z-Score পদ্ধতি</h3>
    <p>
      Z-Score হলো সবচেয়ে সরল পরিসংখ্যান পদ্ধতি। এটি বলে দেয় একটি মান তার গড় থেকে কত স্ট্যান্ডার্ড ডেভিয়েশন দূরে আছে:
    </p>
    <p>
      <strong>Z = (X - μ) / σ</strong>
    </p>
    <p>
      যেখানে X = পর্যবেক্ষিত মান, μ = গড় (mean), σ = স্ট্যান্ডার্ড ডেভিয়েশন।
    </p>
    <p>
      সাধারণ নিয়ম: যদি |Z| > 3, তাহলে সেই পয়েন্টকে অ্যানোমালি ধরা হয়। স্বাভাবিক বিতরণে ~99.7% ডেটা ±3σ-এর মধ্যে থাকে।
    </p>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Sample data তৈরি করা (কিছু আউটলায়ার সহ)
np.random.seed(42)
normal_data = np.random.normal(50, 10, 200)
outliers = np.array([5, 3, 100, 110, 95])
data = np.concatenate([normal_data, outliers])

# Z-Score হিসাব করা
z_scores = np.abs(stats.zscore(data))
threshold = 3.0

# অ্যানোমালি চিহ্নিত করা
anomalies = np.where(z_scores > threshold)[0]
print(f"মোট ডেটা পয়েন্ট: {len(data)}")
print(f"অ্যানোমালি সংখ্যা: {len(anomalies)}")
print(f"অ্যানোমালি মানগুলো: {data[anomalies]}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(range(len(data)), data,
            c=['red' if i in anomalies else 'blue' for i in range(len(data))],
            alpha=0.6, s=20)
plt.title('Z-Score Anomaly Detection')
plt.xlabel('Index')
plt.ylabel('Value')
plt.axhline(y=np.mean(data) + 3*np.std(data), color='red', linestyle='--', label='+3σ')
plt.axhline(y=np.mean(data) - 3*np.std(data), color='red', linestyle='--', label='-3σ')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(z_scores, bins=30, edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.title('Z-Score Distribution')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. IQR (Interquartile Range) পদ্ধতি</h3>
    <p>
      Z-Score সাধারণ বিতরণ (normal distribution) ধরে নেয়, কিন্তু বাস্তব ডেটা সবসময় স্বাভাবিকভাবে বিতরিত নয়। IQR পদ্ধতি আরো robust কারণ এটি non-parametric।
    </p>
    <p>
      <strong>IQR = Q3 - Q1</strong>
    </p>
    <p>
      যেকোনো মান যা নিচের সীমার (Q1 - 1.5 × IQR) নিচে বা উপরের সীমার (Q3 + 1.5 × IQR) উপরে, সেটি আউটলায়ার।
    </p>
    <pre><code>import numpy as np
import pandas as pd

def iqr_outlier_detection(data, multiplier=1.5):
    """IQR পদ্ধতিতে আউটলায়ার ডিটেকশন"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outliers = data[outlier_mask]

    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"নিচের সীমা: {lower_bound:.2f}")
    print(f"উপরের সীমা: {upper_bound:.2f}")
    print(f"আউটলায়ার সংখ্যা: {outlier_mask.sum()}")
    print(f"আউটলায়ার মানগুলো: {outliers}")

    return outlier_mask, lower_bound, upper_bound

# উদাহরণ ডেটা
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 200),
    [5, 3, 100, 110, 95]
])

mask, lb, ub = iqr_outlier_detection(data)

# Box Plot দিয়ে Visualization
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.boxplot(data, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
ax1.set_title('Box Plot (IQR Visualization)')
ax1.set_ylabel('Value')

ax2.scatter(range(len(data)), data,
            c=['red' if m else 'blue' for m in mask],
            alpha=0.6, s=20)
ax2.axhline(y=ub, color='red', linestyle='--', label=f'Upper ({ub:.1f})')
ax2.axhline(y=lb, color='orange', linestyle='--', label=f'Lower ({lb:.1f})')
ax2.set_title('IQR Anomaly Detection')
ax2.legend()
plt.tight_layout()
plt.show()</code></pre>

    <h3>৫. Grubbs Test</h3>
    <p>
      Grubbs Test বিশেষভাবে একটি ডেটাসেটে সর্বোচ্চ বা সর্বনিম্ন মান আউটলায়ার কিনা তা পরীক্ষা করে। এটি একটি formal statistical hypothesis test।
    </p>
    <p>
      <strong>Grubbs Statistic: G = max|Xᵢ - X̄| / s</strong>
    </p>
    <p>
      যেখানে X̄ = sample mean, s = sample standard deviation। যদি G > critical value হয়, তাহলে আউটলায়ার সনাক্ত।
    </p>
    <pre><code>import numpy as np
from scipy import stats

def grubbs_test(data, alpha=0.05):
    """
    Grubbs Test for outlier detection
    H0: কোনো আউটলায়ার নেই
    H1: ঠিক একটি আউটলায়ার আছে
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # G statistic
    g_stat = np.max(np.abs(data - mean)) / std

    # Critical value (two-tailed t-distribution)
    t_critical = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
        t_critical**2 / (n - 2 + t_critical**2)
    )

    outlier_idx = np.argmax(np.abs(data - mean))
    is_outlier = g_stat > g_critical

    print(f"G Statistic: {g_stat:.4f}")
    print(f"Critical Value (alpha={alpha}): {g_critical:.4f}")
    print(f"সন্দেহজনক মান: {data[outlier_idx]:.2f} (index {outlier_idx})")
    print(f"আউটলায়ার? {'হ্যাঁ' if is_outlier else 'না'}")

    return is_outlier, g_stat, g_critical

# পরীক্ষা করা
data = np.array([14.2, 15.1, 14.8, 15.3, 14.6, 15.0, 14.9, 25.8, 15.2])
is_out, g, gc = grubbs_test(data)</code></pre>

    <h3>৬. Modified Z-Score (MAD ভিত্তিক)</h3>
    <p>
      যখন ডেটায় আগে থেকেই অনেক আউটলায়ার থাকে, তখন সাধারণ Z-Score কার্যকর নয় কারণ আউটলায়ারগুলো নিজেই mean ও std পরিবর্তন করে দেয়। Modified Z-Score এই সমস্যা এড়াতে <strong>Median Absolute Deviation (MAD)</strong> ব্যবহার করে।
    </p>
    <pre><code>import numpy as np

def modified_z_score(data, threshold=3.5):
    """
    MAD-ভিত্তিক Modified Z-Score
    সাধারণ Z-Score-এর চেয়ে robust
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Consistency constant = 0.6745
    modified_z = 0.6745 * (data - median) / mad

    outlier_mask = np.abs(modified_z) > threshold
    return outlier_mask, modified_z

# তুলনামূলক বিশ্লেষণ
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 5, 100),
    [80, 85, 90]  # আউটলায়ার
])

# Standard Z-Score
z_scores = np.abs((data - np.mean(data)) / np.std(data))
std_outliers = z_scores > 3

# Modified Z-Score
mod_mask, mod_z = modified_z_score(data)

print(f"Standard Z-Score আউটলায়ার: {std_outliers.sum()}")
print(f"Modified Z-Score আউটলায়ার: {mod_mask.sum()}")
print(f"Standard পায়: {data[std_outliers]}")
print(f"Modified পায়: {data[mod_mask]}")</code></pre>

    <h3>৭. পরিসংখ্যান থ্রেশহোল্ড নির্বাচন</h3>
    <p>
      সঠিক থ্রেশহোল্ড বেছে নেওয়া অ্যানোমালি ডিটেকশনের সবচেয়ে গুরুত্বপূর্ণ অংশ। থ্রেশহোল্ড কম রাখলে বেশি False Positive হয়, বেশি রাখলে Real Anomaly মিস হয়।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

def threshold_analysis(data, true_labels=None):
    """
    বিভিন্ন থ্রেশহোল্ডে অ্যানোমালি সংখ্যা বিশ্লেষণ
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    thresholds = np.linspace(1.0, 5.0, 50)
    n_anomalies = []

    for t in thresholds:
        n_anomalies.append((z_scores > t).sum())

    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, n_anomalies, marker='o', markersize=3)
    plt.axvline(x=2.5, color='orange', linestyle='--', label='threshold=2.5')
    plt.axvline(x=3.0, color='red', linestyle='--', label='threshold=3.0')
    plt.xlabel('Z-Score Threshold')
    plt.ylabel('Detected Anomalies Count')
    plt.title('Threshold vs. Anomaly Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

np.random.seed(42)
data = np.concatenate([np.random.normal(50, 10, 500), [5, 3, 100, 110]])
threshold_analysis(data)</code></pre>

    <h3>৮. কোন পদ্ধতি কখন ব্যবহার করবেন?</h3>
    <p>
      পরিসংখ্যান পদ্ধতিগুলোর মধ্যে সঠিক পদ্ধতি বেছে নেওয়া নির্ভর করে ডেটার ধরনের উপর:
    </p>
    <ul>
      <li><strong>Z-Score:</strong> ডেটা যদি approximately normal distribution হয় এবং আউটলায়ার কম থাকে</li>
      <li><strong>IQR:</strong> ডেটা skewed হলে বা non-normal distribution হলে এটি ভালো কাজ করে</li>
      <li><strong>Modified Z-Score:</strong> ডেটায় অনেক আউটলায়ার থাকলে, বা ডেটা robust analysis চাইলে</li>
      <li><strong>Grubbs Test:</strong> শুধুমাত্র একটি আউটলায়ার আছে কিনা formal পরীক্ষা করতে</li>
    </ul>

    <h3>৯. সীমাবদ্ধতা ও পরবর্তী পদক্ষেপ</h3>
    <p>
      পরিসংখ্যান পদ্ধতির সীমাবদ্ধতা হলো এগুলো <strong>univariate</strong> — একটি ফিচারে ভালো কাজ করে কিন্তু উচ্চ-মাত্রার ডেটায় (multi-dimensional) দুর্বল। উদাহরণস্বরূপ, একজন মানুষের বয়স ৩০ এবং উচ্চতা ৫ ফুট আলাদাভাবে স্বাভাবিক, কিন্তু একসাথে অস্বাভাবিক হতে পারে।
    </p>
    <p>
      জটিল, বহুমাত্রিক ডেটার জন্য মেশিন লার্নিং-ভিত্তিক পদ্ধতি যেমন <strong>Isolation Forest</strong>, <strong>One-Class SVM</strong>, এবং <strong>Local Outlier Factor</strong> ব্যবহার করা হয়, যেগুলো পরবর্তী পর্বে আলোচনা করা হবে।
    </p>
  `
};
