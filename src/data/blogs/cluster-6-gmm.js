export const cluster_6_gmm = {
  title: "Gaussian Mixture Models (GMM): Soft Clustering ও Probability",
  description: "K-Means-এর সীমাবদ্ধতা কাটিয়ে GMM দিয়ে probabilistic soft clustering — EM algorithm, covariance types, BIC/AIC দিয়ে component selection বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 11,
  slug: "clustering-gmm",
  content: `
    <h3>১. K-Means-এর সীমাবদ্ধতা ও GMM-এর ধারণা</h3>
    <p>K-Means প্রতিটি point-কে একটি cluster-এ <strong>hard assign</strong> করে। কিন্তু বাস্তবে অনেক point দুটো cluster-এর সীমানায় থাকে। GMM প্রতিটি point-কে প্রতিটি cluster-এ <strong>কতটুকু সম্ভাবনায়</strong> belong করে তা বলে — এটাই <em>soft clustering</em>।</p>
    <pre><code># K-Means: point X → cluster 2 (100% নিশ্চিত)
# GMM:    point X → cluster 1: 30%, cluster 2: 65%, cluster 3: 5%

# GMM ধরে নেয় data k-টি Gaussian distribution-এর mixture থেকে এসেছে:
# P(x) = Σ_k  π_k * N(x | μ_k, Σ_k)
#
# π_k = mixing weight (sum = 1)
# μ_k = k-তম Gaussian-এর mean
# Σ_k = k-তম Gaussian-এর covariance matrix</code></pre>

    <h3>২. EM Algorithm — GMM কীভাবে শেখে</h3>
    <p>GMM শেখে <strong>Expectation-Maximization (EM)</strong> দিয়ে:</p>
    <table>
      <thead><tr><th>Step</th><th>কী করে</th><th>K-Means-এর সাথে মিল</th></tr></thead>
      <tbody>
        <tr><td><strong>E-step</strong> (Expectation)</td><td>প্রতিটি point-এর জন্য প্রতিটি Gaussian-এ soft responsibility r(i,k) হিসাব করে</td><td>Assignment step</td></tr>
        <tr><td><strong>M-step</strong> (Maximization)</td><td>responsibilities দিয়ে μ_k, Σ_k, π_k update করে</td><td>Centroid update</td></tr>
        <tr><td><strong>Convergence</strong></td><td>log-likelihood আর বাড়ে না</td><td>Centroids stable হলে</td></tr>
      </tbody>
    </table>

    <h3>৩. Covariance Type-এর প্রভাব</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)

cov_types = ['full', 'tied', 'diag', 'spherical']
fig, axes  = plt.subplots(1, 4, figsize=(16, 4))

for ax, cov in zip(axes, cov_types):
    gmm = GaussianMixture(n_components=4, covariance_type=cov, random_state=42)
    labels = gmm.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
    ax.set_title(f'{cov}\nBIC={gmm.bic(X):.0f}', fontsize=9)
    ax.axis('off')

plt.suptitle('GMM — Covariance Types তুলনা', fontsize=12)
plt.tight_layout(); plt.show()</code></pre>
    <table>
      <thead><tr><th>Covariance Type</th><th>আকার</th><th>Parameters</th><th>কখন ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td><strong>full</strong></td><td>যেকোনো ellipse</td><td>সবচেয়ে বেশি</td><td>Clusters-এর আলাদা আলাদা shape</td></tr>
        <tr><td><strong>tied</strong></td><td>একই ellipse সব cluster-এ</td><td>মাঝারি</td><td>Clusters same shape কিন্তু আলাদা size</td></tr>
        <tr><td><strong>diag</strong></td><td>axis-aligned ellipse</td><td>কম</td><td>Features uncorrelated</td></tr>
        <tr><td><strong>spherical</strong></td><td>বৃত্তাকার (K-Means-এর মতো)</td><td>সবচেয়ে কম</td><td>Simple, circular clusters</td></tr>
      </tbody>
    </table>

    <h3>৪. BIC ও AIC দিয়ে Component সংখ্যা নির্বাচন</h3>
    <pre><code>from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Component সংখ্যা নির্বাচনে BIC (Bayesian Information Criterion) ব্যবহার
# BIC কম = ভালো মডেল (goodness-of-fit + complexity penalty)

n_components_range = range(1, 11)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full',
                          random_state=42, n_init=3)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

best_n_bic = n_components_range[np.argmin(bic_scores)]
best_n_aic = n_components_range[np.argmin(aic_scores)]
print(f"BIC সেরা component: {best_n_bic}")
print(f"AIC সেরা component: {best_n_aic}")

plt.figure(figsize=(8, 4))
plt.plot(n_components_range, bic_scores, 'o-', label='BIC')
plt.plot(n_components_range, aic_scores, 's-', label='AIC')
plt.axvline(best_n_bic, color='crimson', ls='--', label=f'Best BIC n={best_n_bic}')
plt.xlabel('Number of Components'); plt.ylabel('Score')
plt.title('GMM: BIC ও AIC vs Components')
plt.legend(); plt.grid(alpha=0.3); plt.show()</code></pre>

    <h3>৫. Soft Probability — GMM-এর বিশেষত্ব</h3>
    <pre><code">from sklearn.mixture import GaussianMixture
import numpy as np

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)

# Hard labels (argmax)
hard_labels = gmm.predict(X)

# Soft probabilities — GMM-এর মূল সুবিধা
soft_probs = gmm.predict_proba(X)  # shape: (n_samples, n_components)

print("প্রথম ৫টি point-এর cluster probabilities:")
for i in range(5):
    probs = soft_probs[i]
    print(f"  Point {i}: {[f'{p:.2f}' for p in probs]} → cluster {hard_labels[i]}")

# Anomaly detection: low probability = outlier
log_prob = gmm.score_samples(X)  # log-likelihood per sample
threshold = np.percentile(log_prob, 5)  # সবচেয়ে কম 5%
anomalies = X[log_prob < threshold]
print(f"\nসম্ভাব্য anomalies: {len(anomalies)}")</code></pre>

    <h3>৬. GMM vs K-Means — কখন কোনটি</h3>
    <pre><code">from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X_moon, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# K-Means (spherical assumption fails on moons)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
axes[0].scatter(X_moon[:,0], X_moon[:,1], c=km.fit_predict(X_moon), cmap='bwr', s=20)
axes[0].set_title('K-Means — Moon Dataset (ব্যর্থ)')

# GMM full covariance (still struggles but better)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
axes[1].scatter(X_moon[:,0], X_moon[:,1], c=gmm.fit_predict(X_moon), cmap='bwr', s=20)
axes[1].set_title(f'GMM full — BIC={gmm.bic(X_moon):.0f}')

plt.suptitle('Non-spherical data-তে K-Means vs GMM')
plt.tight_layout(); plt.show()</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>K-Means</th><th>GMM</th></tr></thead>
      <tbody>
        <tr><td>Assignment type</td><td>Hard (0 বা 1)</td><td>Soft (probability)</td></tr>
        <tr><td>Cluster shape</td><td>Spherical only</td><td>Elliptical (covariance দিয়ে)</td></tr>
        <tr><td>Model selection</td><td>Elbow / Silhouette</td><td>BIC / AIC</td></tr>
        <tr><td>Anomaly detection</td><td>সম্ভব (centroid distance)</td><td>ভালো (log-likelihood)</td></tr>
        <tr><td>Computational cost</td><td>কম</td><td>বেশি (EM iteration)</td></tr>
        <tr><td>কখন ব্যবহার</td><td>Simple, spherical clusters</td><td>Overlapping, elliptical, probability দরকার</td></tr>
      </tbody>
    </table>
  `,
};
