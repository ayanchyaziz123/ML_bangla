export const cluster_5_project = {
  title: "Customer Segmentation: K-Means ও DBSCAN দিয়ে সম্পূর্ণ প্রজেক্ট",
  description:
    "Mall Customer Segmentation প্রজেক্ট — EDA থেকে K-Means clustering, cluster interpretation, DBSCAN তুলনা, এবং business insight পর্যন্ত সম্পূর্ণ pipeline।",
  date: "২৩ মে, ২০২৬",
  category: "ক্লাস্টারিং",
  readTime: 13,
  slug: "clustering-project",
  content: `
    <h3>১. প্রজেক্ট পরিচিতি: Customer Segmentation কেন?</h3>
    <p>
      একটি মলের বিপণন দল চায় তাদের গ্রাহকদের বিভিন্ন গ্রুপে ভাগ করতে — যাতে প্রতিটি গ্রুপকে আলাদা কৌশলে টার্গেট করা যায়। এটিই হলো <strong>Customer Segmentation</strong>, এবং Clustering এই কাজের জন্য আদর্শ।
    </p>
    <p>
      এই প্রজেক্টে আমরা দুটি feature ব্যবহার করব: <br/>
      <strong>Annual Income (k$):</strong> গ্রাহকের বার্ষিক আয় <br/>
      <strong>Spending Score (1-100):</strong> মলের দেওয়া score — গ্রাহক কতটা খরচ করে তার ভিত্তিতে
    </p>
    <p>
      পুরো pipeline: ডেটা তৈরি → EDA → Feature Scaling → Optimal K খোঁজা → K-Means → Cluster Interpretation → DBSCAN তুলনা → Business Insight।
    </p>

    <h3>২. ডেটা তৈরি ও Exploratory Data Analysis (EDA)</h3>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# Synthetic mall customers data তৈরি
# ৫টি স্বাভাবিক গ্রুপ: conservative, target, careful, spendthrift, sensible
n = 40
income_groups = [
    (20, 5), (20, 5), (55, 8),  # কম আয়
    (85, 8), (85, 8)             # বেশি আয়
]
spending_groups = [
    (75, 8), (25, 8), (50, 10),  # বিভিন্ন spending
    (80, 8), (20, 8)
]

incomes, scores = [], []
for (inc_mean, inc_std), (sc_mean, sc_std) in zip(income_groups, spending_groups):
    incomes.extend(np.random.normal(inc_mean, inc_std, n).clip(10, 120))
    scores.extend(np.random.normal(sc_mean, sc_std, n).clip(1, 100))

df = pd.DataFrame({
    'CustomerID': range(1, len(incomes) + 1),
    'Annual_Income': np.round(incomes, 1),
    'Spending_Score': np.round(scores, 1)
})

print("Dataset Info:")
print(df.describe())
print(f"\nTotal customers: {len(df)}")

# EDA: Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(df['Annual_Income'], bins=20, color='#3498db',
             edgecolor='white', alpha=0.8)
axes[0].set_title('Annual Income Distribution', fontsize=12)
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Count')

axes[1].hist(df['Spending_Score'], bins=20, color='#e74c3c',
             edgecolor='white', alpha=0.8)
axes[1].set_title('Spending Score Distribution', fontsize=12)
axes[1].set_xlabel('Spending Score (1-100)')

axes[2].scatter(df['Annual_Income'], df['Spending_Score'],
                alpha=0.6, color='#9b59b6', s=50)
axes[2].set_title('Income vs Spending Score', fontsize=12)
axes[2].set_xlabel('Annual Income (k$)')
axes[2].set_ylabel('Spending Score')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Customer Data: Exploratory Analysis', fontsize=14)
plt.tight_layout()
plt.savefig('customer_eda.png', dpi=150)
plt.show()</code></pre>

    <h3>৩. Feature Scaling ও Optimal K নির্বাচন</h3>
    <pre><code># Feature selection ও scaling
X = df[['Annual_Income', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling এর আগে:")
print(f"  Income range: {X[:, 0].min():.1f} — {X[:, 0].max():.1f}")
print(f"  Score range: {X[:, 1].min():.1f} — {X[:, 1].max():.1f}")
print("\nScaling এর পরে:")
print(f"  Income range: {X_scaled[:, 0].min():.2f} — {X_scaled[:, 0].max():.2f}")
print(f"  Score range: {X_scaled[:, 1].min():.2f} — {X_scaled[:, 1].max():.2f}")

# Elbow Method + Silhouette Score
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++',
                n_init=15, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method', fontsize=13)
axes[0].set_xlabel('K (ক্লাস্টার সংখ্যা)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].plot(list(K_range), silhouette_scores, 'ro-', linewidth=2, markersize=8)
best_k = list(K_range)[np.argmax(silhouette_scores)]
axes[1].axvline(x=best_k, color='blue', linestyle='--',
                label=f'Best K={best_k}')
axes[1].set_title('Silhouette Score', fontsize=13)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.suptitle(f'Optimal K নির্বাচন (Best = {best_k})', fontsize=14)
plt.tight_layout()
plt.savefig('optimal_k.png', dpi=150)
plt.show()
print(f"\nElbow ও Silhouette উভয়ে suggest করছে K={best_k}")</code></pre>

    <h3>৪. K-Means Clustering ও Cluster Interpretation</h3>
    <pre><code># Final K-Means model
kmeans_final = KMeans(n_clusters=5, init='k-means++',
                      n_init=15, random_state=42)
df['KMeans_Cluster'] = kmeans_final.fit_predict(X_scaled)
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Cluster statistics
print("Cluster Statistics:")
cluster_stats = df.groupby('KMeans_Cluster').agg({
    'Annual_Income': ['mean', 'std', 'count'],
    'Spending_Score': ['mean', 'std']
}).round(2)
print(cluster_stats)

# Cluster names দেওয়া (centroid দেখে)
cluster_names = {}
for i, (inc, sc) in enumerate(centroids_original):
    if inc < 45 and sc > 60:
        cluster_names[i] = 'Impulsive (কম আয়, বেশি খরচ)'
    elif inc < 45 and sc < 40:
        cluster_names[i] = 'Conservative (কম আয়, কম খরচ)'
    elif inc > 65 and sc > 60:
        cluster_names[i] = 'Target (বেশি আয়, বেশি খরচ)'
    elif inc > 65 and sc < 40:
        cluster_names[i] = 'Careful (বেশি আয়, কম খরচ)'
    else:
        cluster_names[i] = 'Sensible (মধ্যম আয় ও খরচ)'

print("\nCluster Interpretations:")
for k, name in cluster_names.items():
    print(f"  Cluster {k}: {name}")
    print(f"    Income: {centroids_original[k][0]:.1f}k$, "
          f"Score: {centroids_original[k][1]:.1f}")

# Visualization
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
plt.figure(figsize=(11, 8))

for i in range(5):
    mask = df['KMeans_Cluster'] == i
    label = cluster_names.get(i, f'Cluster {i}')
    plt.scatter(df[mask]['Annual_Income'],
                df[mask]['Spending_Score'],
                c=colors[i], s=60, alpha=0.8,
                label=label, edgecolors='white', linewidths=0.5)

# Centroids
plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
            c='black', marker='*', s=300,
            label='Centroids', zorder=6)

plt.title('Customer Segmentation — K-Means (K=5)', fontsize=15)
plt.xlabel('Annual Income (k$)', fontsize=13)
plt.ylabel('Spending Score (1-100)', fontsize=13)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_kmeans.png', dpi=150)
plt.show()</code></pre>

    <h3>৫. DBSCAN তুলনা ও Business Insight</h3>
    <pre><code># DBSCAN: eps নির্বাচন (k-distance graph)
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_dist = np.sort(distances[:, 4])[::-1]

plt.figure(figsize=(8, 4))
plt.plot(k_dist, linewidth=2, color='#e74c3c')
plt.axhline(y=0.5, color='blue', linestyle='--', label='eps=0.5')
plt.title('K-Distance Graph (k=5)', fontsize=13)
plt.xlabel('Points (sorted)')
plt.ylabel('5th Nearest Neighbor Distance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('customer_kdistance.png', dpi=150)
plt.show()

# DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df['DBSCAN_Cluster'])) - (
    1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise_db = (df['DBSCAN_Cluster'] == -1).sum()

print(f"DBSCAN: {n_clusters_db} clusters, {n_noise_db} noise points")

# K-Means vs DBSCAN side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# K-Means
for i in range(5):
    mask = df['KMeans_Cluster'] == i
    axes[0].scatter(df[mask]['Annual_Income'],
                    df[mask]['Spending_Score'],
                    c=colors[i], s=55, alpha=0.8,
                    label=cluster_names.get(i, f'C{i}'))
axes[0].scatter(centroids_original[:, 0], centroids_original[:, 1],
                c='black', marker='*', s=250, label='Centroids', zorder=5)
axes[0].set_title('K-Means (K=5)', fontsize=13)
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# DBSCAN
db_colors = {-1: '#999999', 0: '#e74c3c', 1: '#3498db',
              2: '#2ecc71', 3: '#f39c12', 4: '#9b59b6'}
for lbl in sorted(df['DBSCAN_Cluster'].unique()):
    mask = df['DBSCAN_Cluster'] == lbl
    name = 'Noise' if lbl == -1 else f'Cluster {lbl+1}'
    marker = 'x' if lbl == -1 else 'o'
    axes[1].scatter(df[mask]['Annual_Income'],
                    df[mask]['Spending_Score'],
                    c=db_colors.get(lbl, '#333333'),
                    s=55, alpha=0.8, label=name, marker=marker)
axes[1].set_title(f'DBSCAN ({n_clusters_db} clusters, {n_noise_db} noise)',
                   fontsize=13)
axes[1].set_xlabel('Annual Income (k$)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Customer Segmentation: K-Means vs DBSCAN', fontsize=14)
plt.tight_layout()
plt.savefig('kmeans_vs_dbscan_customers.png', dpi=150)
plt.show()

# Business Insight Summary
print("\n" + "="*55)
print("BUSINESS INSIGHT SUMMARY")
print("="*55)
insights = {
    'Target (বেশি আয়, বেশি খরচ)':
        'সর্বোচ্চ মূল্যের গ্রাহক। প্রিমিয়াম অফার ও loyalty program।',
    'Conservative (কম আয়, কম খরচ)':
        'বাজেট-সচেতন। ছাড় ও কিস্তি সুবিধা কার্যকর।',
    'Impulsive (কম আয়, বেশি খরচ)':
        'আবেগী ক্রেতা। Flash sale ও limited-time offers।',
    'Careful (বেশি আয়, কম খরচ)':
        'মূল্য-সচেতন ধনী। উচ্চমানের পণ্যের ROI দেখান।',
    'Sensible (মধ্যম)':
        'ভারসাম্যপূর্ণ গ্রাহক। সাধারণ promotional campaigns।',
}
for segment, strategy in insights.items():
    print(f"\n{segment}:")
    print(f"  কৌশল: {strategy}")</code></pre>

    <h3>৬. প্রজেক্ট থেকে মূল শিক্ষা</h3>
    <table>
      <thead>
        <tr>
          <th>দিক</th>
          <th>K-Means</th>
          <th>DBSCAN</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>এই ডেটায় ফলাফল</td>
          <td>৫টি পরিষ্কার গোলাকার cluster</td>
          <td>Noise সহ cluster, সীমানা ভিন্ন</td>
        </tr>
        <tr>
          <td>Business Interpretation</td>
          <td>সহজ, প্রতিটি cluster স্পষ্ট</td>
          <td>Outlier গ্রাহকদের আলাদা করে</td>
        </tr>
        <tr>
          <td>Scaling প্রয়োজন?</td>
          <td>হ্যাঁ (জরুরি)</td>
          <td>হ্যাঁ (জরুরি)</td>
        </tr>
        <tr>
          <td>এই ডেটায় কোনটি ভালো?</td>
          <td>K-Means (গোলাকার cluster)</td>
          <td>DBSCAN (outlier detection-এ)</td>
        </tr>
      </tbody>
    </table>
    <p>
      এই প্রজেক্টে আমরা দেখলাম কীভাবে clustering একটি real business problem সমাধান করতে পারে। মূল takeaway: সমস্যার ধরন ও ডেটার structure বুঝে algorithm বেছে নিন। Feature scaling সবসময় করুন। একাধিক metric দিয়ে evaluate করুন। এবং cluster interpretation সবসময় domain knowledge ব্যবহার করে করুন।
    </p>
  `,
};
