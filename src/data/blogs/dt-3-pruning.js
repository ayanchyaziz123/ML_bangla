export const dt_3_pruning = {
  title: "Overfitting ও Pruning — Decision Tree-র গাছ ছাঁটাই",
  description: "Decision Tree কেন Overfit করে, Pre-pruning ও Post-pruning পদ্ধতি, Cost Complexity Pruning — গাছকে সঠিক আকারে রাখার সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 10,
  slug: "dt-overfitting-pruning",
  content: `
    <h3>১. Decision Tree কেন Overfit করে?</h3>
    <p>কোনো constraint ছাড়া Decision Tree প্রতিটি training sample-কে আলাদা leaf-এ রাখতে পারে — Train Accuracy = 100%, কিন্তু নতুন ডেটায় ব্যর্থ। এটাকে বলে <strong>Overfitting বা High Variance</strong>।</p>
    <pre><code>from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# No constraint
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

print(f"Train Accuracy: {dt_full.score(X_train, y_train):.4f}")  # 1.0000
print(f"Test  Accuracy: {dt_full.score(X_test,  y_test):.4f}")   # কম
print(f"Tree depth:     {dt_full.get_depth()}")
print(f"Leaf count:     {dt_full.get_n_leaves()}")</code></pre>

    <h3>২. Pre-pruning — আগে থেকেই সীমা দাও</h3>
    <p>Tree তৈরির সময়ই থামার শর্ত দিয়ে দেওয়া।</p>
    <pre><code">train_accs, test_accs, cv_accs = [], [], []
depths = range(1, 20)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(dt.score(X_train, y_train))
    test_accs.append(dt.score(X_test, y_test))
    cv_accs.append(cross_val_score(dt, X, y, cv=5).mean())

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, 'o-', label='Train')
plt.plot(depths, test_accs,  's-', label='Test')
plt.plot(depths, cv_accs,    '^-', label='CV (5-fold)')
plt.axvline(np.argmax(cv_accs)+1, color='red', linestyle='--',
            label=f'Best depth={np.argmax(cv_accs)+1}')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Pre-pruning: max_depth প্রভাব')
plt.legend()
plt.show()

# Pre-pruning parameters:
dt_pruned = DecisionTreeClassifier(
    max_depth=5,             # গভীরতা সীমিত
    min_samples_split=10,    # split করতে কমপক্ষে ১০টি sample
    min_samples_leaf=5,      # leaf-এ কমপক্ষে ৫টি sample
    max_leaf_nodes=20,       # সর্বোচ্চ leaf সংখ্যা
    max_features='sqrt',     # প্রতি split-এ random feature subset
    random_state=42,
)
dt_pruned.fit(X_train, y_train)
print(f"Pruned Train: {dt_pruned.score(X_train, y_train):.4f}")
print(f"Pruned Test:  {dt_pruned.score(X_test,  y_test):.4f}")</code></pre>

    <h3>৩. Post-pruning — Cost Complexity Pruning (CCP)</h3>
    <p>আগে পূর্ণ tree তৈরি করো, তারপর দুর্বল branch ছেঁটে ফেলো। sklearn এটিকে বলে <strong>Minimal Cost-Complexity Pruning</strong>।</p>
    <pre><code">from sklearn.tree import DecisionTreeClassifier

# Step 1: সব সম্ভব ccp_alpha খোঁজো
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(
    X_train, y_train
)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"মোট {len(ccp_alphas)}টি alpha মান পাওয়া গেছে")
print(f"alpha range: {ccp_alphas[0]:.6f} থেকে {ccp_alphas[-1]:.6f}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">CCP: alpha বাড়লে tree ছোট হয়</text>
        <!-- tree shapes shrinking -->
        <polygon points="60,95 40,95 50,40" fill="#dbeafe" stroke="#93c5fd"/>
        <line x1="45" y1="78" x2="35" y2="95" stroke="#93c5fd"/>
        <line x1="55" y1="78" x2="65" y2="95" stroke="#93c5fd"/>
        <line x1="42" y1="62" x2="30" y2="78" stroke="#93c5fd"/>
        <line x1="48" y1="62" x2="58" y2="78" stroke="#93c5fd"/>
        <text x="50" y="105" text-anchor="middle" font-size="9" fill="#6b7280">alpha=0</text>
        <polygon points="180,95 160,95 170,55" fill="#dbeafe" stroke="#93c5fd"/>
        <line x1="165" y1="82" x2="155" y2="95" stroke="#93c5fd"/>
        <line x1="175" y1="82" x2="185" y2="95" stroke="#93c5fd"/>
        <text x="170" y="105" text-anchor="middle" font-size="9" fill="#6b7280">alpha=0.002</text>
        <polygon points="290,95 270,95 280,65" fill="#dbeafe" stroke="#93c5fd"/>
        <line x1="275" y1="85" x2="265" y2="95" stroke="#93c5fd"/>
        <line x1="285" y1="85" x2="295" y2="95" stroke="#93c5fd"/>
        <text x="280" y="105" text-anchor="middle" font-size="9" fill="#6b7280">alpha=0.01</text>
        <polygon points="410,95 390,95 400,72" fill="#dcfce7" stroke="#86efac"/>
        <text x="400" y="105" text-anchor="middle" font-size="9" fill="#16a34a">alpha=0.05</text>
        <text x="400" y="88" text-anchor="middle" font-size="8" fill="#166534">1 leaf!</text>
        <text x="240" y="25" text-anchor="middle" font-size="9" fill="#6b7280">→ alpha বাড়ো → tree ছোট হয় → সেরা alpha cross-validation দিয়ে খোঁজো</text>
      </svg>
    </div>

    <pre><code"># Step 2: প্রতিটি alpha দিয়ে tree তৈরি করো
clfs = []
for alpha in ccp_alphas[:-1]:   # শেষেরটা শুধু root, বাদ দাও
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Step 3: সেরা alpha খোঁজো
train_scores = [c.score(X_train, y_train) for c in clfs]
test_scores  = [c.score(X_test,  y_test)  for c in clfs]
cv_scores    = [cross_val_score(c, X, y, cv=5).mean() for c in clfs]

best_idx = np.argmax(cv_scores)
best_alpha = ccp_alphas[best_idx]
print(f"সেরা ccp_alpha: {best_alpha:.6f}")
print(f"Test Accuracy: {test_scores[best_idx]:.4f}")

# Step 4: সেরা alpha দিয়ে চূড়ান্ত model
best_dt = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
best_dt.fit(X_train, y_train)
print(f"Leaf count: {best_dt.get_n_leaves()}")</code></pre>

    <h3>৪. Pre-pruning vs Post-pruning তুলনা</h3>
    <table>
      <thead><tr><th></th><th>Pre-pruning</th><th>Post-pruning (CCP)</th></tr></thead>
      <tbody>
        <tr><td><strong>কখন</strong></td><td>tree তৈরির সময়</td><td>tree তৈরির পর</td></tr>
        <tr><td><strong>Parameter</strong></td><td>max_depth, min_samples</td><td>ccp_alpha</td></tr>
        <tr><td><strong>সুবিধা</strong></td><td>দ্রুত, সহজ</td><td>সর্বোত্তম tree খোঁজে</td></tr>
        <tr><td><strong>অসুবিধা</strong></td><td>সঠিক parameter বেছে নিতে হয়</td><td>ধীর (অনেক tree তৈরি করে)</td></tr>
        <tr><td><strong>সুপারিশ</strong></td><td>প্রথম চেষ্টায়</td><td>সূক্ষ্ম tuning-এ</td></tr>
      </tbody>
    </table>

    <h3>৫. Validation Curve দিয়ে সেরা depth খোঁজা</h3>
    <pre><code">from sklearn.model_selection import validation_curve

depths = np.arange(1, 15)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=depths,
    cv=5,
    scoring='accuracy',
)

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(depths, val_scores.mean(axis=1),   's-', label='CV Validation')
plt.fill_between(depths,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.2)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve — max_depth')
plt.legend()
plt.show()</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Overfitting</td><td>constraint ছাড়া tree → Train=100%, Test কম</td></tr>
        <tr><td>Pre-pruning</td><td>max_depth, min_samples দিয়ে আগেই থামো</td></tr>
        <tr><td>CCP (Post-pruning)</td><td>ccp_alpha দিয়ে দুর্বল branch ছাঁটাও</td></tr>
        <tr><td>সেরা alpha</td><td>cost_complexity_pruning_path + CV দিয়ে খোঁজো</td></tr>
        <tr><td>Validation Curve</td><td>parameter প্রভাব ভিজুয়ালি দেখো</td></tr>
      </tbody>
    </table>
  `,
};
