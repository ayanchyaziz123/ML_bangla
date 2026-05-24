export const knn_6_project = {
  title: "End-to-End Project: Wine Quality Classification — KNN vs অন্য Classifiers",
  description: "Wine Quality dataset-এ KNN, SVM, Random Forest তুলনা — সম্পূর্ণ ML pipeline, optimal K selection, feature importance ও final evaluation বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 13,
  slug: "knn-project-wine-classification",
  content: `
    <h3>১. Project Overview</h3>
    <p>UCI Wine Quality dataset ব্যবহার করে red wine-এর quality (3–8 scale) predict করবো। Binary classification: quality ≥ 7 = "good", বাকি = "not good"।</p>
    <table>
      <thead><tr><th>Dataset Info</th><th>মান</th></tr></thead>
      <tbody>
        <tr><td>Samples</td><td>1,599</td></tr>
        <tr><td>Features</td><td>11 (acidity, pH, alcohol, sulphates…)</td></tr>
        <tr><td>Good quality (≥7)</td><td>217 (13.6%)</td></tr>
        <tr><td>Not good (&lt;7)</td><td>1,382 (86.4%)</td></tr>
        <tr><td>Target</td><td>binary (imbalanced)</td></tr>
      </tbody>
    </table>

    <h3>২. ডেটা লোড ও EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df  = pd.read_csv(url, sep=';')

print(df.shape)
print(df['quality'].value_counts().sort_index())

# Binary target
df['good'] = (df['quality'] >= 7).astype(int)
print(f"\nGood wine: {df['good'].sum()} ({df['good'].mean()*100:.1f}%)")

# Feature correlation with quality
corr = df.corr()['quality'].sort_values()
plt.figure(figsize=(8, 5))
corr.drop('quality').plot(kind='barh', color=['crimson' if x < 0 else 'steelblue' for x in corr.drop('quality')])
plt.title('Feature Correlation with Wine Quality')
plt.axvline(0, color='black', lw=0.8)
plt.tight_layout(); plt.show()</code></pre>

    <h3>৩. Preprocessing</h3>
    <pre><code>from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = df.drop(['quality', 'good'], axis=1).values
y = df['good'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")
print(f"Train good: {y_train.sum()},  Test good: {y_test.sum()}")

# KNN-এ scaling অত্যন্ত জরুরি — unscaled-এ alcohol (8–15) vs pH (2.9–3.9) এর scale আলাদা
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"Scaled mean: {X_train_s.mean():.4f}, std: {X_train_s.std():.4f}")</code></pre>

    <h3>৪. Optimal K খোঁজা</h3>
    <pre><code>from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn  = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski')
    pipe = Pipeline([('s', StandardScaler()), ('m', knn)])
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
    cv_scores.append(cv)

best_k = k_range[np.argmax(cv_scores)]
print(f"Best K: {best_k},  CV F1: {max(cv_scores):.4f}")

plt.figure(figsize=(9, 4))
plt.plot(k_range, cv_scores, 'o-', color='steelblue')
plt.axvline(best_k, color='crimson', ls='--', label=f'Best K={best_k}')
plt.xlabel('K'); plt.ylabel('CV F1 Score')
plt.title('K vs Cross-Validated F1')
plt.legend(); plt.grid(alpha=0.3); plt.show()</code></pre>

    <h3>৫. মডেল তুলনা</h3>
    <pre><code>from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

models = {
    f'KNN (K={best_k})':      KNeighborsClassifier(n_neighbors=best_k, weights='distance'),
    'SVM (RBF)':              SVC(C=10, gamma='scale', probability=True, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression':    LogisticRegression(C=1.0, max_iter=1000),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = pipe.decision_function(X_test)
    results[name] = {
        'CV F1 (5-fold)':  cross_val_score(pipe, X, y, cv=5, scoring='f1').mean(),
        'Test Accuracy':   accuracy_score(y_test, y_pred),
        'Test F1':         f1_score(y_test, y_pred),
        'ROC-AUC':         roc_auc_score(y_test, y_prob),
    }

import pandas as pd
results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
print(results_df.round(4))</code></pre>

    <h3>৬. চূড়ান্ত Evaluation</h3>
    <pre><code>from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

# Best model pipeline (KNN with optimal K)
best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  KNeighborsClassifier(n_neighbors=best_k, weights='distance')),
])
best_pipe.fit(X_train, y_train)
y_pred  = best_pipe.predict(X_test)
y_prob  = best_pipe.predict_proba(X_test)[:, 1]

print("=== KNN Final Evaluation ===")
print(classification_report(y_test, y_pred, target_names=['Not Good', 'Good']))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Not Good','Good']).plot(ax=axes[0])
axes[0].set_title(f'KNN (K={best_k}) — Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, lw=2, label=f'KNN AUC={auc:.3f}')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
axes[1].legend()
plt.tight_layout(); plt.show()</code></pre>

    <h3>৭. Imbalanced Data — Class Weight</h3>
    <pre><code># Good wine মাত্র 13.6% — imbalanced dataset
# KNN-এ class_weight নেই, তবে SMOTE বা threshold tuning ব্যবহার করা যায়

# Approach 1: threshold tuning
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_thresh:.2f}, F1: {max(f1_scores):.4f}")

# Approach 2: Random Forest with class_weight='balanced' (KNN-এর চেয়ে এখানে সুবিধাজনক)
rf_bal = Pipeline([
    ('s', StandardScaler()),
    ('m', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
])
rf_bal.fit(X_train, y_train)
print(f"RF balanced F1: {f1_score(y_test, rf_bal.predict(X_test)):.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>মডেল</th><th>সুবিধা</th><th>অসুবিধা</th><th>Wine-এ পারফরম্যান্স</th></tr></thead>
      <tbody>
        <tr><td>KNN</td><td>Simple, non-parametric</td><td>Slow prediction, no class_weight</td><td>ভালো — সঠিক K ও scaling-এ</td></tr>
        <tr><td>SVM (RBF)</td><td>High margin, robust</td><td>Slow large data</td><td>ভালো</td></tr>
        <tr><td>Random Forest</td><td>class_weight, feature importance</td><td>Black box</td><td>সেরা imbalanced data-তে</td></tr>
        <tr><td>Gradient Boosting</td><td>Most accurate</td><td>Slow train</td><td>সবচেয়ে ভালো overall</td></tr>
        <tr><td>Logistic Regression</td><td>Interpretable, fast</td><td>Linear boundary</td><td>Baseline হিসেবে ভালো</td></tr>
      </tbody>
    </table>
  `,
};
