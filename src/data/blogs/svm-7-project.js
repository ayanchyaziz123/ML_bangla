export const svm_7_project = {
  title: "End-to-End Project: Cancer Detection — SVM vs অন্য Classifiers",
  description: "Breast Cancer dataset-এ SVM, Random Forest, Logistic Regression তুলনা করো — সম্পূর্ণ ML pipeline, hyperparameter tuning ও model selection বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "সাপোর্ট ভেক্টর মেশিন",
  readTime: 14,
  slug: "svm-project-cancer-detection",
  content: `
    <h3>১. Project Overview</h3>
    <p>Wisconsin Breast Cancer Dataset ব্যবহার করে malignant (ক্যান্সার) এবং benign (সৌম্য) tumor classify করবো।</p>
    <table>
      <thead><tr><th>Dataset Info</th><th>মান</th></tr></thead>
      <tbody>
        <tr><td>Samples</td><td>569</td></tr>
        <tr><td>Features</td><td>30 (nucleus-এর geometric measurements)</td></tr>
        <tr><td>Malignant</td><td>212 (37.3%)</td></tr>
        <tr><td>Benign</td><td>357 (62.7%)</td></tr>
        <tr><td>Target</td><td>binary: 0=malignant, 1=benign</td></tr>
      </tbody>
    </table>

    <h3>২. ডেটা লোড ও EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.shape)
print(df['target'].value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# Feature distributions by class
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for i, col in enumerate(data.feature_names[:15]):
    ax = axes[i//5, i%5]
    df[df['target']==0][col].hist(ax=ax, bins=25, alpha=0.6,
        color='crimson', label='Malignant', density=True)
    df[df['target']==1][col].hist(ax=ax, bins=25, alpha=0.6,
        color='steelblue', label='Benign', density=True)
    ax.set_title(col[:20], fontsize=8)
    ax.legend(fontsize=6)
plt.suptitle('Feature Distributions: Malignant vs Benign', fontsize=12)
plt.tight_layout(); plt.show()

# Correlation heatmap (top features)
corr = df[list(data.feature_names[:15]) + ['target']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()</code></pre>

    <h3>৩. Preprocessing Pipeline</h3>
    <pre><code">from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train malignant: {(y_train==0).sum()}, benign: {(y_train==1).sum()}")

# Feature variance analysis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print(f"\nAfter scaling — Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")</code></pre>

    <h3>৪. পাঁচটি মডেল একসাথে তুলনা</h3>
    <pre><code">from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import cross_val_score
import pandas as pd

models = {
    'SVM (RBF)':            SVC(kernel='rbf', C=10, gamma='scale',
                                probability=True, random_state=42),
    'SVM (Linear)':         LinearSVC(C=1.0, max_iter=5000),
    'Logistic Regression':  LogisticRegression(C=1.0, max_iter=1000),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Gaussian NB':          GaussianNB(),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv_acc = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

    try:
        y_prob = pipe.predict_proba(X_test)[:,1]
        auc    = roc_auc_score(y_test, y_prob)
    except AttributeError:
        y_score = pipe.decision_function(X_test)
        auc     = roc_auc_score(y_test, y_score)

    results[name] = {
        'CV Accuracy': cv_acc,
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'F1 (Malignant)': f1_score(y_test, y_pred, pos_label=0),
        'ROC-AUC': auc,
    }

results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
print(results_df.round(4))</code></pre>

    <h3>৫. SVM Hyperparameter Tuning</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(probability=True, random_state=42)),
])

param_dist = {
    'svm__kernel': ['rbf', 'linear'],
    'svm__C':      loguniform(0.1, 1000),
    'svm__gamma':  loguniform(1e-5, 10),  # rbf-এর জন্য
}

search = RandomizedSearchCV(
    pipe_svm, param_dist,
    n_iter=50, cv=5,
    scoring='roc_auc',
    random_state=42, n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

print(f"সেরা params:      {search.best_params_}")
print(f"CV AUC:           {search.best_score_:.4f}")
print(f"Test AUC:         {roc_auc_score(y_test, search.predict_proba(X_test)[:,1]):.4f}")
print(f"Test Accuracy:    {search.score(X_test, y_test):.4f}")</code></pre>

    <h3>৬. চূড়ান্ত Evaluation</h3>
    <pre><code">from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt

best_model = search.best_estimator_
y_pred_final = best_model.predict(X_test)
y_prob_final = best_model.predict_proba(X_test)[:,1]

print("=== Final SVM Performance ===")
print(classification_report(y_test, y_pred_final,
      target_names=['Malignant','Benign']))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm, display_labels=['Malignant','Benign']).plot(ax=axes[0])
axes[0].set_title('Confusion Matrix')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_final)
auc = roc_auc_score(y_test, y_prob_final)
axes[1].plot(fpr, tpr, lw=2, label=f'AUC={auc:.3f}')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
axes[1].legend()

# 3. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob_final, pos_label=0)
axes[2].plot(rec, prec, lw=2)
axes[2].set(xlabel='Recall', ylabel='Precision', title='PR Curve (Malignant)')

plt.tight_layout(); plt.show()</code></pre>

    <h3>৭. Error Analysis</h3>
    <pre><code"># False Negative = malignant → benign বলেছে (সবচেয়ে বিপদজনক!)
# False Positive = benign → malignant বলেছে (অপ্রয়োজনীয় উদ্বেগ)

import numpy as np

fn_mask = (y_test == 0) & (y_pred_final == 1)  # missed cancers
fp_mask = (y_test == 1) & (y_pred_final == 0)  # false alarms

print(f"False Negatives (missed cancer): {fn_mask.sum()}")
print(f"False Positives (false alarm):   {fp_mask.sum()}")

# Lower threshold to catch more cancers (improve Recall):
threshold = 0.3  # default 0.5
y_pred_low_thresh = (y_prob_final < threshold).astype(int)
print(f"\nWith threshold={threshold}:")
print(f"  False Negatives: {((y_test==0) & (y_pred_low_thresh==1)).sum()}")
print(classification_report(y_test, y_pred_low_thresh,
      target_names=['Malignant','Benign']))</code></pre>

    <h3>সারসংক্ষেপ — মডেল তুলনা</h3>
    <table>
      <thead><tr><th>মডেল</th><th>সুবিধা</th><th>অসুবিধা</th><th>Cancer Detection-এ</th></tr></thead>
      <tbody>
        <tr><td>SVM (RBF)</td><td>High accuracy, non-linear</td><td>ধীর large data-তে</td><td>ভালো — small dataset</td></tr>
        <tr><td>SVM (Linear)</td><td>দ্রুত, interpretable</td><td>Linear only</td><td>ভালো — high feature count</td></tr>
        <tr><td>Logistic Regression</td><td>Probability, interpretable</td><td>Linear boundary</td><td>Baseline হিসেবে ভালো</td></tr>
        <tr><td>Random Forest</td><td>Robust, feature importance</td><td>Slow prediction</td><td>ভালো — explainability দরকার</td></tr>
        <tr><td>Gradient Boosting</td><td>সবচেয়ে accurate</td><td>Hyperparameter বেশি</td><td>Production-এ সেরা</td></tr>
      </tbody>
    </table>
  `,
};
