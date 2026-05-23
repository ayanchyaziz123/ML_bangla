export const logit_6_project = {
  title: "End-to-End Project: ডায়াবেটিস ভবিষ্যদ্বাণী — Logistic Regression",
  description: "Pima Indians Diabetes ডেটাসেট দিয়ে সম্পূর্ণ ML pipeline — EDA, preprocessing, model training, evaluation এবং deployment-ready কোড বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লজিস্টিক রিগ্রেশন",
  readTime: 14,
  slug: "logit-project-diabetes",
  content: `
    <h3>১. Project Overview</h3>
    <p>আমরা <strong>Pima Indians Diabetes Dataset</strong> ব্যবহার করে ভবিষ্যদ্বাণী করবো — একজন ব্যক্তির ডায়াবেটিস আছে কিনা।</p>
    <table>
      <thead><tr><th>Feature</th><th>বাংলা অর্থ</th></tr></thead>
      <tbody>
        <tr><td>Pregnancies</td><td>গর্ভধারণের সংখ্যা</td></tr>
        <tr><td>Glucose</td><td>রক্তে গ্লুকোজ মাত্রা</td></tr>
        <tr><td>BloodPressure</td><td>রক্তচাপ</td></tr>
        <tr><td>SkinThickness</td><td>ত্বকের পুরুত্ব</td></tr>
        <tr><td>Insulin</td><td>ইনসুলিন মাত্রা</td></tr>
        <tr><td>BMI</td><td>Body Mass Index</td></tr>
        <tr><td>DiabetesPedigreeFunction</td><td>পারিবারিক ইতিহাস স্কোর</td></tr>
        <tr><td>Age</td><td>বয়স</td></tr>
        <tr><td>Outcome</td><td>১=ডায়াবেটিস, ০=নেই (target)</td></tr>
      </tbody>
    </table>

    <h3>২. ডেটা লোড ও EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline

# ডেটা লোড (sklearn-এর pima dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

# Alternatively manually create/load:
# df = pd.read_csv(url, names=cols)
# অথবা sklearn দিয়ে:
from sklearn.datasets import fetch_openml
pima = fetch_openml(name='diabetes', version=1, as_frame=True)
df = pima.frame
df.columns = [c.lower() for c in df.columns]
df['class'] = (df['class'] == 'tested_positive').astype(int)
df = df.rename(columns={'class': 'outcome'})

print(df.shape)           # (768, 9)
print(df['outcome'].value_counts())
# 0    500  (65%)
# 1    268  (35%)
print(df.describe().T.round(2))</code></pre>

    <h3>৩. Missing Value সামলানো</h3>
    <p>এই ডেটাসেটে 0 মানে আসলে missing — Glucose=0, BMI=0 চিকিৎসাগতভাবে অসম্ভব।</p>
    <pre><code">cols_with_zeros = ['glucose', 'bloodpressure', 'skinthickness',
                   'insulin', 'bmi']

# 0 → NaN করো
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

print("Missing values:")
print(df.isnull().sum())

# Median দিয়ে imputation (outlier-এ robust)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

# এটি পরে Pipeline-এ যোগ করবো</code></pre>

    <h3>৪. Exploratory Data Analysis</h3>
    <pre><code">fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
            'insulin', 'bmi', 'diabetespedigreefunction', 'age']

for i, col in enumerate(features):
    ax = axes[i // 4][i % 4]
    df[df['outcome']==0][col].hist(ax=ax, alpha=0.6, label='No Diabetes', bins=20)
    df[df['outcome']==1][col].hist(ax=ax, alpha=0.6, label='Diabetes', bins=20)
    ax.set_title(col)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

# Correlation Matrix
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()</code></pre>

    <h3>৫. Feature Engineering</h3>
    <pre><code"># BMI category
df['bmi_category'] = pd.cut(df['bmi'],
    bins=[0, 18.5, 24.9, 29.9, np.inf],
    labels=['underweight', 'normal', 'overweight', 'obese']
)

# Age group
df['age_group'] = pd.cut(df['age'],
    bins=[0, 30, 45, 60, np.inf],
    labels=['young', 'middle', 'senior', 'elderly']
)

# Glucose-Insulin ratio (insulin resistance সূচক)
df['glucose_insulin_ratio'] = df['glucose'] / (df['insulin'] + 1)

# Encoding
from sklearn.preprocessing import OrdinalEncoder
cat_cols = ['bmi_category', 'age_group']
df[cat_cols] = OrdinalEncoder().fit_transform(df[cat_cols].fillna('normal'))</code></pre>

    <h3>৬. মডেল তৈরি ও Evaluation</h3>
    <pre><code">feature_cols = ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
                'insulin', 'bmi', 'diabetespedigreefunction', 'age',
                'glucose_insulin_ratio']

X = df[feature_cols]
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Full Pipeline
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )),
])

pipe.fit(X_train, y_train)

# Evaluation
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("=== Test Set Results ===")
print(classification_report(y_test, y_pred,
      target_names=['No Diabetes', 'Diabetes']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# 5-Fold CV
cv = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>৭. Hyperparameter Tuning</h3>
    <pre><code">from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C':            [0.001, 0.01, 0.1, 1, 10],
    'model__penalty':      ['l1', 'l2'],
    'model__class_weight': ['balanced', None],
    'model__solver':       ['saga'],
}

grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"সেরা parameter: {grid.best_params_}")
print(f"সেরা CV AUC:    {grid.best_score_:.4f}")
print(f"Test AUC:       {roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>৮. Feature Importance ও ব্যাখ্যা</h3>
    <pre><code">best_model = grid.best_estimator_.named_steps['model']
feature_names = feature_cols

imp_df = pd.DataFrame({
    'Feature':     feature_names,
    'Coefficient': best_model.coef_[0],
    'Odds Ratio':  np.exp(best_model.coef_[0]),
}).sort_values('Odds Ratio', ascending=False)

print(imp_df.round(3))
# Odds Ratio > 1 → ডায়াবেটিসের ঝুঁকি বাড়ায়
# Odds Ratio < 1 → ডায়াবেটিসের ঝুঁকি কমায়

# নমুনা prediction
sample = pd.DataFrame([{
    'pregnancies': 3, 'glucose': 140, 'bloodpressure': 80,
    'skinthickness': 25, 'insulin': 100, 'bmi': 33.5,
    'diabetespedigreefunction': 0.6, 'age': 45,
    'glucose_insulin_ratio': 140/101
}])

prob = grid.predict_proba(sample)[0, 1]
pred = grid.predict(sample)[0]
print(f"ডায়াবেটিসের সম্ভাবনা: {prob:.1%}")
print(f"ভবিষ্যদ্বাণী: {'ডায়াবেটিস আছে' if pred == 1 else 'ডায়াবেটিস নেই'}")</code></pre>

    <h3>৯. ROC Curve</h3>
    <pre><code">fpr, tpr, thresholds = roc_curve(y_test,
    grid.predict_proba(X_test)[:, 1])
auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', linewidth=2,
         label=f'Logistic Reg (AUC = {auc:.3f})')
plt.fill_between(fpr, tpr, alpha=0.1)
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve — Diabetes Prediction')
plt.legend()
plt.show()</code></pre>

    <h3>সারসংক্ষেপ — সম্পূর্ণ Workflow</h3>
    <table>
      <thead><tr><th>ধাপ</th><th>কাজ</th><th>Tool</th></tr></thead>
      <tbody>
        <tr><td>১. EDA</td><td>ডেটা বোঝো, distribution দেখো</td><td>pandas, matplotlib</td></tr>
        <tr><td>২. Preprocessing</td><td>Missing values, encoding, scaling</td><td>SimpleImputer, StandardScaler</td></tr>
        <tr><td>৩. Feature Engineering</td><td>নতুন ফিচার তৈরি</td><td>pandas</td></tr>
        <tr><td>৪. Baseline Model</td><td>সহজ মডেল দিয়ে শুরু</td><td>LogisticRegression</td></tr>
        <tr><td>৫. Evaluation</td><td>F1, ROC-AUC, Recall মাপো</td><td>classification_report</td></tr>
        <tr><td>৬. Tuning</td><td>Hyperparameter optimize করো</td><td>GridSearchCV</td></tr>
        <tr><td>৭. Interpretation</td><td>Odds Ratio দিয়ে ব্যাখ্যা করো</td><td>coef_, np.exp</td></tr>
      </tbody>
    </table>
  `,
};
