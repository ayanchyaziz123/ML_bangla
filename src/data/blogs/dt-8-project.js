export const dt_8_project = {
  title: "End-to-End Project: Heart Disease ভবিষ্যদ্বাণী — Decision Tree থেকে XGBoost",
  description: "Heart disease ডেটাসেটে Decision Tree, Random Forest, AdaBoost ও XGBoost তুলনা করো — সম্পূর্ণ ML pipeline ও model selection বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 14,
  slug: "dt-project-heart-disease",
  content: `
    <h3>১. Project Overview</h3>
    <p>Cleveland Heart Disease Dataset ব্যবহার করে ভবিষ্যদ্বাণী করবো — কোনো রোগীর হার্টের সমস্যা আছে কিনা।</p>
    <table>
      <thead><tr><th>Feature</th><th>বাংলা অর্থ</th><th>ধরন</th></tr></thead>
      <tbody>
        <tr><td>age</td><td>বয়স</td><td>Numeric</td></tr>
        <tr><td>sex</td><td>লিঙ্গ (1=পুরুষ, 0=নারী)</td><td>Binary</td></tr>
        <tr><td>cp</td><td>বুকের ব্যথার ধরন (0–3)</td><td>Categorical</td></tr>
        <tr><td>trestbps</td><td>বিশ্রামে রক্তচাপ</td><td>Numeric</td></tr>
        <tr><td>chol</td><td>কোলেস্টেরল</td><td>Numeric</td></tr>
        <tr><td>fbs</td><td>উপোসী রক্তে শর্করা &gt; 120</td><td>Binary</td></tr>
        <tr><td>thalach</td><td>সর্বোচ্চ heart rate</td><td>Numeric</td></tr>
        <tr><td>oldpeak</td><td>ST depression</td><td>Numeric</td></tr>
        <tr><td>target</td><td>১=heart disease, ০=নেই</td><td>Target</td></tr>
      </tbody>
    </table>

    <h3>২. ডেটা লোড ও EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

# Cleveland Heart Disease (UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','target']

# Alternative: sklearn dataset
from sklearn.datasets import fetch_openml
heart = fetch_openml('heart-c', version=1, as_frame=True)
df = heart.frame.copy()
df['target'] = (df['class'] == 'positive').astype(int)
df = df.drop('class', axis=1)

# Numeric করো
df = df.apply(pd.to_numeric, errors='coerce')
print(df.shape)
print(df['target'].value_counts())
print(df.isnull().sum())</code></pre>

    <h3>৩. ডেটা প্রস্তুতি</h3>
    <pre><code">from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Missing values পূরণ
df_clean = df.dropna()   # বা imputation ব্যবহার করো
print(f"পরিষ্কার ডেটা: {df_clean.shape}")

X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
], remainder='passthrough')</code></pre>

    <h3>৪. চারটি মডেল একসাথে তুলনা</h3>
    <pre><code">from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)
import xgboost as xgb

models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, learning_rate=0.5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])

    # 5-fold CV
    cv_auc = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring='roc_auc').mean()

    # Train + Test
    pipe.fit(X_train, y_train)
    test_auc  = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    test_acc  = pipe.score(X_test, y_test)

    results[name] = {'CV AUC': cv_auc, 'Test AUC': test_auc, 'Test Acc': test_acc}
    print(f"{name:20s} | CV AUC={cv_auc:.4f} | Test AUC={test_auc:.4f} | Acc={test_acc:.4f}")

results_df = pd.DataFrame(results).T.sort_values('Test AUC', ascending=False)
print("\\n=== চূড়ান্ত ফলাফল ===")
print(results_df.round(4))</code></pre>

    <h3>৫. সেরা মডেল Tune করো</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV

# XGBoost tune করো
param_dist = {
    'model__n_estimators':    [50, 100, 200],
    'model__learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'model__max_depth':       [3, 4, 5, 6],
    'model__subsample':       [0.7, 0.8, 1.0],
    'model__colsample_bytree':[0.7, 0.8, 1.0],
    'model__reg_alpha':       [0, 0.1, 0.5],
}

best_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', xgb.XGBClassifier(
        eval_metric='logloss', random_state=42, n_jobs=-1)),
])

search = RandomizedSearchCV(
    best_pipe, param_dist,
    n_iter=30, cv=5,
    scoring='roc_auc',
    random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"সেরা params: {search.best_params_}")
print(f"Test AUC:    {roc_auc_score(y_test, search.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>৬. চূড়ান্ত Evaluation</h3>
    <pre><code">best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred,
      target_names=['No Disease', 'Heart Disease']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Heart Disease']).plot()
plt.title('Confusion Matrix — XGBoost')
plt.show()

# ROC Curve সব model-এর
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('m', model)])
    pipe.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve তুলনা')
plt.legend()
plt.show()</code></pre>

    <h3>৭. SHAP দিয়ে ব্যাখ্যা</h3>
    <pre><code"># XGBoost + SHAP
import shap

# Preprocessed test data
X_test_prep = best_model.named_steps['prep'].transform(X_test)
xgb_model   = best_model.named_steps['model']

explainer   = shap.TreeExplainer(xgb_model)
shap_vals   = explainer.shap_values(X_test_prep)

# Feature names পাওয়া
num_names = numeric_features
cat_names = list(best_model.named_steps['prep']
                 .named_transformers_['cat']
                 .get_feature_names_out(categorical_features))
all_feature_names = num_names + cat_names + ['ca', 'thal']

shap.summary_plot(shap_vals, X_test_prep,
                  feature_names=all_feature_names)</code></pre>

    <h3>সারসংক্ষেপ — মডেল পারফরম্যান্স তুলনা</h3>
    <table>
      <thead><tr><th>মডেল</th><th>সুবিধা</th><th>অসুবিধা</th><th>কখন বেছে নেবে</th></tr></thead>
      <tbody>
        <tr><td>Decision Tree</td><td>ব্যাখ্যাযোগ্য, দ্রুত</td><td>Overfitting-প্রবণ</td><td>Interpretability লাগলে</td></tr>
        <tr><td>Random Forest</td><td>robust, OOB error</td><td>ধীর prediction</td><td>সাধারণ baseline</td></tr>
        <tr><td>AdaBoost</td><td>bias কমায়</td><td>outlier sensitive</td><td>clean data-তে</td></tr>
        <tr><td>Gradient Boosting</td><td>flexible, accurate</td><td>ধীর training</td><td>medium data</td></tr>
        <tr><td>XGBoost</td><td>সবচেয়ে accurate</td><td>বেশি hyperparameter</td><td>production ও competition</td></tr>
      </tbody>
    </table>
  `,
};
