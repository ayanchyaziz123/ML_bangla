export const logitEn = [
  {
    slug: "logit-math-deep-dive",
    title: "Logistic Regression Mathematics — Sigmoid, Log-Odds, MLE",
    description: "The complete math of logistic regression: sigmoid function, log-odds, maximum likelihood estimation, and the cost function.",
    category: "Logistic Regression",
    readTime: 12,
    content: `
    <h3>1. The Sigmoid Function</h3>
    <p>Logistic regression applies the sigmoid to map a linear combination of features to a probability in (0,1).</p>
    <pre><code>import numpy as np, matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
z = np.linspace(-8, 8, 200)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color='red', linestyle='--', label='Decision boundary')
plt.xlabel('z'); plt.ylabel('P(y=1)'); plt.legend(); plt.show()</code></pre>

    <h3>2. Log-Odds (Logit)</h3>
    <pre><code># The logit (log-odds) is the inverse of sigmoid:
# log(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ

# This is why it's called LOGISTIC regression:
# the linear predictor is the log-odds of the outcome

p = np.linspace(0.01, 0.99, 100)
log_odds = np.log(p / (1 - p))
plt.plot(p, log_odds)
plt.xlabel('Probability'); plt.ylabel('Log-Odds (Logit)')
plt.axhline(0, color='gray'); plt.show()</code></pre>

    <h3>3. Maximum Likelihood Estimation</h3>
    <pre><code># Logistic regression maximises the log-likelihood:
# ℓ(β) = Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
# This is equivalent to minimising Binary Cross-Entropy loss.

def log_likelihood(X, y, beta):
    z = X @ beta
    p = sigmoid(z)
    return np.mean(y*np.log(p+1e-15) + (1-y)*np.log(1-p+1e-15))

def gradient(X, y, beta):
    return X.T @ (sigmoid(X@beta) - y) / len(y)</code></pre>

    <h3>4. Regularisation</h3>
    <table>
      <thead><tr><th>Parameter</th><th>sklearn</th><th>Effect</th></tr></thead>
      <tbody>
        <tr><td>L2 (Ridge)</td><td>penalty='l2', C=1/λ</td><td>Shrinks all coefficients</td></tr>
        <tr><td>L1 (Lasso)</td><td>penalty='l1', solver='liblinear'</td><td>Sparse coefficients (feature selection)</td></tr>
        <tr><td>Elastic Net</td><td>penalty='elasticnet', solver='saga'</td><td>Combines L1 + L2</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "logit-python-guide",
    title: "Logistic Regression in Python — Complete sklearn Guide",
    description: "Full workflow: preprocessing, fitting, threshold tuning, and interpreting coefficients with LogisticRegression.",
    category: "Logistic Regression",
    readTime: 10,
    content: `
    <h3>1. Basic Logistic Regression</h3>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)

y_pred  = pipe.predict(X_test)
y_prob  = pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")</code></pre>

    <h3>2. Threshold Tuning</h3>
    <pre><code>from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores  = 2 * precision * recall / (precision + recall + 1e-10)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Default threshold:  F1={f1_scores[np.searchsorted(thresholds,0.5)]:.4f}")
print(f"Optimal threshold:  {best_thresh:.4f}, F1={f1_scores.max():.4f}")

y_pred_tuned = (y_prob >= best_thresh).astype(int)
print(classification_report(y_test, y_pred_tuned))</code></pre>

    <h3>3. Interpret Coefficients</h3>
    <pre><code>import pandas as pd

clf    = pipe.named_steps['clf']
coef_df = pd.DataFrame({
    'Feature':    data.feature_names,
    'Coefficient': clf.coef_[0],
    'Odds Ratio':  np.exp(clf.coef_[0]),
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.head(10))
# Odds Ratio > 1: feature increases P(positive)
# Odds Ratio < 1: feature decreases P(positive)</code></pre>
    `,
  },
  {
    slug: "logit-multiclass",
    title: "Multiclass Logistic Regression — One-vs-Rest and Softmax",
    description: "OvR, OvO, and multinomial (softmax) strategies for classifying more than two classes.",
    category: "Logistic Regression",
    readTime: 10,
    content: `
    <h3>1. Strategies for Multiclass</h3>
    <table>
      <thead><tr><th>Strategy</th><th>How It Works</th><th>sklearn</th></tr></thead>
      <tbody>
        <tr><td>One-vs-Rest (OvR)</td><td>K binary classifiers, one per class</td><td>multi_class='ovr'</td></tr>
        <tr><td>One-vs-One (OvO)</td><td>K(K-1)/2 classifiers</td><td>multi_class='ovo' via OneVsOneClassifier</td></tr>
        <tr><td>Softmax (Multinomial)</td><td>Single model, softmax output</td><td>multi_class='multinomial'</td></tr>
      </tbody>
    </table>

    <h3>2. sklearn Multiclass Example</h3>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Softmax (multinomial)
lr_softmax = LogisticRegression(multi_class='multinomial',
                                 solver='lbfgs', max_iter=1000)
lr_softmax.fit(X_train_s, y_train)
print(classification_report(y_test, lr_softmax.predict(X_test_s),
      target_names=data.target_names))

# Probabilities sum to 1 across classes
probs = lr_softmax.predict_proba(X_test_s[:3])
print(probs.round(3))</code></pre>
    `,
  },
  {
    slug: "logit-eval-metrics",
    title: "Classification Evaluation — ROC-AUC, Precision, Recall, F1",
    description: "Confusion matrix, ROC-AUC, precision-recall curve, and which metric to use in each business scenario.",
    category: "Logistic Regression",
    readTime: 11,
    content: `
    <h3>1. Confusion Matrix Components</h3>
    <pre><code>from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Negative','Positive']).plot()
plt.show()

# TP = correctly predicted positive
# TN = correctly predicted negative
# FP = false alarm (Type I error)
# FN = missed positive (Type II error)</code></pre>

    <h3>2. Key Metrics</h3>
    <pre><code>from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score)

precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)

print(f"Precision: {precision:.4f}  (of predicted positives, how many are correct?)")
print(f"Recall:    {recall:.4f}  (of actual positives, how many did we catch?)")
print(f"F1:        {f1:.4f}  (harmonic mean of precision & recall)")
print(f"ROC-AUC:   {auc:.4f}  (probability model ranks positive above negative)")</code></pre>

    <h3>3. ROC Curve</h3>
    <pre><code>from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'k--', label='Random')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()</code></pre>

    <h3>4. Metric Choice Guide</h3>
    <table>
      <thead><tr><th>Scenario</th><th>Priority Metric</th><th>Reason</th></tr></thead>
      <tbody>
        <tr><td>Medical diagnosis</td><td>Recall</td><td>Missing a disease is very costly</td></tr>
        <tr><td>Spam detection</td><td>Precision</td><td>False alarms delete legitimate mail</td></tr>
        <tr><td>Imbalanced classes</td><td>F1 or PR-AUC</td><td>Accuracy misleads</td></tr>
        <tr><td>Threshold-free evaluation</td><td>ROC-AUC</td><td>Evaluates ranking, not a specific threshold</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "logit-imbalanced-dataset",
    title: "Imbalanced Classification — SMOTE, Class Weights, Threshold Tuning",
    description: "Handling imbalanced datasets with resampling, cost-sensitive learning, and evaluation with PR-AUC.",
    category: "Logistic Regression",
    readTime: 12,
    content: `
    <h3>1. The Imbalance Problem</h3>
    <p>When one class is rare (e.g., 95% negative, 5% positive), a model that always predicts "negative" gets 95% accuracy but is completely useless. Use F1, PR-AUC, or ROC-AUC instead of accuracy.</p>

    <h3>2. Class Weights</h3>
    <pre><code>from sklearn.linear_model import LogisticRegression

# class_weight='balanced' automatically weighs minority class higher
lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_balanced.fit(X_train, y_train)

# Or specify manually:
lr_custom = LogisticRegression(class_weight={0:1, 1:10})  # 10× penalty for missing class 1
lr_custom.fit(X_train, y_train)</code></pre>

    <h3>3. SMOTE — Oversampling</h3>
    <pre><code>from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE generates synthetic minority samples by interpolating between real samples
pipe = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('scaler', StandardScaler()),
    ('clf',   LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)

print(f"F1: {f1_score(y_test, pipe.predict(X_test)):.4f}")
print(f"AUC: {roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>4. Strategies Summary</h3>
    <table>
      <thead><tr><th>Strategy</th><th>Method</th><th>Best For</th></tr></thead>
      <tbody>
        <tr><td>Cost-sensitive</td><td>class_weight='balanced'</td><td>Simple, no data modification</td></tr>
        <tr><td>Oversampling</td><td>SMOTE</td><td>Small minority class</td></tr>
        <tr><td>Undersampling</td><td>RandomUnderSampler</td><td>Very large majority class</td></tr>
        <tr><td>Threshold tuning</td><td>Lower decision threshold</td><td>Post-hoc adjustment</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "logit-project-diabetes",
    title: "Project: Diabetes Prediction — End-to-End Logistic Regression",
    description: "Full ML pipeline on the Pima Indians Diabetes dataset: EDA, preprocessing, model selection, and evaluation.",
    category: "Logistic Regression",
    readTime: 13,
    content: `
    <h3>1. Dataset Overview</h3>
    <pre><code>import pandas as pd
from sklearn.datasets import load_diabetes
# Using Pima Indians dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigree','Age','Outcome']
df = pd.read_csv(url, names=cols)
print(df.describe())
print(df['Outcome'].value_counts())  # 500 non-diabetic, 268 diabetic</code></pre>

    <h3>2. EDA and Preprocessing</h3>
    <pre><code>import numpy as np

# Some zero values are physiologically impossible — treat as missing
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print(df.isnull().sum())

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)</code></pre>

    <h3>3. Pipeline and Model Comparison</h3>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, class_weight='balanced'),
}

for name, model in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     model),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    auc    = roc_auc_score(y_test, y_prob)
    print(f"{name:22s}  AUC={auc:.4f}")
    print(classification_report(y_test, y_pred))</code></pre>
    `,
  },
];
