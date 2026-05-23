export const dtEn = [
  {
    slug: "dt-gini-entropy-information-gain",
    title: "Decision Trees — Gini Impurity, Entropy, and Information Gain",
    description: "How decision trees choose splits using Gini and entropy, with Python examples and visual diagrams.",
    category: "Decision Tree",
    readTime: 11,
    content: `
    <h3>1. How a Decision Tree Splits</h3>
    <p>At each node, the tree finds the feature and threshold that best separates the classes. "Best" is measured by purity metrics: Gini impurity or information entropy.</p>
    <pre><code>import numpy as np

# Gini Impurity: G = 1 - Σ pᵢ²
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

# Entropy: H = -Σ pᵢ log₂(pᵢ)
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

y_pure   = [0,0,0,0]
y_mixed  = [0,0,1,1]
y_impure = [0,1,0,1,0,1]

print(f"Pure set   — Gini: {gini(y_pure):.4f}, Entropy: {entropy(y_pure):.4f}")
print(f"Mixed set  — Gini: {gini(y_mixed):.4f}, Entropy: {entropy(y_mixed):.4f}")
print(f"Impure set — Gini: {gini(y_impure):.4f}, Entropy: {entropy(y_impure):.4f}")</code></pre>

    <h3>2. Information Gain</h3>
    <pre><code># IG = Entropy(parent) − weighted average of children entropies

def information_gain(parent, left, right):
    n = len(parent)
    w_left  = len(left)  / n
    w_right = len(right) / n
    return entropy(parent) - (w_left * entropy(left) + w_right * entropy(right))

# The split that maximises IG (or minimises Gini) is chosen
parent = [0,0,0,1,1,1]
left   = [0,0,0]    # perfect split
right  = [1,1,1]
print(f"IG = {information_gain(parent, left, right):.4f}")  # 1.0 = perfect</code></pre>

    <h3>3. Gini vs Entropy</h3>
    <table>
      <thead><tr><th>Metric</th><th>Range</th><th>Speed</th><th>In Practice</th></tr></thead>
      <tbody>
        <tr><td>Gini</td><td>[0, 0.5]</td><td>Faster (no log)</td><td>sklearn default</td></tr>
        <tr><td>Entropy</td><td>[0, log₂K]</td><td>Slower</td><td>Nearly identical results</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "dt-classification-python",
    title: "Decision Tree Classifier in Python",
    description: "sklearn DecisionTreeClassifier: fit, visualise, tune, and evaluate for binary and multiclass problems.",
    category: "Decision Tree",
    readTime: 11,
    content: `
    <h3>1. Basic DecisionTreeClassifier</h3>
    <pre><code>from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42,
)
dt.fit(X_train, y_train)
print(classification_report(y_test, dt.predict(X_test),
      target_names=data.target_names))</code></pre>

    <h3>2. Visualising the Tree</h3>
    <pre><code>import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plot_tree(dt, feature_names=data.feature_names,
          class_names=data.target_names, filled=True, rounded=True)
plt.title('Decision Tree — Iris Dataset')
plt.show()</code></pre>

    <h3>3. Key Hyperparameters</h3>
    <table>
      <thead><tr><th>Parameter</th><th>Controls</th><th>Effect of Increasing</th></tr></thead>
      <tbody>
        <tr><td>max_depth</td><td>Maximum tree depth</td><td>More complex, more overfit</td></tr>
        <tr><td>min_samples_split</td><td>Min samples to split node</td><td>Simpler tree, less overfit</td></tr>
        <tr><td>min_samples_leaf</td><td>Min samples in leaf</td><td>Smoother predictions</td></tr>
        <tr><td>max_features</td><td>Features per split</td><td>More randomness → less overfit</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "dt-overfitting-pruning",
    title: "Overfitting and Pruning in Decision Trees",
    description: "Why trees overfit, pre-pruning vs cost-complexity pruning, and cross-validation to find optimal alpha.",
    category: "Decision Tree",
    readTime: 10,
    content: `
    <h3>1. Why Decision Trees Overfit</h3>
    <p>Without constraints, a tree will grow until every leaf is pure — perfectly memorising training data but generalising poorly. The bias-variance tradeoff is controlled by pruning.</p>
    <pre><code>from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

for depth in [None, 10, 7, 5, 3]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    cv = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    print(f"max_depth={str(depth):5s}  CV={cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>2. Cost-Complexity Pruning (ccp_alpha)</h3>
    <pre><code>dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
path    = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

cv_scores = []
for alpha in ccp_alphas:
    dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    cv = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cv.mean())

best_alpha = ccp_alphas[np.argmax(cv_scores)]
dt_pruned  = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_pruned.fit(X_train, y_train)
print(f"Best ccp_alpha: {best_alpha:.6f}")
print(f"Test accuracy:  {dt_pruned.score(X_test, y_test):.4f}")</code></pre>
    `,
  },
  {
    slug: "dt-random-forest-deep",
    title: "Random Forest — Ensemble Learning via Bagging",
    description: "How Random Forest reduces variance through bootstrap aggregation, key hyperparameters, and feature importance.",
    category: "Decision Tree",
    readTime: 12,
    content: `
    <h3>1. From Decision Tree to Random Forest</h3>
    <p>Random Forest grows many diverse trees by (1) training each on a bootstrap sample of the data, and (2) considering only a random subset of features at each split. Predictions are made by majority vote (classification) or averaging (regression).</p>
    <pre><code>from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',   # √p features per split
    min_samples_leaf=3,
    oob_score=True,        # out-of-bag estimate
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
print(f"OOB Accuracy:  {rf.oob_score_:.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")</code></pre>

    <h3>2. Hyperparameter Tuning</h3>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [5, 10, 20, None],
    'max_features':    ['sqrt', 'log2', 0.5],
    'min_samples_leaf':[1, 3, 5],
}
search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=42),
    param_dist, n_iter=30, cv=5, scoring='roc_auc', random_state=42)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")</code></pre>
    `,
  },
  {
    slug: "dt-adaboost",
    title: "AdaBoost — Adaptive Boosting Explained",
    description: "How AdaBoost upweights misclassified samples, the weak learner algorithm, and sklearn implementation.",
    category: "Decision Tree",
    readTime: 11,
    content: `
    <h3>1. AdaBoost Algorithm</h3>
    <p>AdaBoost builds an ensemble sequentially. After each weak learner, misclassified samples get higher weight so the next learner focuses on them. The final prediction is a weighted vote.</p>
    <pre><code>import numpy as np
# Simplified AdaBoost mechanics:
# 1. Start: equal sample weights wᵢ = 1/n
# 2. Train weak classifier hₜ
# 3. Compute error: ε = Σ wᵢ × 𝟙[hₜ(xᵢ) ≠ yᵢ]
# 4. Classifier weight: αₜ = 0.5 × log((1-ε)/ε)
# 5. Update sample weights: wᵢ ← wᵢ × exp(-αₜ yᵢ hₜ(xᵢ))
# 6. Final: H(x) = sign(Σ αₜ hₜ(x))</code></pre>

    <h3>2. sklearn AdaBoostClassifier</h3>
    <pre><code>from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stumps
    n_estimators=200,
    learning_rate=0.5,
    random_state=42,
)
ada.fit(X_train, y_train)

cv  = cross_val_score(ada, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")</code></pre>

    <h3>3. Key Properties</h3>
    <table>
      <thead><tr><th>Property</th><th>Detail</th></tr></thead>
      <tbody>
        <tr><td>Weak learner</td><td>Decision stump (max_depth=1) is default</td></tr>
        <tr><td>Sensitive to</td><td>Outliers — they get very high weights</td></tr>
        <tr><td>Overfitting</td><td>Less prone than single tree; can overfit with high n_estimators</td></tr>
        <tr><td>Learning rate</td><td>Lower lr + more estimators = better generalisation</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "dt-gradient-boosting",
    title: "Gradient Boosting — Sequential Error Correction",
    description: "How gradient boosting fits residuals, the learning rate-trees tradeoff, and sklearn GradientBoostingClassifier.",
    category: "Decision Tree",
    readTime: 12,
    content: `
    <h3>1. Gradient Boosting Algorithm</h3>
    <p>Gradient Boosting builds trees sequentially, where each tree fits the negative gradient (pseudo-residuals) of the loss function. This generalises boosting to any differentiable loss.</p>
    <pre><code># F₀(x) = initial prediction (e.g., mean of y)
# For t = 1 to T:
#   rᵢ = −∂L(yᵢ, F_{t-1}(xᵢ)) / ∂F  (pseudo-residuals)
#   hₜ = tree fitted to rᵢ
#   F_t(x) = F_{t-1}(x) + η × hₜ(x)  (η = learning rate)

# For MSE loss: rᵢ = yᵢ − F_{t-1}(xᵢ)  (actual residuals)</code></pre>

    <h3>2. sklearn GradientBoostingClassifier</h3>
    <pre><code>from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)
gb.fit(X_train, y_train)
print(f"Test AUC: {roc_auc_score(y_test, gb.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>3. Learning Rate vs n_estimators</h3>
    <table>
      <thead><tr><th>Learning Rate</th><th>n_estimators</th><th>Effect</th></tr></thead>
      <tbody>
        <tr><td>High (0.1–0.3)</td><td>Few (50–100)</td><td>Fast training, may underfit</td></tr>
        <tr><td>Low (0.01–0.05)</td><td>Many (500+)</td><td>Better generalisation, slow training</td></tr>
        <tr><td>Use early stopping</td><td>Auto</td><td>Stop when validation loss stops improving</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "dt-feature-importance-shap",
    title: "Feature Importance and SHAP — Why Did the Model Decide This?",
    description: "MDI importance, Permutation Importance, SHAP values — three methods to interpret tree-based model decisions.",
    category: "Decision Tree",
    readTime: 11,
    content: `
    <h3>1. Why Model Interpretability Matters</h3>
    <p>Accuracy alone is not enough. We need to know which features drive predictions, whether the model learned spurious patterns, and how to debug failures.</p>

    <h3>2. Gini-based MDI Importance</h3>
    <pre><code>from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
X, y = data.data, data.target
rf   = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

mdi_df = pd.DataFrame({'Feature': data.feature_names,
                        'Importance': rf.feature_importances_}
                      ).sort_values('Importance', ascending=False)
print(mdi_df.head(10))
# Limitation: biased toward high-cardinality features</code></pre>

    <h3>3. Permutation Importance</h3>
    <pre><code>from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)

perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'Feature': data.feature_names,
                         'Mean': perm.importances_mean,
                         'Std':  perm.importances_std}
                       ).sort_values('Mean', ascending=False)
print(perm_df.head(10))</code></pre>

    <h3>4. SHAP Values</h3>
    <pre><code>import shap

explainer  = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X_test)

# Global summary plot
shap.summary_plot(shap_vals[1], X_test, feature_names=data.feature_names)

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_vals[1][0],
    base_values=explainer.expected_value[1],
    data=X_test[0],
    feature_names=data.feature_names))</code></pre>

    <h3>5. Method Comparison</h3>
    <table>
      <thead><tr><th>Method</th><th>Type</th><th>Pros</th><th>Cons</th></tr></thead>
      <tbody>
        <tr><td>MDI</td><td>Global</td><td>Fast, built-in</td><td>High-cardinality bias</td></tr>
        <tr><td>Permutation</td><td>Global</td><td>Model-agnostic, reliable</td><td>Correlated features confound</td></tr>
        <tr><td>SHAP</td><td>Global + Local</td><td>Most accurate, additive</td><td>Slower on non-tree models</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "dt-project-heart-disease",
    title: "Project: Heart Disease Prediction — Decision Tree to XGBoost",
    description: "End-to-end ML pipeline on the Cleveland Heart Disease dataset: EDA, model comparison, tuning, and SHAP explanations.",
    category: "Decision Tree",
    readTime: 14,
    content: `
    <h3>1. Dataset Overview</h3>
    <table>
      <thead><tr><th>Feature</th><th>Description</th><th>Type</th></tr></thead>
      <tbody>
        <tr><td>age</td><td>Patient age</td><td>Numeric</td></tr>
        <tr><td>sex</td><td>Gender (1=male, 0=female)</td><td>Binary</td></tr>
        <tr><td>cp</td><td>Chest pain type (0–3)</td><td>Categorical</td></tr>
        <tr><td>trestbps</td><td>Resting blood pressure</td><td>Numeric</td></tr>
        <tr><td>chol</td><td>Cholesterol level</td><td>Numeric</td></tr>
        <tr><td>thalach</td><td>Maximum heart rate</td><td>Numeric</td></tr>
        <tr><td>target</td><td>1=heart disease, 0=none</td><td>Target</td></tr>
      </tbody>
    </table>

    <h3>2. Data Loading and EDA</h3>
    <pre><code>from sklearn.datasets import fetch_openml
import pandas as pd, numpy as np

heart = fetch_openml('heart-c', version=1, as_frame=True)
df    = heart.frame.copy()
df['target'] = (df['class'] == 'positive').astype(int)
df = df.drop('class', axis=1).apply(pd.to_numeric, errors='coerce')

print(df.shape, df['target'].value_counts())</code></pre>

    <h3>3. Model Comparison</h3>
    <pre><code>from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier)
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

df_clean = df.dropna()
X = df_clean.drop('target', axis=1)
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Decision Tree':     DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'AdaBoost':          AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost':           xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    cv  = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc').mean()
    print(f"{name:20s}  CV AUC={cv:.4f}  Test AUC={auc:.4f}")</code></pre>

    <h3>4. Tune the Best Model</h3>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV

best_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)),
])
param_dist = {
    'model__n_estimators':  [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth':     [3, 4, 5],
    'model__subsample':     [0.7, 0.8, 1.0],
}
search = RandomizedSearchCV(best_pipe, param_dist, n_iter=30, cv=5,
                            scoring='roc_auc', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print(f"Best AUC: {roc_auc_score(y_test, search.predict_proba(X_test)[:,1]):.4f}")</code></pre>

    <h3>Model Performance Summary</h3>
    <table>
      <thead><tr><th>Model</th><th>Strengths</th><th>Weaknesses</th><th>When to Use</th></tr></thead>
      <tbody>
        <tr><td>Decision Tree</td><td>Interpretable, fast</td><td>Prone to overfit</td><td>Need explainability</td></tr>
        <tr><td>Random Forest</td><td>Robust, OOB error</td><td>Slow prediction</td><td>General baseline</td></tr>
        <tr><td>AdaBoost</td><td>Reduces bias</td><td>Outlier sensitive</td><td>Clean data</td></tr>
        <tr><td>XGBoost</td><td>Most accurate</td><td>Many hyperparameters</td><td>Production, competitions</td></tr>
      </tbody>
    </table>
  `,
  },
];
