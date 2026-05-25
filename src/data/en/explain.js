export const explainEn = [
  {
    slug: 'explain-1-intro',
    title: 'Model Explainability: Why Does It Matter?',
    description: 'Black box vs interpretable models, global vs local explanations, and the EU AI Act',
    category: 'Model Explainability',
    content: `
<h3>The Black Box Problem</h3>
<p>Complex models like XGBoost and neural networks achieve high accuracy but are difficult to interpret. This matters for trust, debugging, regulatory compliance, and detecting bias.</p>

<h4>Types of Explanations</h4>
<ul>
<li><strong>Global:</strong> How the model behaves overall — which features matter most?</li>
<li><strong>Local:</strong> Why did the model make this specific prediction?</li>
<li><strong>Model-specific:</strong> Built for a specific model type (TreeSHAP for trees)</li>
<li><strong>Model-agnostic:</strong> Works for any model (LIME, KernelSHAP)</li>
</ul>

<pre><code class="language-python">from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Built-in feature importance (Gini impurity-based)
importance_df = sorted(zip(feature_names, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:10]

plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in importance_df], [x[1] for x in importance_df])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance (Gini)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'explain-2-shap',
    title: 'SHAP: Explaining Models with Shapley Values',
    description: 'Game-theoretic Shapley values, TreeSHAP, force plots, and summary plots',
    category: 'Model Explainability',
    content: `
<h3>Shapley Values</h3>
<p>From cooperative game theory: each "player" (feature) receives a fair share of the total "payout" (prediction difference from baseline) based on their marginal contribution across all feature subsets.</p>

<pre><code class="language-python">import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# TreeSHAP - efficient exact computation for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global summary - beeswarm plot
shap.summary_plot(shap_values[1], X_test, feature_names=data.feature_names)

# Local explanation - force plot for one prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0],
                X_test[0], feature_names=data.feature_names, matplotlib=True)

# Dependence plot - feature interaction
shap.dependence_plot('worst radius', shap_values[1], X_test,
                     feature_names=data.feature_names)
</code></pre>
`
  },
  {
    slug: 'explain-3-lime',
    title: 'LIME: Local Interpretable Model-Agnostic Explanations',
    description: 'Perturbing inputs, fitting local linear models, and explaining tabular/text predictions',
    category: 'Model Explainability',
    content: `
<h3>LIME Algorithm</h3>
<ol>
<li>Select a prediction to explain</li>
<li>Generate perturbed versions of the input</li>
<li>Get predictions from the black-box model</li>
<li>Weight samples by proximity to original</li>
<li>Fit a simple (linear) model to these weighted samples</li>
<li>Interpret the linear model as a local explanation</li>
</ol>

<pre><code class="language-python">import lime
import lime.lime_tabular
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=data.feature_names,
    class_names=['malignant', 'benign'], mode='classification'
)

# Explain a specific prediction
idx = 5
exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)
print(f"True label: {['malignant', 'benign'][y_test[idx]]}")
print(f"Predicted: {['malignant', 'benign'][model.predict([X_test[idx]])[0]]}")
print("\\nTop features:")
for feat, weight in exp.as_list():
    direction = "↑" if weight > 0 else "↓"
    print(f"  {direction} {feat}: {weight:+.4f}")

exp.show_in_notebook()
</code></pre>
`
  },
  {
    slug: 'explain-4-permutation',
    title: 'Permutation Importance and Partial Dependence Plots',
    description: 'Feature importance via shuffling, PDPs, and Individual Conditional Expectation plots',
    category: 'Model Explainability',
    content: `
<h3>Permutation Importance</h3>
<p>Randomly shuffle one feature at a time and measure the drop in model performance. A large drop = the feature is important. Unlike Gini importance, it works with any metric on any holdout set.</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Permutation importance
perm_imp = permutation_importance(model, X_test, y_test,
                                   n_repeats=30, random_state=42, n_jobs=-1)

sorted_idx = perm_imp.importances_mean.argsort()[::-1][:8]
plt.figure(figsize=(10, 6))
plt.boxplot(perm_imp.importances[sorted_idx].T,
            vert=False, labels=data.feature_names[sorted_idx])
plt.xlabel('Decrease in R² score')
plt.title('Permutation Importance (test set)')
plt.tight_layout()
plt.show()

# Partial Dependence Plots
fig, ax = plt.subplots(figsize=(12, 5))
PartialDependenceDisplay.from_estimator(model, X_train,
    features=[0, 1, (0, 1)],
    feature_names=data.feature_names, ax=ax)
plt.tight_layout()
plt.show()
</code></pre>
`
  },
  {
    slug: 'explain-5-project',
    title: 'Explainability Project: Loan Approval Model',
    description: 'SHAP global analysis, LIME local explanations, and regulatory-ready audit reports',
    category: 'Model Explainability',
    content: `
<h3>Loan Approval Explainability Pipeline</h3>
<pre><code class="language-python">import shap
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'age': np.random.randint(22, 65, n),
    'income': np.random.randint(20000, 150000, n),
    'loan_amount': np.random.randint(50000, 500000, n),
    'credit_score': np.random.randint(300, 850, n),
    'employment_years': np.random.randint(0, 30, n),
    'existing_loans': np.random.randint(0, 5, n),
})

df['approved'] = (
    (df['credit_score'] > 600) &
    (df['income'] > 40000) &
    (df['loan_amount'] < df['income'] * 5)
).astype(int)

feature_cols = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years', 'existing_loans']
X = df[feature_cols].values
y = df['approved'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test), target_names=['Rejected', 'Approved']))

# SHAP global analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_cols, show=False)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_global.png")
plt.show()

# LIME for rejected applicants
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=feature_cols,
    class_names=['Rejected', 'Approved'], mode='classification'
)
rejected_idx = np.where(model.predict(X_test) == 0)[0][0]
exp = lime_exp.explain_instance(X_test[rejected_idx], model.predict_proba, num_features=6)
print("\\nRejection reasons:")
for feat, weight in sorted(exp.as_list(), key=lambda x: x[1]):
    print(f"  {feat}: {weight:+.3f}")
</code></pre>
`
  },
];
