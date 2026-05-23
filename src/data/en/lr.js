export const lrEn = [
  {
    slug: "lr-ki-keno-math",
    title: "Linear Regression — What, Why, and the Math",
    description: "Why linear regression works, the cost function, OLS, and building it from scratch in Python.",
    category: "Linear Regression",
    readTime: 12,
    content: `
    <h3>1. What is Linear Regression?</h3>
    <p>Linear regression models the relationship between a dependent variable y and one or more independent variables X by fitting a linear equation to the data.</p>
    <pre><code>y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
# β₀ = intercept, β₁...βₙ = coefficients, ε = error term</code></pre>

    <h3>2. Cost Function — Mean Squared Error</h3>
    <pre><code>import numpy as np

# MSE = (1/n) Σ (yᵢ − ŷᵢ)²
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# OLS closed-form solution: β = (XᵀX)⁻¹ Xᵀy
X = np.array([[1,1],[1,2],[1,3],[1,4]])  # add bias column
y = np.array([2, 4, 5, 4])
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Intercept: {beta[0]:.4f}, Slope: {beta[1]:.4f}")</code></pre>

    <h3>3. OLS Assumptions</h3>
    <table>
      <thead><tr><th>Assumption</th><th>Description</th><th>Violation Effect</th></tr></thead>
      <tbody>
        <tr><td>Linearity</td><td>Relationship between X and y is linear</td><td>Biased estimates</td></tr>
        <tr><td>Independence</td><td>Observations are independent</td><td>Inflated R²</td></tr>
        <tr><td>Homoscedasticity</td><td>Constant variance of residuals</td><td>Inefficient estimates</td></tr>
        <tr><td>Normality</td><td>Residuals are normally distributed</td><td>Invalid inference</td></tr>
        <tr><td>No multicollinearity</td><td>Predictors are not highly correlated</td><td>Unstable coefficients</td></tr>
      </tbody>
    </table>

    <h3>4. Gradient Descent Alternative</h3>
    <pre><code># OLS is closed-form but doesn't scale to large datasets.
# Gradient descent iteratively minimizes MSE:
# β ← β − α × (∂MSE/∂β)

def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    beta = np.zeros(n)
    for _ in range(epochs):
        y_pred = X @ beta
        grad   = -(2/m) * X.T @ (y - y_pred)
        beta  -= lr * grad
    return beta</code></pre>
    `,
  },
  {
    slug: "lr-python",
    title: "Linear Regression in Python — sklearn from Start to Finish",
    description: "Complete sklearn LinearRegression workflow: fit, predict, evaluate, and interpret coefficients.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. sklearn LinearRegression</h3>
    <pre><code>from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np, pandas as pd

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")</code></pre>

    <h3>2. Interpret Coefficients</h3>
    <pre><code>coef_df = pd.DataFrame({
    'Feature':     data.feature_names,
    'Coefficient': lr.coef_,
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df)
# Positive coef → higher feature value → higher prediction
# Negative coef → higher feature value → lower prediction</code></pre>

    <h3>3. Residual Diagnostics</h3>
    <pre><code>import matplotlib.pyplot as plt

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].scatter(y_pred, residuals, alpha=0.3)
axes[0].axhline(0, color='red')
axes[0].set(xlabel='Predicted', ylabel='Residual', title='Residual Plot')

axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white')
axes[1].set(xlabel='Residual', title='Residual Distribution')
plt.tight_layout(); plt.show()</code></pre>
    `,
  },
  {
    slug: "lr-metrics-r2-mse",
    title: "Regression Metrics — R², MSE, MAE, RMSE",
    description: "Understand each regression metric, when to use which, and how to interpret them correctly.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Core Regression Metrics</h3>
    <pre><code>import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)

y_true = np.array([100, 200, 150, 300, 250])
y_pred = np.array([110, 190, 160, 280, 260])

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}  ← same unit as y")
print(f"MAE:  {mae:.2f}   ← less sensitive to outliers")
print(f"R²:   {r2:.4f}   ← 1.0 = perfect, 0 = baseline mean")
print(f"MAPE: {mape:.2%}  ← percentage error")</code></pre>

    <h3>2. Metric Comparison</h3>
    <table>
      <thead><tr><th>Metric</th><th>Formula</th><th>Range</th><th>Sensitive to Outliers</th><th>Best For</th></tr></thead>
      <tbody>
        <tr><td>MSE</td><td>mean((y−ŷ)²)</td><td>[0, ∞)</td><td>Yes</td><td>Penalising large errors</td></tr>
        <tr><td>RMSE</td><td>√MSE</td><td>[0, ∞)</td><td>Yes</td><td>Same units as y</td></tr>
        <tr><td>MAE</td><td>mean(|y−ŷ|)</td><td>[0, ∞)</td><td>Less</td><td>Robust evaluation</td></tr>
        <tr><td>R²</td><td>1 − SS_res/SS_tot</td><td>(-∞, 1]</td><td>No</td><td>Explained variance</td></tr>
        <tr><td>MAPE</td><td>mean(|y−ŷ|/y)</td><td>[0, ∞)</td><td>No</td><td>Business reporting</td></tr>
      </tbody>
    </table>

    <h3>3. Adjusted R²</h3>
    <pre><code>n = len(y_true)
p = 3  # number of predictors

r2_adj = 1 - (1-r2) * (n-1) / (n-p-1)
print(f"R²:          {r2:.4f}")
print(f"Adjusted R²: {r2_adj:.4f}")
# Adjusted R² penalises adding unhelpful features</code></pre>
    `,
  },
  {
    slug: "lr-multicollinearity-ridge-lasso",
    title: "Multicollinearity, Ridge, and Lasso Regression",
    description: "Detect multicollinearity with VIF, understand L1 and L2 regularization, and tune Ridge and Lasso with cross-validation.",
    category: "Linear Regression",
    readTime: 12,
    content: `
    <h3>1. What is Multicollinearity?</h3>
    <p>Multicollinearity occurs when predictor variables are highly correlated, making coefficient estimates unstable and difficult to interpret.</p>
    <pre><code>import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

print("Correlation matrix:")
print(X.corr().round(2))</code></pre>

    <h3>2. Ridge Regression (L2)</h3>
    <pre><code>from sklearn.linear_model import RidgeCV
import numpy as np

# Ridge adds λΣβᵢ² to the cost function — shrinks coefficients toward 0
alphas = np.logspace(-3, 3, 100)
ridge  = RidgeCV(alphas=alphas, cv=5)

scaler = StandardScaler()
X_s    = scaler.fit_transform(X)
ridge.fit(X_s, y)
print(f"Best alpha: {ridge.alpha_:.4f}")
print(f"R²:         {ridge.score(X_s, y):.4f}")</code></pre>

    <h3>3. Lasso Regression (L1)</h3>
    <pre><code>from sklearn.linear_model import LassoCV

# Lasso adds λΣ|βᵢ| — drives some coefficients exactly to 0 (feature selection)
lasso = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000)
lasso.fit(X_s, y)

print(f"Best alpha: {lasso.alpha_:.6f}")
print(f"Non-zero coefficients: {(lasso.coef_ != 0).sum()}")</code></pre>

    <h3>4. Ridge vs Lasso vs OLS</h3>
    <table>
      <thead><tr><th>Method</th><th>Penalty</th><th>Effect</th><th>Feature Selection</th></tr></thead>
      <tbody>
        <tr><td>OLS</td><td>None</td><td>Unbiased, high variance</td><td>No</td></tr>
        <tr><td>Ridge (L2)</td><td>λΣβ²</td><td>Shrinks all coefficients</td><td>No</td></tr>
        <tr><td>Lasso (L1)</td><td>λΣ|β|</td><td>Drives some to exactly 0</td><td>Yes</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "lr-interview-questions",
    title: "Linear Regression — Top Interview Questions and Answers",
    description: "Concise answers to the most common Linear Regression interview questions for data science roles.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Core Concepts</h3>
    <table>
      <thead><tr><th>Question</th><th>Answer</th></tr></thead>
      <tbody>
        <tr><td>What is linear regression?</td><td>A supervised learning algorithm that models the linear relationship between predictors and a continuous response variable.</td></tr>
        <tr><td>What are OLS assumptions?</td><td>Linearity, independence, homoscedasticity, normality of residuals, no multicollinearity.</td></tr>
        <tr><td>How is R² interpreted?</td><td>The proportion of variance in y explained by the model. R²=0.8 means 80% of variance is explained.</td></tr>
        <tr><td>When does R² fail?</td><td>Adding any variable (even irrelevant) never decreases R². Use Adjusted R² instead.</td></tr>
        <tr><td>What is the difference between correlation and regression?</td><td>Correlation measures linear association (symmetric). Regression predicts y from X (directional).</td></tr>
      </tbody>
    </table>

    <h3>2. Regularization Questions</h3>
    <table>
      <thead><tr><th>Question</th><th>Answer</th></tr></thead>
      <tbody>
        <tr><td>What does Ridge do?</td><td>Adds L2 penalty (λΣβ²) to cost — shrinks all coefficients toward zero.</td></tr>
        <tr><td>What does Lasso do?</td><td>Adds L1 penalty (λΣ|β|) — can drive coefficients exactly to zero (automatic feature selection).</td></tr>
        <tr><td>When to use Ridge vs Lasso?</td><td>Ridge: all features matter. Lasso: suspect many features are irrelevant.</td></tr>
        <tr><td>What is Elastic Net?</td><td>Combines L1+L2 penalties. Best when correlated features exist and some should be zeroed out.</td></tr>
      </tbody>
    </table>

    <h3>3. Practical Questions</h3>
    <pre><code>from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Q: How do you detect overfitting?
# A: Large gap between train and test R²

lr = LinearRegression()
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Q: What is heteroscedasticity?
# A: Non-constant variance of residuals — visible in residual vs fitted plot
# Fix: log-transform y, use WLS, or use robust regression</code></pre>
    `,
  },
  {
    slug: "lr-residual-analysis",
    title: "Residual Analysis — Checking Linear Regression Assumptions",
    description: "How to diagnose OLS assumption violations with residual plots, Q-Q plots, and statistical tests.",
    category: "Linear Regression",
    readTime: 11,
    content: `
    <h3>1. What are Residuals?</h3>
    <p>A residual is the difference between the observed value and the model's prediction: eᵢ = yᵢ − ŷᵢ. Good residuals should be randomly scattered around zero.</p>

    <h3>2. Four Key Diagnostic Plots</h3>
    <pre><code>import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# 1. Residuals vs Fitted — check linearity & homoscedasticity
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color='red', lw=1)
plt.xlabel('Fitted values'); plt.ylabel('Residuals')

# 2. Q-Q Plot — check normality of residuals
stats.probplot(residuals, plot=plt)

# 3. Scale-Location — check homoscedasticity
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.4)

# 4. Residuals vs Leverage — detect influential observations
from statsmodels.graphics.regressionplots import influence_plot</code></pre>

    <h3>3. Statistical Tests</h3>
    <pre><code>from scipy import stats
import statsmodels.stats.api as sms

# Normality: Shapiro-Wilk (small n) or D'Agostino-K² (large n)
stat, p = stats.shapiro(residuals[:50])
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}")
# p > 0.05 → cannot reject normality

# Homoscedasticity: Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_intercept)
print(f"Breusch-Pagan p-value: {bp_p:.4f}")
# p < 0.05 → heteroscedasticity detected

# Autocorrelation: Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.4f}")
# ~2.0 = no autocorrelation, <1 or >3 = concern</code></pre>
    `,
  },
  {
    slug: "lr-polynomial-regression",
    title: "Polynomial Regression — Fitting Non-Linear Relationships",
    description: "When linear isn't enough: PolynomialFeatures, degree selection, bias-variance tradeoff.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Why Polynomial Regression?</h3>
    <p>When the relationship between X and y is curved (not linear), adding polynomial features (X², X³, ...) lets linear regression fit non-linear patterns.</p>
    <pre><code>y = β₀ + β₁x + β₂x² + β₃x³ + ε
# Still "linear" in parameters — uses same OLS/gradient descent</code></pre>

    <h3>2. sklearn PolynomialFeatures</h3>
    <pre><code>from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np, matplotlib.pyplot as plt

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X.ravel()**3 - 2*X.ravel() + np.random.normal(0, 0.5, 100)

for degree in [1, 2, 3, 5]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('lr',   LinearRegression()),
    ])
    pipe.fit(X, y)
    print(f"Degree {degree}: R²={r2_score(y, pipe.predict(X)):.4f}")</code></pre>

    <h3>3. Selecting Optimal Degree</h3>
    <pre><code>from sklearn.model_selection import cross_val_score

degrees     = range(1, 10)
cv_scores   = []
train_scores = []

for d in degrees:
    pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
    cv_scores.append(cross_val_score(pipe, X, y, cv=5, scoring='r2').mean())
    pipe.fit(X, y)
    train_scores.append(r2_score(y, pipe.predict(X)))

best_degree = degrees[np.argmax(cv_scores)]
print(f"Best degree by CV: {best_degree}")</code></pre>
    `,
  },
  {
    slug: "lr-feature-selection",
    title: "Feature Selection for Linear Regression",
    description: "Filter, wrapper (RFE), and embedded methods — how to select the most informative features.",
    category: "Linear Regression",
    readTime: 11,
    content: `
    <h3>1. Why Feature Selection?</h3>
    <p>Irrelevant or redundant features increase noise, hurt interpretability, and can cause overfitting. Feature selection finds the minimal subset that maximises model performance.</p>

    <h3>2. Filter Methods</h3>
    <pre><code>from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# F-test (linear correlation)
f_stat, p_val = f_regression(X, y)
filter_df = pd.DataFrame({'Feature': data.feature_names,
                           'F-stat': f_stat, 'p-value': p_val})
print(filter_df.sort_values('F-stat', ascending=False))

# Mutual information (non-linear dependencies)
mi = mutual_info_regression(X, y, random_state=42)
print(pd.Series(mi, index=data.feature_names).sort_values(ascending=False))</code></pre>

    <h3>3. Recursive Feature Elimination (RFE)</h3>
    <pre><code>from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LinearRegression

rfe = RFECV(LinearRegression(), min_features_to_select=2, cv=5, scoring='r2')
rfe.fit(X, y)

selected = pd.DataFrame({'Feature': data.feature_names,
                          'Selected': rfe.support_,
                          'Rank':     rfe.ranking_})
print(selected[selected.Selected])</code></pre>

    <h3>4. Embedded — Lasso</h3>
    <pre><code>from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_s, y)

selected_lasso = [name for name, coef in
                  zip(data.feature_names, lasso.coef_) if coef != 0]
print(f"Lasso selected {len(selected_lasso)} features: {selected_lasso}")</code></pre>
    `,
  },
  {
    slug: "lr-project-house-price",
    title: "Project: House Price Prediction — End-to-End Regression",
    description: "Full ML pipeline on California Housing: EDA, preprocessing, model comparison, hyperparameter tuning, and SHAP.",
    category: "Linear Regression",
    readTime: 14,
    content: `
    <h3>1. Project Setup</h3>
    <pre><code>import pandas as pd, numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing(as_frame=True)
df   = data.frame.copy()
print(df.describe())
print(df.isnull().sum())</code></pre>

    <h3>2. EDA</h3>
    <pre><code>import matplotlib.pyplot as plt, seaborn as sns

fig, axes = plt.subplots(2, 4, figsize=(16,8))
for i, col in enumerate(data.feature_names):
    axes[i//4, i%4].scatter(df[col], df['MedHouseVal'], alpha=0.1, s=5)
    axes[i//4, i%4].set(xlabel=col, ylabel='Price')
plt.tight_layout(); plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')</code></pre>

    <h3>3. Model Comparison</h3>
    <pre><code>from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

X = data.data; y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'Lasso':             Lasso(alpha=0.01),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:22s}  R²={r2:.4f}  RMSE={rmse:.4f}")</code></pre>

    <h3>4. SHAP for Model Explanation</h3>
    <pre><code>import shap
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test[:100])

shap.summary_plot(shap_vals, X_test[:100], feature_names=data.feature_names)</code></pre>
    `,
  },
  {
    slug: "lr-vif-deep-dive",
    title: "VIF — Detecting and Fixing Multicollinearity",
    description: "What VIF measures, how to compute it, and practical strategies to resolve multicollinearity.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Variance Inflation Factor (VIF)</h3>
    <p>VIF measures how much the variance of a coefficient is inflated due to linear relationships with other predictors. VIF = 1/(1 − R²ⱼ) where R²ⱼ is from regressing predictor j on all other predictors.</p>
    <pre><code>from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np

# Rule of thumb: VIF > 5 = concern, VIF > 10 = severe multicollinearity

def compute_vif(X_df):
    X_arr = X_df.values
    return pd.DataFrame({
        'Feature': X_df.columns,
        'VIF':     [variance_inflation_factor(X_arr, i) for i in range(X_arr.shape[1])]
    }).sort_values('VIF', ascending=False)

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
print(compute_vif(data.data))</code></pre>

    <h3>2. Fixing Multicollinearity</h3>
    <pre><code># Strategy 1: Remove high-VIF features
def remove_high_vif(X_df, threshold=5):
    while True:
        vif_df = compute_vif(X_df)
        max_vif = vif_df['VIF'].max()
        if max_vif < threshold:
            break
        drop_col = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
        X_df = X_df.drop(columns=[drop_col])
        print(f"Dropped '{drop_col}' (VIF={max_vif:.2f})")
    return X_df

# Strategy 2: Ridge regression (handles collinearity naturally)
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.logspace(-3,3,100), cv=5)
ridge.fit(X_scaled, y)  # Ridge dampens inflated coefficients</code></pre>
    `,
  },
  {
    slug: "lr-feature-encoding",
    title: "Feature Encoding — Handling Categorical Variables",
    description: "One-Hot Encoding, Label Encoding, Target Encoding, and Ordinal Encoding — when and how to use each.",
    category: "Linear Regression",
    readTime: 11,
    content: `
    <h3>1. Why Encode Categorical Features?</h3>
    <p>Linear regression requires numeric inputs. Categorical variables like "City" or "Color" must be encoded numerically before fitting.</p>

    <h3>2. One-Hot Encoding</h3>
    <pre><code>import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'City': ['Dhaka','Chittagong','Dhaka','Sylhet'],
                   'Price': [500, 400, 520, 350]})

ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded = ohe.fit_transform(df[['City']])
print(pd.DataFrame(encoded, columns=ohe.get_feature_names_out()))</code></pre>

    <h3>3. Label Encoding (Ordinal)</h3>
    <pre><code>from sklearn.preprocessing import OrdinalEncoder

# Only use for truly ordinal variables (Low < Medium < High)
df['Education'] = ['High School','Bachelor','Master','PhD']
oe = OrdinalEncoder(categories=[['High School','Bachelor','Master','PhD']])
df['Education_enc'] = oe.fit_transform(df[['Education']])</code></pre>

    <h3>4. Target Encoding</h3>
    <pre><code># Replace category with mean of target — powerful but can overfit
target_means = df.groupby('City')['Price'].mean()
df['City_target_enc'] = df['City'].map(target_means)

# Use cross-validation target encoding to prevent leakage
from category_encoders import TargetEncoder
te = TargetEncoder()
df['City_te'] = te.fit_transform(df['City'], df['Price'])</code></pre>

    <h3>5. Encoding Method Comparison</h3>
    <table>
      <thead><tr><th>Method</th><th>Best For</th><th>Pros</th><th>Cons</th></tr></thead>
      <tbody>
        <tr><td>One-Hot</td><td>Nominal, low cardinality</td><td>No ordinal assumption</td><td>High cardinality → many columns</td></tr>
        <tr><td>Label/Ordinal</td><td>Ordinal categories</td><td>Single column</td><td>Wrong for nominal</td></tr>
        <tr><td>Target</td><td>High cardinality nominal</td><td>Compact, informative</td><td>Risk of leakage</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "lr-feature-scaling",
    title: "Feature Scaling — StandardScaler, MinMaxScaler, RobustScaler",
    description: "When scaling is necessary, which scaler to choose, and common pitfalls like train-test leakage.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Why Scale Features?</h3>
    <p>Without scaling, features with larger magnitudes dominate gradient descent. Ridge/Lasso regularization is also sensitive to scale. Always fit scalers on training data only.</p>

    <h3>2. Three Common Scalers</h3>
    <pre><code>import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = np.array([[100, 0.1], [200, 0.2], [150, 0.15], [500, 0.5]])

# StandardScaler: z = (x - μ) / σ  → mean=0, std=1
std = StandardScaler().fit_transform(X)

# MinMaxScaler: x' = (x - min) / (max - min) → [0, 1]
mm = MinMaxScaler().fit_transform(X)

# RobustScaler: uses median and IQR → robust to outliers
rob = RobustScaler().fit_transform(X)</code></pre>

    <h3>3. Correct Pipeline (No Leakage)</h3>
    <pre><code>from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CORRECT: scaler fits only on train data
pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
pipe.fit(X_train, y_train)
print(f"R²: {pipe.score(X_test, y_test):.4f}")</code></pre>

    <h3>4. Which Scaler to Choose?</h3>
    <table>
      <thead><tr><th>Scaler</th><th>Use When</th><th>Outlier Sensitive</th></tr></thead>
      <tbody>
        <tr><td>StandardScaler</td><td>Gaussian distribution assumed</td><td>Yes</td></tr>
        <tr><td>MinMaxScaler</td><td>Neural networks, image data</td><td>Yes</td></tr>
        <tr><td>RobustScaler</td><td>Many outliers present</td><td>No</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "lr-gradient-descent",
    title: "Gradient Descent — Batch, Stochastic, and Mini-Batch",
    description: "The math and intuition behind gradient descent, learning rate selection, and convergence diagnostics.",
    category: "Linear Regression",
    readTime: 11,
    content: `
    <h3>1. Gradient Descent Intuition</h3>
    <p>Gradient descent minimises the cost function by iteratively stepping in the direction of steepest descent. The step size is controlled by the learning rate α.</p>
    <pre><code># Cost: J(β) = (1/2n) Σ (yᵢ - ŷᵢ)²
# Update: β ← β − α × ∂J/∂β

import numpy as np

def gd_linear_regression(X, y, lr=0.01, epochs=1000):
    n, d  = X.shape
    beta  = np.zeros(d)
    costs = []
    for _ in range(epochs):
        y_pred = X @ beta
        grad   = -(1/n) * X.T @ (y - y_pred)
        beta  -= lr * grad
        costs.append(np.mean((y - y_pred)**2) / 2)
    return beta, costs</code></pre>

    <h3>2. Three Variants</h3>
    <pre><code>from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Stochastic Gradient Descent (one sample per step)
sgd = SGDRegressor(learning_rate='invscaling', eta0=0.01,
                   max_iter=1000, random_state=42)
sgd.fit(X_train_scaled, y_train)

# Mini-batch: balance between stability and speed
# sklearn's SGDRegressor uses mini-batch internally</code></pre>

    <h3>3. Learning Rate Selection</h3>
    <table>
      <thead><tr><th>Learning Rate</th><th>Effect</th></tr></thead>
      <tbody>
        <tr><td>Too large</td><td>Diverges — cost increases</td></tr>
        <tr><td>Too small</td><td>Very slow convergence</td></tr>
        <tr><td>Just right</td><td>Smooth, fast convergence</td></tr>
        <tr><td>Learning rate schedule</td><td>Start large, decay over time — best practice</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "lr-cross-validation",
    title: "Cross-Validation — Reliable Model Evaluation",
    description: "K-Fold, Stratified K-Fold, LOOCV, and nested cross-validation for unbiased performance estimation.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Why Cross-Validation?</h3>
    <p>A single train/test split gives a noisy, potentially misleading performance estimate. Cross-validation averages performance over multiple splits for a more reliable estimate.</p>
    <pre><code>from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X, y = data.data, data.target

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
print(f"CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")</code></pre>

    <h3>2. Common CV Strategies</h3>
    <pre><code>from sklearn.model_selection import (KFold, RepeatedKFold,
                                       LeaveOneOut, ShuffleSplit)

# 5×2 repeated CV — more stable estimates
rkf    = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
# LOOCV — good for very small datasets (expensive for large)
loocv  = LeaveOneOut()
# Monte Carlo / ShuffleSplit — fast approximation
mc     = ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)</code></pre>

    <h3>3. Nested CV for Hyperparameter Tuning</h3>
    <pre><code>from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge

# Nested CV: outer CV evaluates model, inner CV tunes hyperparameters
inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=2)

ridge_gs = GridSearchCV(Ridge(), {'alpha': [0.1,1,10,100]}, cv=inner_cv)
scores   = cross_val_score(ridge_gs, X, y, cv=outer_cv, scoring='r2')
print(f"Nested CV R²: {scores.mean():.4f} ± {scores.std():.4f}")</code></pre>
    `,
  },
  {
    slug: "lr-outlier-detection",
    title: "Outlier Detection and Treatment in Regression",
    description: "IQR, Z-score, Isolation Forest, DBSCAN — how to detect and handle outliers for better regression.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Why Outliers Matter</h3>
    <p>OLS minimises squared errors, so outliers have a disproportionately large influence on fitted coefficients. Detecting and treating them is essential for robust models.</p>

    <h3>2. IQR and Z-Score Methods</h3>
    <pre><code>import numpy as np, pandas as pd

df = pd.DataFrame({'price': [100,120,110,130,115,1000,105]})

# IQR method
Q1, Q3 = df['price'].quantile([0.25, 0.75])
IQR     = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers_iqr = df[(df['price'] < lower) | (df['price'] > upper)]

# Z-score method (assumes normality)
from scipy import stats
z_scores = np.abs(stats.zscore(df['price']))
outliers_z = df[z_scores > 3]</code></pre>

    <h3>3. Isolation Forest</h3>
    <pre><code>from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)
scores    = iso.decision_function(X_train)  # more negative = more anomalous
outlier_mask = iso.predict(X_train) == -1   # -1 = outlier, 1 = inlier

print(f"Detected {outlier_mask.sum()} outliers out of {len(X_train)}")</code></pre>

    <h3>4. Robust Regression</h3>
    <pre><code>from sklearn.linear_model import HuberRegressor, RANSACRegressor

# HuberRegressor: less sensitive to outliers than OLS
huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)

# RANSAC: fits on a random subset of inliers, ignores outliers
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_train, y_train)</code></pre>
    `,
  },
  {
    slug: "lr-elastic-net",
    title: "Elastic Net — Combining Ridge and Lasso Regularization",
    description: "When to use Elastic Net over Ridge or Lasso, the l1_ratio parameter, and cross-validated tuning.",
    category: "Linear Regression",
    readTime: 9,
    content: `
    <h3>1. What is Elastic Net?</h3>
    <p>Elastic Net combines the L1 (Lasso) and L2 (Ridge) penalties. It selects groups of correlated features together — Lasso tends to pick one and discard the rest.</p>
    <pre><code>Cost = MSE + α×l1_ratio×Σ|βᵢ| + α×(1-l1_ratio)×Σβᵢ²
#  l1_ratio=1 → Lasso
#  l1_ratio=0 → Ridge
#  0 < l1_ratio < 1 → Elastic Net</code></pre>

    <h3>2. ElasticNetCV</h3>
    <pre><code>from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
X_s  = StandardScaler().fit_transform(X)

en = ElasticNetCV(
    l1_ratio=np.arange(0.1, 1.0, 0.1),
    alphas=np.logspace(-4, 1, 50),
    cv=5,
    max_iter=10000,
)
en.fit(X_s, y)

print(f"Best alpha:     {en.alpha_:.6f}")
print(f"Best l1_ratio:  {en.l1_ratio_:.2f}")
print(f"Non-zero coefs: {(en.coef_ != 0).sum()}")</code></pre>

    <h3>3. Method Comparison</h3>
    <table>
      <thead><tr><th>Method</th><th>When to Use</th></tr></thead>
      <tbody>
        <tr><td>Ridge</td><td>All features contribute — just want coefficient shrinkage</td></tr>
        <tr><td>Lasso</td><td>Want automatic feature selection; features are independent</td></tr>
        <tr><td>Elastic Net</td><td>Correlated features; want selection and shrinkage</td></tr>
      </tbody>
    </table>
    `,
  },
  {
    slug: "lr-missing-values",
    title: "Missing Values — Imputation Strategies for Regression",
    description: "SimpleImputer, KNN imputation, MICE, and how to include imputation correctly inside a Pipeline.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Types of Missingness</h3>
    <table>
      <thead><tr><th>Type</th><th>Description</th><th>Implication</th></tr></thead>
      <tbody>
        <tr><td>MCAR</td><td>Missing Completely At Random</td><td>Any imputation is safe</td></tr>
        <tr><td>MAR</td><td>Missing At Random (conditional)</td><td>Model-based imputation appropriate</td></tr>
        <tr><td>MNAR</td><td>Missing Not At Random</td><td>Biased — domain expertise needed</td></tr>
      </tbody>
    </table>

    <h3>2. SimpleImputer</h3>
    <pre><code>from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1,2,np.nan],[4,np.nan,6],[7,8,9],[np.nan,2,3]])

imputer_mean   = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_const  = SimpleImputer(strategy='constant', fill_value=0)

print(imputer_mean.fit_transform(X))</code></pre>

    <h3>3. KNN Imputation</h3>
    <pre><code>from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = knn_imp.fit_transform(X)
# Uses k nearest neighbours to estimate missing values</code></pre>

    <h3>4. MICE — Iterative Imputation</h3>
    <pre><code>from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_mice = mice.fit_transform(X)
# Iteratively regresses each feature on others — most accurate</code></pre>
    `,
  },
  {
    slug: "lr-logistic-regression",
    title: "Logistic Regression — Classification with Linear Models",
    description: "Sigmoid function, log-odds, binary and multiclass classification with sklearn LogisticRegression.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. From Linear to Logistic</h3>
    <p>Linear regression predicts continuous values. Logistic regression applies the sigmoid function to map linear output to probabilities between 0 and 1.</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 200)
plt.plot(z, sigmoid(z))
plt.xlabel('z = Xβ'); plt.ylabel('P(y=1)')
plt.title('Sigmoid Function'); plt.show()</code></pre>

    <h3>2. sklearn LogisticRegression</h3>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
# Note: must fit scaler on train, transform test

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))</code></pre>
    `,
  },
  {
    slug: "lr-tree-regression",
    title: "Decision Tree Regressor — Non-Linear Regression",
    description: "How decision trees split continuous targets, max_depth tuning, and comparing with linear regression.",
    category: "Linear Regression",
    readTime: 10,
    content: `
    <h3>1. Decision Tree for Regression</h3>
    <p>Instead of predicting class labels, a Decision Tree Regressor predicts the mean of target values in each leaf node. It can capture non-linear patterns that linear regression cannot.</p>
    <pre><code>from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

for depth in [3, 5, 7, 10, None]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    r2 = r2_score(y_test, dt.predict(X_test))
    print(f"max_depth={str(depth):5s}  Test R²={r2:.4f}")</code></pre>

    <h3>2. Bias-Variance with Depth</h3>
    <pre><code>from sklearn.model_selection import cross_val_score

depths = range(1, 20)
cv_scores  = [cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=42),
                               X, y, cv=5, scoring='r2').mean() for d in depths]
best_depth = depths[np.argmax(cv_scores)]
print(f"Best max_depth by CV: {best_depth}")</code></pre>
    `,
  },
  {
    slug: "lr-xgboost",
    title: "XGBoost for Regression — Extreme Gradient Boosting",
    description: "XGBRegressor, hyperparameter tuning, early stopping, and why XGBoost dominates regression competitions.",
    category: "Linear Regression",
    readTime: 11,
    content: `
    <h3>1. Why XGBoost?</h3>
    <p>XGBoost builds trees sequentially, each correcting the errors of the previous. It adds second-order gradients (Newton boosting), built-in regularization, and hardware-optimised tree building.</p>

    <h3>2. XGBRegressor</h3>
    <pre><code>import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)

y_pred = model.predict(X_test)
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")</code></pre>

    <h3>3. Hyperparameter Tuning</h3>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators':     [100, 200, 500],
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 5, 7],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha':        [0, 0.1, 0.5],
}
search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_dist, n_iter=30, cv=5, scoring='r2', random_state=42)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")</code></pre>
    `,
  },
];
