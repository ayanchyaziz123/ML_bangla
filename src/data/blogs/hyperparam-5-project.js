export const hyperparam_5_project = {
  slug: 'hyperparam-5-project',
  title: 'হাইপারপ্যারামিটার অপটিমাইজেশন প্রজেক্ট',
  description: 'Grid Search, Random Search, Bayesian Optimization ও Optuna তুলনা করে সেরা XGBoost পাইপলাইন তৈরি করুন',
  date: 'মে ২০২৫',
  category: 'হাইপারপ্যারামিটার অপটিমাইজেশন',
  readTime: 15,
  content: `
<h3>প্রজেক্ট লক্ষ্য</h3>
<p>House Price ডেটাসেটে XGBoost মডেল টিউন করে চারটি পদ্ধতি তুলনা করব: Grid Search, Random Search, Hyperopt (Bayesian) এবং Optuna।</p>

<h3>ডেটা প্রস্তুতি</h3>

<pre><code class="language-python">import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ডেটা লোড
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"ডেটা আকার: {X.shape}")
print(f"লক্ষ্য: বাড়ির মূল্য (মিলিয়ন ডলার)")
print(X.describe().round(2))

# ক্রস-ভ্যালিডেশন সেটআপ
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_params(params, X, y, cv):
    """RMSE দিয়ে মূল্যায়ন"""
    model = xgb.XGBRegressor(**params, random_state=42, eval_metric='rmse', verbosity=0)
    scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return scores.mean()

# বেসলাইন
baseline_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
baseline_score = evaluate_params(baseline_params, X, y, cv)
print(f"\\nবেসলাইন RMSE: {baseline_score:.4f}")
</code></pre>

<h3>পদ্ধতি ১: Grid Search</h3>

<pre><code class="language-python">from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
}
# মোট সমন্বয়: 3 * 3 * 3 = 27

grid_model = xgb.XGBRegressor(random_state=42, verbosity=0)

start = time.time()
grid_search = GridSearchCV(
    grid_model, param_grid, cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=0
)
grid_search.fit(X, y)
grid_time = time.time() - start

grid_score = -grid_search.best_score_
print(f"Grid Search RMSE: {grid_score:.4f}")
print(f"সময়: {grid_time:.1f} সেকেন্ড")
print(f"সেরা প্যারামিটার: {grid_search.best_params_}")
</code></pre>

<h3>পদ্ধতি ২: Random Search</h3>

<pre><code class="language-python">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 1000),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.001, 0.5),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 1),
}

start = time.time()
random_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_dist, n_iter=50, cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, random_state=42
)
random_search.fit(X, y)
random_time = time.time() - start

random_score = -random_search.best_score_
print(f"Random Search RMSE: {random_score:.4f}")
print(f"সময়: {random_time:.1f} সেকেন্ড")
</code></pre>

<h3>পদ্ধতি ৩: Optuna</h3>

<pre><code class="language-python">import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
    }
    return evaluate_params(params, X, y, cv)

start = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(xgb_objective, n_trials=50, n_jobs=-1)
optuna_time = time.time() - start

optuna_score = study.best_value
print(f"Optuna RMSE: {optuna_score:.4f}")
print(f"সময়: {optuna_time:.1f} সেকেন্ড")
</code></pre>

<h3>চূড়ান্ত তুলনা</h3>

<pre><code class="language-python">import matplotlib.pyplot as plt

results = {
    'বেসলাইন': {'rmse': baseline_score, 'time': 0, 'trials': 1},
    'Grid Search': {'rmse': grid_score, 'time': grid_time, 'trials': 27},
    'Random Search': {'rmse': random_score, 'time': random_time, 'trials': 50},
    'Optuna': {'rmse': optuna_score, 'time': optuna_time, 'trials': 50},
}

print("\\n" + "="*55)
print(f"{'পদ্ধতি':<20} {'RMSE':>8} {'উন্নতি':>10} {'সময় (সে)':>12}")
print("="*55)
for method, res in results.items():
    improvement = (baseline_score - res['rmse']) / baseline_score * 100
    print(f"{method:<20} {res['rmse']:>8.4f} {improvement:>9.1f}% {res['time']:>11.1f}")

# বার চার্ট
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods = list(results.keys())
rmse_vals = [r['rmse'] for r in results.values()]
time_vals = [r['time'] for r in results.values()]

colors = ['gray', 'blue', 'green', 'purple']
bars = ax1.bar(methods, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('RMSE (কম = ভালো)')
ax1.set_title('RMSE তুলনা')
for bar, val in zip(bars, rmse_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax2.bar(methods[1:], time_vals[1:], color=colors[1:], alpha=0.7, edgecolor='black')
ax2.set_ylabel('সময় (সেকেন্ড, কম = দ্রুত)')
ax2.set_title('কম্পিউটেশন সময় তুলনা')
plt.tight_layout()
plt.savefig('hyperparameter_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
</code></pre>

<h4>চূড়ান্ত সুপারিশ</h4>
<table>
<tr><th>পরিস্থিতি</th><th>সুপারিশ</th></tr>
<tr><td>সার্চ স্পেস ছোট (&lt;1000)</td><td>Grid Search</td></tr>
<tr><td>প্রথম অন্বেষণ</td><td>Random Search</td></tr>
<tr><td>সীমিত বাজেট, সর্বোচ্চ পারফরম্যান্স</td><td>Optuna</td></tr>
<tr><td>নিউরাল নেটওয়ার্ক আর্কিটেকচার সার্চ</td><td>Optuna + Pruning</td></tr>
<tr><td>GPU ক্লাস্টারে বড় এক্সপেরিমেন্ট</td><td>Optuna (distributed)</td></tr>
</table>
`
};
