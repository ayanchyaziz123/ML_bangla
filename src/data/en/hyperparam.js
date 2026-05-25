export const hyperparamEn = [
  {
    slug: 'hyperparam-1-basics',
    title: 'Hyperparameters: The Hidden Controls of Models',
    description: 'Parameters vs hyperparameters, why tuning matters, and validation strategy',
    category: 'Hyperparameter Optimization',
    content: `
<h3>Parameters vs Hyperparameters</h3>
<ul>
<li><strong>Parameters:</strong> Learned from data (weights, biases) — optimized by gradient descent</li>
<li><strong>Hyperparameters:</strong> Set before training (learning rate, depth) — optimized by you</li>
</ul>

<pre><code class="language-python">from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impact of hyperparameters
results = []
for n_estimators in [10, 50, 100, 200]:
    for max_depth in [2, 4, 6, None]:
        model = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        results.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'mean_acc': scores.mean(),
            'std_acc': scores.std()
        })

import pandas as pd
df = pd.DataFrame(results).sort_values('mean_acc', ascending=False)
print(df.head(5).to_string(index=False))

# Key hyperparameters by model type
hyperparams = {
    'Random Forest': ['n_estimators', 'max_depth', 'min_samples_split'],
    'XGBoost': ['learning_rate', 'n_estimators', 'max_depth', 'subsample'],
    'Neural Network': ['learning_rate', 'hidden_layers', 'batch_size', 'dropout'],
    'SVM': ['C', 'kernel', 'gamma'],
}
for model, params in hyperparams.items():
    print(f"{model}: {params}")
</code></pre>
`
  },
  {
    slug: 'hyperparam-2-search',
    title: 'Grid Search and Random Search',
    description: 'Exhaustive grid search vs efficient random search, search space design, sklearn usage',
    category: 'Hyperparameter Optimization',
    content: `
<h3>Grid Search vs Random Search</h3>
<p>Bergstra & Bengio (2012) showed that random search is more efficient than grid search when few hyperparameters truly matter — randomly sampling explores the important dimensions more times in the same budget.</p>

<pre><code class="language-python">from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
import time

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

start = time.time()
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42),
                            param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start
print(f"Grid Search: {-grid_search.best_score_:.4f} ROC-AUC in {grid_time:.1f}s")
print(f"  Trials: {len(grid_search.cv_results_['mean_test_score'])}")

# Random Search
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 12),
    'learning_rate': uniform(0.001, 0.4),
    'subsample': uniform(0.5, 0.5),
    'min_samples_split': randint(2, 20),
}

start = time.time()
random_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
                                    param_dist, n_iter=50, cv=5,
                                    scoring='roc_auc', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
random_time = time.time() - start
print(f"Random Search: {-random_search.best_score_:.4f} ROC-AUC in {random_time:.1f}s")
print(f"Best params: {random_search.best_params_}")
</code></pre>
`
  },
  {
    slug: 'hyperparam-3-bayesian',
    title: 'Bayesian Optimization: Smart Search',
    description: 'Gaussian Process surrogate, Expected Improvement, scikit-optimize and Hyperopt',
    category: 'Hyperparameter Optimization',
    content: `
<h3>Bayesian Optimization Algorithm</h3>
<ol>
<li>Evaluate a few initial random points</li>
<li>Fit a surrogate model (Gaussian Process) to observed results</li>
<li>Use acquisition function to select next point (balance explore/exploit)</li>
<li>Evaluate objective at chosen point</li>
<li>Update surrogate and repeat</li>
</ol>

<pre><code class="language-python">from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define search space with prior knowledge
search_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 12),
    'learning_rate': Real(1e-3, 0.5, prior='log-uniform'),  # log scale
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 2),
    'reg_alpha': Real(1e-5, 1, prior='log-uniform'),
    'reg_lambda': Real(1e-5, 1, prior='log-uniform'),
}

bayes_search = BayesSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    search_space, n_iter=50, cv=5,
    scoring='roc_auc', n_jobs=-1, random_state=42
)

start = time.time()
bayes_search.fit(X_train, y_train)
print(f"Bayesian ROC-AUC: {bayes_search.best_score_:.4f} in {time.time()-start:.1f}s")
print(f"Best params: {bayes_search.best_params_}")
</code></pre>
`
  },
  {
    slug: 'hyperparam-4-optuna',
    title: 'Optuna: Modern Hyperparameter Optimization',
    description: 'TPE sampler, MedianPruner, visualization, and integration with XGBoost and PyTorch',
    category: 'Hyperparameter Optimization',
    content: `
<h3>Optuna Key Features</h3>
<ul>
<li><strong>Define-by-run API:</strong> Define search space inside the objective function</li>
<li><strong>TPE Sampler:</strong> Tree-structured Parzen Estimator — efficient Bayesian search</li>
<li><strong>Pruning:</strong> Kill unpromising trials early (like early stopping for HPO)</li>
<li><strong>Parallelism:</strong> Distribute across multiple processes/machines</li>
</ul>

<pre><code class="language-python">import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return scores.mean()

# Storage for resumable studies
study = optuna.create_study(
    direction='maximize',
    study_name='gbc-tuning',
    storage='sqlite:///optuna.db',
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)
study.optimize(objective, n_trials=100, n_jobs=-1)

print(f"Best ROC-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualization
from optuna.visualization import plot_optimization_history, plot_param_importances
plot_optimization_history(study).show()
plot_param_importances(study).show()
</code></pre>
`
  },
  {
    slug: 'hyperparam-5-project',
    title: 'Hyperparameter Optimization Project',
    description: 'End-to-end comparison of Grid, Random, Bayesian, and Optuna on an XGBoost pipeline',
    category: 'Hyperparameter Optimization',
    content: `
<h3>Complete Tuning Comparison</h3>
<pre><code class="language-python">import numpy as np, pandas as pd, time
import xgboost as xgb, optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)

housing = fetch_california_housing()
X, y = housing.data, housing.target
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_params(params):
    m = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    return -cross_val_score(m, X, y, cv=cv, scoring='neg_root_mean_squared_error').mean()

# Baseline
base = eval_params({'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1})
print(f"Baseline RMSE: {base:.4f}")

# Grid Search (small grid)
grid_res = GridSearchCV(xgb.XGBRegressor(random_state=42, verbosity=0),
    {'n_estimators': [100, 300], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]},
    cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
t = time.time(); grid_res.fit(X, y); grid_t = time.time() - t
print(f"Grid   RMSE: {-grid_res.best_score_:.4f}  ({grid_t:.0f}s)")

# Random Search
rnd_res = RandomizedSearchCV(xgb.XGBRegressor(random_state=42, verbosity=0),
    {'n_estimators': randint(50,500), 'max_depth': randint(3,12),
     'learning_rate': uniform(0.001,0.4), 'subsample': uniform(0.5,0.5)},
    n_iter=50, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
t = time.time(); rnd_res.fit(X, y); rnd_t = time.time() - t
print(f"Random RMSE: {-rnd_res.best_score_:.4f}  ({rnd_t:.0f}s)")

# Optuna
def obj(trial):
    return eval_params({
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.4, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    })
study = optuna.create_study(direction='minimize')
t = time.time(); study.optimize(obj, n_trials=50, n_jobs=-1); opt_t = time.time() - t
print(f"Optuna RMSE: {study.best_value:.4f}  ({opt_t:.0f}s)")

# Summary
methods = ['Baseline', 'Grid', 'Random', 'Optuna']
rmses = [base, -grid_res.best_score_, -rnd_res.best_score_, study.best_value]
improvement = [(base - r) / base * 100 for r in rmses]
print("\\n" + "="*50)
for m, r, i in zip(methods, rmses, improvement):
    print(f"{m:<10} RMSE={r:.4f}  Improvement={i:+.1f}%")
</code></pre>
`
  },
];
