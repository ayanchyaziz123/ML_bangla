export const hyperparam_4_optuna = {
  slug: 'hyperparam-4-optuna',
  title: 'Optuna: আধুনিক হাইপারপ্যারামিটার অপটিমাইজেশন',
  description: 'TPE স্যাম্পলার, প্রুনিং ও ভিজুয়ালাইজেশন সহ Optuna ফ্রেমওয়ার্ক দিয়ে দক্ষ হাইপারপ্যারামিটার টিউনিং',
  date: 'মে ২০২৫',
  category: 'হাইপারপ্যারামিটার অপটিমাইজেশন',
  readTime: 13,
  content: `
<h3>Optuna কেন?</h3>
<p>Optuna একটি ফ্রেমওয়ার্ক-অজ্ঞেয়বাদী (framework-agnostic) হাইপারপ্যারামিটার অপ্টিমাইজেশন লাইব্রেরি। বৈশিষ্ট্য:</p>
<ul>
<li><strong>Define-by-run API:</strong> Python কোডেই সার্চ স্পেস সংজ্ঞায়িত করুন</li>
<li><strong>TPE স্যাম্পলার:</strong> Bayesian-ভিত্তিক দক্ষ অনুসন্ধান</li>
<li><strong>Pruning:</strong> অকার্যকর ট্রায়াল মাঝপথে থামিয়ে দেয়</li>
<li><strong>ভিজুয়ালাইজেশন:</strong> অপ্টিমাইজেশন ইতিহাস দেখুন</li>
</ul>

<h3>প্রথম Optuna স্টাডি</h3>

<pre><code class="language-python">import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    """প্রতিটি ট্রায়ালে মূল্যায়ন ফাংশন"""
    # হাইপারপ্যারামিটার সার্চ স্পেস সংজ্ঞায়ন
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    return cv_scores.mean()

# স্টাডি তৈরি ও চালানো
study = optuna.create_study(direction='maximize')  # সর্বোচ্চ ROC-AUC চাই
study.optimize(objective, n_trials=100, n_jobs=-1)

print(f"সেরা ROC-AUC: {study.best_value:.4f}")
print(f"সেরা প্যারামিটার: {study.best_params}")
</code></pre>

<h3>প্রুনিং — অকার্যকর ট্রায়াল থামানো</h3>

<pre><code class="language-python">import lightgbm as lgb
from sklearn.model_selection import train_test_split
from optuna.integration import LightGBMPruningCallback

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective_with_pruning(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        callbacks=[
            LightGBMPruningCallback(trial, 'auc')
        ]
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model.best_score_['valid_0']['auc']

# Median Pruner — মধ্যমানের নিচে থাকলে ট্রায়াল থামায়
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study_lgbm = optuna.create_study(direction='maximize', pruner=pruner)
study_lgbm.optimize(objective_with_pruning, n_trials=200, n_jobs=2)

print(f"\\nসেরা AUC: {study_lgbm.best_value:.4f}")
</code></pre>

<h3>Optuna ভিজুয়ালাইজেশন</h3>

<pre><code class="language-python">from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_slice
)

# অপ্টিমাইজেশন ইতিহাস
fig = plot_optimization_history(study)
fig.show()

# প্যারামিটার গুরুত্ব
fig = plot_param_importances(study)
fig.show()

# প্যারামিটার সম্পর্ক কনটুর প্লট
fig = plot_contour(study, params=['n_estimators', 'max_depth'])
fig.show()

# স্লাইস প্লট — প্রতিটি প্যারামিটারের প্রভাব
fig = plot_slice(study)
fig.show()
</code></pre>

<h3>PyTorch দিয়ে নিউরাল নেটওয়ার্ক টিউনিং</h3>

<pre><code class="language-python">import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def build_model(trial, input_dim):
    """Optuna দিয়ে নিউরাল আর্কিটেকচার সার্চ"""
    n_layers = trial.suggest_int('n_layers', 1, 4)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'elu'])

    layers = [nn.Linear(input_dim, trial.suggest_int('units_l0', 32, 256))]
    layers.append(getattr(nn, activation.upper() if activation == 'relu' else activation.capitalize())())

    for i in range(1, n_layers):
        units = trial.suggest_int(f'units_l{i}', 32, 256)
        dropout = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
        layers.extend([
            nn.Linear(layers[-2].out_features if hasattr(layers[-2], 'out_features')
                     else units, units),
            nn.Dropout(dropout),
            getattr(nn, activation.upper() if activation == 'relu' else activation.capitalize())()
        ])

    layers.append(nn.Linear(units, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

# ব্যবহার:
# study_nn = optuna.create_study(direction='minimize')
# study_nn.optimize(lambda trial: train_nn(trial, X_train, y_train), n_trials=50)
</code></pre>

<h3>স্টাডি স্টোরেজ ও পুনরায় চালানো</h3>

<pre><code class="language-python"># SQLite দিয়ে স্থায়ী স্টোরেজ
storage = "sqlite:///optuna_study.db"
study = optuna.create_study(
    study_name="rf-tuning-v2",
    direction='maximize',
    storage=storage,
    load_if_exists=True  # বিদ্যমান স্টাডি লোড করুন
)

study.optimize(objective, n_trials=50)

# পরে পুনরায় চালানো
study_loaded = optuna.load_study(study_name="rf-tuning-v2", storage=storage)
print(f"মোট ট্রায়াল: {len(study_loaded.trials)}")
print(f"সেরা মান: {study_loaded.best_value:.4f}")
</code></pre>
`
};
