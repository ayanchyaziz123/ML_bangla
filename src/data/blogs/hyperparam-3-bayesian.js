export const hyperparam_3_bayesian = {
  slug: 'hyperparam-3-bayesian',
  title: 'Bayesian অপটিমাইজেশন: স্মার্ট অনুসন্ধান',
  description: 'Surrogate model (Gaussian Process), acquisition function (Expected Improvement, UCB), exploration বনাম exploitation, এবং scikit-optimize ও Hyperopt দিয়ে বাস্তব উদাহরণ।',
  date: 'মে ২০২৫',
  category: 'হাইপারপ্যারামিটার অপটিমাইজেশন',
  readTime: 12,
  content: `
    <h3>১. Bayesian Optimization কী এবং কেন দরকার?</h3>
    <p>
      Grid Search ও Random Search-এর একটি মৌলিক দুর্বলতা আছে: তারা <em>স্মৃতিহীন</em>। প্রতিটি
      hyperparameter combination আগের ফলাফলকে বিবেচনা না করে independently evaluate করা হয়।
      ভালো ফলাফলের কাছাকাছি আরও ভালো কিছু থাকতে পারে, কিন্তু Grid/Random Search সেটা জানে না।
    </p>
    <p>
      <strong>Bayesian Optimization</strong> এই সমস্যা সমাধান করে। এটি একজন চতুর গোয়েন্দার মতো কাজ করে —
      প্রতিটি পরীক্ষার ফলাফল থেকে শিখে, একটি মানচিত্র তৈরি করে, এবং পরবর্তী পরীক্ষার জন্য সবচেয়ে
      প্রতিশ্রুতিবান জায়গাটি বেছে নেয়।
    </p>
    <p>মূল ধারণা তিনটি:</p>
    <p>
      <strong>Surrogate Model:</strong> একটি সহজ model যা আসল objective function (CV score) approximation করে।<br/>
      <strong>Acquisition Function:</strong> কোথায় পরবর্তী evaluation করব তা নির্ধারণ করে।<br/>
      <strong>Sequential Update:</strong> নতুন ফলাফল পেলে surrogate model আপডেট করে।
    </p>

    <h3>২. Surrogate Model: Gaussian Process</h3>
    <p>
      সবচেয়ে জনপ্রিয় surrogate model হলো <strong>Gaussian Process (GP)</strong>। GP শুধু একটি prediction
      করে না — প্রতিটি point-এ সে বলে দেয় "এখানে আমার prediction হলো X, এবং আমি কতটা sure"।
      এই uncertainty তথ্যটাই Bayesian Optimization-কে শক্তিশালী করে।
    </p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# একটি সহজ 1D উদাহরণ দিয়ে GP বোঝা যাক
# ধরুন এটা হলো learning_rate vs CV score-এর সম্পর্ক (unknown)
def true_objective(x):
    """আসল objective function — আমরা এটা জানি না, GP শিখবে"""
    return np.sin(3*x) * x + 0.3 * np.cos(5*x) + 0.1 * x**2

# মাত্র ৫টি evaluation পয়েন্ট (প্রথম কয়েকটি random sample)
np.random.seed(42)
X_obs = np.array([0.1, 0.4, 0.7, 1.5, 2.3]).reshape(-1, 1)
y_obs = true_objective(X_obs.ravel()) + np.random.normal(0, 0.05, len(X_obs))

# Gaussian Process Regressor
kernel = Matern(nu=2.5)  # Matern kernel: smooth function-এর জন্য ভালো
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                               alpha=1e-6, normalize_y=True)
gp.fit(X_obs, y_obs)

# Prediction করুন পুরো space-এ
X_pred = np.linspace(0, 3, 200).reshape(-1, 1)
y_pred, y_std = gp.predict(X_pred, return_std=True)

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X_pred, true_objective(X_pred.ravel()),
        'k--', linewidth=2, label='True Function (Unknown)')
ax.plot(X_pred, y_pred, 'b-', linewidth=2, label='GP Prediction (Mean)')
ax.fill_between(
    X_pred.ravel(),
    y_pred - 2*y_std,
    y_pred + 2*y_std,
    alpha=0.3, color='blue', label='95% Confidence Interval'
)
ax.scatter(X_obs, y_obs, s=100, c='red', zorder=5, label='Observations')
ax.set_xlabel('Hyperparameter Value')
ax.set_ylabel('Performance Score')
ax.set_title('Gaussian Process Surrogate Model')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("GP পর্যবেক্ষণের কাছে নিশ্চিত (ছোট uncertainty),")
print("দূরে অনিশ্চিত (বড় uncertainty)।")</code></pre>

    <h3>৩. Acquisition Function: কোথায় পরবর্তী পরীক্ষা করব?</h3>
    <p>
      Acquisition function ঠিক করে পরবর্তী কোন hyperparameter value evaluate করব। এটা
      <strong>exploration</strong> (নতুন জায়গা দেখা) এবং <strong>exploitation</strong>
      (ইতোমধ্যে ভালো জায়গার কাছে আরও ভালো খোঁজা) এর মধ্যে balance করে।
    </p>
    <h4>৩.১ Expected Improvement (EI)</h4>
    <p>
      EI হলো সবচেয়ে জনপ্রিয় acquisition function। এটি জিজ্ঞেস করে: "এই point-এ evaluate করলে
      আমার বর্তমান সেরা ফলাফলের চেয়ে গড়ে কতটুকু উন্নতি প্রত্যাশিত?"
    </p>
    <pre><code>from scipy.stats import norm

def expected_improvement(X, gp, y_best, xi=0.01):
    """
    Expected Improvement acquisition function।

    X: candidate points
    gp: trained Gaussian Process
    y_best: সেরা পর্যবেক্ষিত মান
    xi: exploration parameter (বড় xi = বেশি exploration)
    """
    mu, sigma = gp.predict(X, return_std=True)

    # Improvement = max(0, f(x) - y_best)
    Z = (mu - y_best - xi) / (sigma + 1e-9)

    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0  # sigma=0 হলে EI = 0
    return ei

def upper_confidence_bound(X, gp, kappa=2.576):
    """
    Upper Confidence Bound (UCB) acquisition function।
    kappa: exploration-exploitation balance (বড় = বেশি exploration)
    """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma

# X_pred-এ EI ও UCB হিসাব করুন
y_best = max(y_obs)
ei_values = expected_improvement(X_pred, gp, y_best)
ucb_values = upper_confidence_bound(X_pred, gp, kappa=2.0)

# পরবর্তী evaluation কোথায়?
next_x_ei = X_pred[np.argmax(ei_values)]
next_x_ucb = X_pred[np.argmax(ucb_values)]

print(f"EI সুপারিশকৃত পরবর্তী x: {next_x_ei[0]:.4f}")
print(f"UCB সুপারিশকৃত পরবর্তী x: {next_x_ucb[0]:.4f}")

# Acquisition function plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(X_pred, y_pred, 'b-', label='GP Mean')
axes[0].fill_between(X_pred.ravel(),
                     y_pred - 2*y_std, y_pred + 2*y_std,
                     alpha=0.3, color='blue')
axes[0].scatter(X_obs, y_obs, s=100, c='red', zorder=5, label='Observations')
axes[0].axvline(next_x_ei, color='green', linestyle='--', label='Next EI')
axes[0].set_ylabel('Score')
axes[0].legend()

axes[1].plot(X_pred, ei_values, 'g-', linewidth=2, label='Expected Improvement')
axes[1].axvline(next_x_ei, color='green', linestyle='--', label=f'Max EI at x={next_x_ei[0]:.2f}')
axes[1].set_xlabel('Hyperparameter Value')
axes[1].set_ylabel('EI')
axes[1].legend()

plt.suptitle('Expected Improvement Acquisition Function', fontsize=14)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. Exploration বনাম Exploitation</h3>
    <p>
      Bayesian Optimization-এর কেন্দ্রীয় tension হলো exploration ও exploitation-এর মধ্যে balance।
    </p>
    <p>
      <strong>Exploitation:</strong> ইতোমধ্যে ভালো ফলাফল পাওয়া জায়গার কাছে আরও evaluate করুন।
      দ্রুত local optimum খোঁজা যায়, কিন্তু global optimum miss হতে পারে।
    </p>
    <p>
      <strong>Exploration:</strong> অজানা জায়গায় evaluate করুন — হয়তো সেখানে আরও ভালো কিছু আছে।
      বেশি exploration মানে বেশি evaluations, কিন্তু global optimum পাওয়ার সম্ভাবনা বেশি।
    </p>

    <h3>৫. scikit-optimize দিয়ে Bayesian Optimization</h3>
    <p>
      <code>scikit-optimize</code> (skopt) একটি চমৎকার Python library যা sklearn API-এর সাথে compatible।
    </p>
    <pre><code># pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Search space সংজ্ঞায়িত করুন
# Real: continuous, Integer: discrete integer, Categorical: choices
search_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(2, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10)
}

gb = GradientBoostingClassifier(random_state=42)

start = time.time()
bayes_search = BayesSearchCV(
    estimator=gb,
    search_spaces=search_space,
    n_iter=40,            # মাত্র ৪০ evaluations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True            # সেরা model দিয়ে refit করবে
)

bayes_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"\nBayesian Optimization সময়: {elapsed:.1f}s")
print(f"সেরা হাইপারপ্যারামিটার: {bayes_search.best_params_}")
print(f"সেরা CV Score: {bayes_search.best_score_:.4f}")
print(f"Test Score: {bayes_search.best_estimator_.score(X_test, y_test):.4f}")</code></pre>

    <h3>৬. Hyperopt: Tree-structured Parzen Estimator</h3>
    <p>
      <code>Hyperopt</code> আরেকটি জনপ্রিয় library যা <strong>Tree-structured Parzen Estimator (TPE)</strong>
      ব্যবহার করে। TPE GP-এর চেয়ে বেশি scalable এবং categorical parameters-এর সাথেও ভালো কাজ করে।
    </p>
    <pre><code># pip install hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Objective function সংজ্ঞায়িত করুন
def objective(params):
    """Hyperopt minimize করে, তাই negative score return করুন।"""
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_split'] = int(params['min_samples_split'])

    model = GradientBoostingClassifier(
        **params, random_state=42
    )
    score = cross_val_score(
        model, X_train, y_train, cv=5, scoring='accuracy'
    ).mean()

    return {'loss': -score, 'status': STATUS_OK, 'score': score}

# Search space Hyperopt style-এ
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 2, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
}

# Trials object: সব evaluation-এর ইতিহাস সংরক্ষণ করে
trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,    # TPE algorithm
    max_evals=50,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print(f"Hyperopt সেরা params: {best}")
print(f"সেরা CV Score: {-min(trials.losses()):.4f}")

# Convergence visualization
losses = trials.losses()
best_so_far = np.minimum.accumulate(losses)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot([-l for l in losses], 'o', alpha=0.5, color='blue', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('CV Score')
plt.title('প্রতিটি Evaluation-এর Score')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot([-l for l in best_so_far], 'r-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best CV Score')
plt.title('Convergence Curve')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()</code></pre>

    <h3>৭. তিন পদ্ধতির তুলনা</h3>
    <pre><code>import pandas as pd

# একটি সংক্ষিপ্ত empirical তুলনা (আগের sections-এর results ব্যবহার করে)
results = {
    'পদ্ধতি': ['Grid Search', 'Random Search', 'Bayesian (skopt)', 'Hyperopt TPE'],
    'Evaluations': [54, 50, 40, 50],
    'Best CV Score': [
        grid_search.best_score_,
        random_search.best_score_,
        bayes_search.best_score_,
        -min(trials.losses())
    ],
    'Test Score': [
        grid_search.best_estimator_.score(X_test, y_test),
        random_search.best_estimator_.score(X_test, y_test),
        bayes_search.best_estimator_.score(X_test, y_test),
        None  # separate model fitting প্রয়োজন
    ]
}

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))</code></pre>
    <p>
      সাধারণভাবে Bayesian Optimization কম evaluations-এ ভালো বা সমান ফলাফল দেয়। Deep learning-এর
      মতো খরচসাপেক্ষ (GPU-intensive) training-এ Bayesian Optimization সবচেয়ে বেশি কার্যকর।
      পরবর্তী পর্বে দেখব Optuna — আধুনিক এবং সবচেয়ে feature-rich hyperparameter optimization framework।
    </p>
  `
};
