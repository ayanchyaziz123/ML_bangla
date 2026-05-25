export const hyperparam_2_search = {
  slug: 'hyperparam-2-search',
  title: 'Grid Search ও Random Search: হাইপারপ্যারামিটার খোঁজা',
  description: 'Exhaustive Grid Search বনাম দক্ষ Random Search — sklearn GridSearchCV ও RandomizedSearchCV দিয়ে হাইপারপ্যারামিটার টিউনিং।',
  date: 'মে ২০২৫',
  category: 'হাইপারপ্যারামিটার অপটিমাইজেশন',
  readTime: 11,
  content: `
    <h3>১. Grid Search: সম্পূর্ণ অনুসন্ধান</h3>
    <p>
      <strong>Grid Search</strong> হলো হাইপারপ্যারামিটার টিউনিংয়ের সবচেয়ে সহজ systematic পদ্ধতি।
      এটি আপনার দেওয়া প্রতিটি হাইপারপ্যারামিটার combination try করে। যদি আপনি ৩টি learning rate
      এবং ৪টি depth value দেন, Grid Search মোট ৩×৪ = ১২টি combination সব train ও evaluate করবে।
    </p>
    <p>
      এই পদ্ধতির সুবিধা হলো নিশ্চিত — আপনার দেওয়া space-এর মধ্যে সর্বোত্তম combination সে খুঁজে পাবেই।
      কিন্তু সমস্যা হলো সময়: parameter space বড় হলে combinatorial explosion ঘটে।
    </p>
    <pre><code>from sklearn.model_selection import GridSearchCV
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

# Parameter grid সংজ্ঞায়িত করুন
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# মোট combinations: 3 * 3 * 3 * 2 = 54
total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
print(f"মোট combinations: {total_combinations}")
print(f"5-fold CV সহ মোট fits: {total_combinations * 5}")

gb = GradientBoostingClassifier(random_state=42)

start = time.time()
grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,       # সব CPU core ব্যবহার করুন
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"\nGrid Search সময়: {elapsed:.1f} সেকেন্ড")
print(f"সেরা হাইপারপ্যারামিটার: {grid_search.best_params_}")
print(f"সেরা CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.best_estimator_.score(X_test, y_test):.4f}")</code></pre>

    <h3>২. Grid Search-এর ফলাফল বিশ্লেষণ</h3>
    <p>
      GridSearchCV-এর results একটি dictionary হিসেবে পাওয়া যায়। এটি পর্যালোচনা করলে কোন
      parameter কতটা গুরুত্বপূর্ণ তা বোঝা যায়।
    </p>
    <pre><code>import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# সব results DataFrame-এ রাখুন
results_df = pd.DataFrame(grid_search.cv_results_)

# Top 10 combinations
top_10 = results_df.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']
]
print("Top 10 Combinations:")
print(top_10.to_string(index=False))

# learning_rate vs mean_test_score visualization
lr_grouped = results_df.groupby(
    results_df['param_learning_rate']
)['mean_test_score'].mean()

plt.figure(figsize=(8, 5))
plt.bar(lr_grouped.index.astype(str), lr_grouped.values, color='steelblue', edgecolor='black')
plt.xlabel('Learning Rate')
plt.ylabel('Mean CV Score')
plt.title('Learning Rate-এর প্রভাব on CV Score')
plt.tight_layout()
plt.show()

# Overfitting check
results_df['overfit_gap'] = results_df['mean_train_score'] - results_df['mean_test_score']
print(f"\nগড় Overfitting Gap: {results_df['overfit_gap'].mean():.4f}")
print(f"সর্বোচ্চ Overfitting Gap: {results_df['overfit_gap'].max():.4f}")</code></pre>

    <h3>৩. Random Search: Bergstra ও Bengio-র পদ্ধতি</h3>
    <p>
      ২০১২ সালে James Bergstra ও Yoshua Bengio একটি গুরুত্বপূর্ণ paper প্রকাশ করেন:
      "Random Search for Hyper-Parameter Optimization"। তারা দেখান যে <strong>Random Search প্রায়ই
      Grid Search-এর চেয়ে কম সময়ে ভালো ফলাফল দেয়।</strong>
    </p>
    <p>
      কারণটা চমৎকার: বাস্তবে সব hyperparameter সমান গুরুত্বপূর্ণ নয়। কিছু hyperparameter (যেমন
      learning rate) performance-এ বড় প্রভাব ফেলে, বাকিগুলো কম। Grid Search-এ অনেক সময় নষ্ট হয়
      কম গুরুত্বপূর্ণ parameter-এর সব values try করতে। Random Search কম গুরুত্বপূর্ণ parameter-এর
      জন্য কম values try করেও গুরুত্বপূর্ণ parameter-এর বেশি বৈচিত্র্য cover করতে পারে।
    </p>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import time

# Continuous distributions ব্যবহার করুন — এটাই Random Search-এর আসল শক্তি
param_dist = {
    'n_estimators': randint(50, 300),          # 50 থেকে 300 পর্যন্ত যেকোনো integer
    'max_depth': randint(2, 15),               # 2 থেকে 15
    'learning_rate': uniform(0.01, 0.29),      # 0.01 থেকে 0.30 পর্যন্ত continuous
    'subsample': uniform(0.6, 0.4),            # 0.6 থেকে 1.0 পর্যন্ত
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

gb2 = GradientBoostingClassifier(random_state=42)

start = time.time()
random_search = RandomizedSearchCV(
    estimator=gb2,
    param_distributions=param_dist,
    n_iter=60,        # Grid Search-এর 54 combinations-এর কাছাকাছি
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)
elapsed_rand = time.time() - start

print(f"\nRandom Search সময়: {elapsed_rand:.1f} সেকেন্ড")
print(f"সেরা হাইপারপ্যারামিটার: {random_search.best_params_}")
print(f"সেরা CV Score: {random_search.best_score_:.4f}")
print(f"Test Score: {random_search.best_estimator_.score(X_test, y_test):.4f}")</code></pre>

    <h3>৪. Grid Search বনাম Random Search: প্রত্যক্ষ তুলনা</h3>
    <pre><code>import matplotlib.pyplot as plt
import numpy as np

# উভয় পদ্ধতির best scores over iterations plot করুন
grid_scores_sorted = sorted(
    grid_search.cv_results_['mean_test_score'], reverse=True
)
rand_scores_sorted = sorted(
    random_search.cv_results_['mean_test_score'], reverse=True
)

# Running maximum: iteration বাড়ার সাথে সেরা score কীভাবে উন্নত হয়
grid_running_max = np.maximum.accumulate(
    grid_search.cv_results_['mean_test_score']
)
rand_running_max = np.maximum.accumulate(
    random_search.cv_results_['mean_test_score']
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Running best score
axes[0].plot(grid_running_max, 'b-', label='Grid Search', linewidth=2)
axes[0].plot(rand_running_max, 'r-', label='Random Search', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Best CV Score (running max)')
axes[0].set_title('সেরা Score-এর উন্নতি')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Score distribution
axes[1].hist(grid_search.cv_results_['mean_test_score'],
             bins=20, alpha=0.6, color='blue', label='Grid Search')
axes[1].hist(random_search.cv_results_['mean_test_score'],
             bins=20, alpha=0.6, color='red', label='Random Search')
axes[1].set_xlabel('CV Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Score Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- তুলনা সারসংক্ষেপ ---")
print(f"Grid Search  - Best: {grid_search.best_score_:.4f}, "
      f"Combinations: {len(grid_search.cv_results_['mean_test_score'])}")
print(f"Random Search - Best: {random_search.best_score_:.4f}, "
      f"Iterations: {len(random_search.cv_results_['mean_test_score'])}")</code></pre>

    <h3>৫. Parallel Execution: n_jobs দিয়ে গতি বাড়ানো</h3>
    <p>
      GridSearchCV এবং RandomizedSearchCV-এ <code>n_jobs</code> parameter দিয়ে parallel processing
      নিয়ন্ত্রণ করা যায়:
    </p>
    <pre><code>import multiprocessing

print(f"আপনার CPU cores: {multiprocessing.cpu_count()}")

# n_jobs=-1 মানে সব available cores ব্যবহার করো
# n_jobs=2  মানে ২টি parallel job চালাও
# n_jobs=1  মানে sequential (কোনো parallelism নেই)

# সময়ের পার্থক্য দেখুন
from sklearn.svm import SVC

svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

svm = SVC(random_state=42)

# Sequential
start = time.time()
gs_seq = GridSearchCV(svm, svm_params, cv=5, n_jobs=1)
gs_seq.fit(X_train, y_train)
time_seq = time.time() - start

# Parallel
start = time.time()
gs_par = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
gs_par.fit(X_train, y_train)
time_par = time.time() - start

print(f"Sequential (n_jobs=1):  {time_seq:.1f}s")
print(f"Parallel (n_jobs=-1):   {time_par:.1f}s")
print(f"Speedup: {time_seq/time_par:.1f}x")</code></pre>

    <h3>৬. Search Space Design: বুদ্ধিমানের সাথে পরিসর নির্ধারণ</h3>
    <p>
      ভালো search space design করা হাইপারপ্যারামিটার টিউনিংয়ের অর্ধেক কাজ। কিছু গুরুত্বপূর্ণ নির্দেশিকা:
    </p>
    <p>
      <strong>Logarithmic scale ব্যবহার করুন:</strong> Learning rate ও C-এর মতো parameters-এ।
      0.001, 0.01, 0.1, 1, 10 হলো uniformly spaced in log scale।
    </p>
    <pre><code>from scipy.stats import loguniform, uniform, randint
import numpy as np

# Log-uniform distribution: learning rate, C, alpha-এর জন্য
# 0.0001 থেকে 1 পর্যন্ত log scale-এ uniform
lr_dist = loguniform(1e-4, 1)
print("Learning rate samples (log-uniform):")
print(sorted(lr_dist.rvs(size=8, random_state=42)))

# Uniform distribution: subsample-এর জন্য
sub_dist = uniform(0.5, 0.5)  # 0.5 থেকে 1.0
print("\nSubsample samples (uniform):")
print(sorted(sub_dist.rvs(size=8, random_state=42)))

# Integer distribution: n_estimators, max_depth-এর জন্য
est_dist = randint(50, 501)
print("\nN_estimators samples (randint):")
print(sorted(est_dist.rvs(size=8, random_state=42)))

# XGBoost-এর জন্য comprehensive search space
xgb_param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': loguniform(1e-3, 0.3),
    'subsample': uniform(0.6, 0.4),           # 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4),    # 0.6-1.0
    'reg_alpha': loguniform(1e-5, 10),        # L1 regularization
    'reg_lambda': loguniform(1e-5, 10),       # L2 regularization
    'min_child_weight': randint(1, 10),
    'gamma': loguniform(1e-5, 1)
}

print("\nXGBoost Search Space সংজ্ঞায়িত হয়েছে।")
print(f"মোট continuous parameters: {len(xgb_param_dist)}")</code></pre>

    <h3>৭. ব্যবহারিক পরামর্শ</h3>
    <p>
      <strong>কখন Grid Search ব্যবহার করবেন:</strong> Parameter space ছোট (৩-৪টি parameters, প্রতিটিতে ৩-৫টি values)
      এবং কম বেশি জানেন কোথায় ভালো values আছে। Fine-tuning-এর জন্য ভালো।
    </p>
    <p>
      <strong>কখন Random Search ব্যবহার করবেন:</strong> Parameter space বড়, continuous parameters আছে,
      কোথায় ভালো values তা জানেন না। প্রাথমিক exploration-এর জন্য সর্বোত্তম।
    </p>
    <pre><code># Best practice: দুই ধাপে tuning
# ধাপ ১: Random Search দিয়ে broad exploration
coarse_params = {
    'max_depth': randint(2, 20),
    'n_estimators': randint(50, 500),
    'learning_rate': loguniform(1e-3, 1)
}

coarse_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    coarse_params, n_iter=50, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)
coarse_search.fit(X_train, y_train)
best_coarse = coarse_search.best_params_
print(f"Coarse best: {best_coarse}")

# ধাপ ২: সেরা এলাকার আশেপাশে Grid Search দিয়ে fine-tune
fine_grid = {
    'max_depth': [
        max(1, best_coarse['max_depth'] - 2),
        best_coarse['max_depth'],
        best_coarse['max_depth'] + 2
    ],
    'n_estimators': [
        max(50, best_coarse['n_estimators'] - 50),
        best_coarse['n_estimators'],
        best_coarse['n_estimators'] + 50
    ],
    'learning_rate': [
        best_coarse['learning_rate'] * 0.5,
        best_coarse['learning_rate'],
        best_coarse['learning_rate'] * 2
    ]
}

fine_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    fine_grid, cv=5, scoring='accuracy', n_jobs=-1
)
fine_search.fit(X_train, y_train)
print(f"Fine-tuned best: {fine_search.best_params_}")
print(f"Final CV Score: {fine_search.best_score_:.4f}")
print(f"Final Test Score: {fine_search.best_estimator_.score(X_test, y_test):.4f}")</code></pre>
    <p>
      Grid Search ও Random Search উভয়ই model-agnostic — যেকোনো sklearn-compatible estimator-এর সাথে কাজ করে।
      তবে উভয়ই প্রতিটি evaluation independent — আগের ফলাফল থেকে কিছু শেখে না। পরবর্তী পর্বে দেখব
      Bayesian Optimization, যা আগের evaluations থেকে শিখে আরও স্মার্টভাবে search করে।
    </p>
  `
};
