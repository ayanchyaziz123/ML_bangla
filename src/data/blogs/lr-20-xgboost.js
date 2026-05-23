export const lr_20_xgboost = {
  title: "XGBoost Regression — প্রতিযোগিতায় সবার প্রিয় মডেল",
  description: "XGBoost কীভাবে Gradient Boosting ব্যবহার করে, Random Forest-এর চেয়ে কেন ভালো, hyperparameter tuning, এবং real-world regression সমস্যায় প্রয়োগ।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 12,
  slug: "lr-xgboost",
  content: `
    <h3>১. XGBoost কী?</h3>
    <p><strong>XGBoost</strong> (eXtreme Gradient Boosting) হলো Gradient Boosting-এর একটি অত্যন্ত দ্রুত এবং শক্তিশালী implementation। Kaggle-সহ বেশিরভাগ ML প্রতিযোগিতায় এটি সবচেয়ে বেশি ব্যবহৃত হয়।</p>
    <p>Random Forest যেখানে tree-গুলো <strong>parallel</strong> (স্বাধীনভাবে) তৈরি করে, XGBoost সেখানে tree-গুলো <strong>sequential</strong> (একটির পর একটি) তৈরি করে — প্রতিটি নতুন tree আগের tree-র ভুল শুধরে নেয়।</p>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Gradient Boosting — ভুল শুধরে শুধরে শেখা</text>
        <!-- Tree 1 -->
        <rect x="20" y="28" width="70" height="36" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="55" y="47" text-anchor="middle" font-size="9" fill="#1e40af">Tree 1</text>
        <text x="55" y="59" text-anchor="middle" font-size="8" fill="#3b82f6">residual→</text>
        <!-- Arrow -->
        <line x1="95" y1="46" x2="120" y2="46" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>
        <!-- Tree 2 -->
        <rect x="125" y="28" width="70" height="36" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="160" y="47" text-anchor="middle" font-size="9" fill="#1e40af">Tree 2</text>
        <text x="160" y="59" text-anchor="middle" font-size="8" fill="#3b82f6">residual→</text>
        <line x1="200" y1="46" x2="225" y2="46" stroke="#94a3b8" stroke-width="1.5"/>
        <!-- Tree 3 -->
        <rect x="230" y="28" width="70" height="36" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="265" y="47" text-anchor="middle" font-size="9" fill="#1e40af">Tree 3</text>
        <text x="265" y="59" text-anchor="middle" font-size="8" fill="#3b82f6">residual→</text>
        <line x1="305" y1="46" x2="330" y2="46" stroke="#94a3b8" stroke-width="1.5"/>
        <text x="345" y="48" font-size="11" fill="#94a3b8">···</text>
        <!-- Final -->
        <rect x="375" y="28" width="85" height="36" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="417" y="44" text-anchor="middle" font-size="9" fill="#16a34a">চূড়ান্ত</text>
        <text x="417" y="57" text-anchor="middle" font-size="9" fill="#16a34a">prediction</text>
        <text x="240" y="100" text-anchor="middle" font-size="9" fill="#6b7280">প্রতিটি tree আগের tree-র error (residual) শেখার চেষ্টা করে</text>
      </svg>
    </div>

    <h3>২. XGBoost Install ও Basic ব্যবহার</h3>
    <pre><code"># Install (যদি না থাকে):
# pip install xgboost

import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# sklearn API দিয়ে
model = xgb.XGBRegressor(
    n_estimators=100,       # কতটি tree
    learning_rate=0.1,      # প্রতিটি tree-র contribution
    max_depth=6,            # tree-র গভীরতা
    subsample=0.8,          # প্রতিটি tree-র জন্য কতটুকু ডেটা (Bagging)
    colsample_bytree=0.8,   # প্রতিটি tree-র জন্য কতটুকু feature
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print(f"Train R²: {r2_score(y_train, model.predict(X_train)):.4f}")
print(f"Test  R²: {r2_score(y_test,  model.predict(X_test)):.4f}")</code></pre>

    <h3>৩. Early Stopping — কখন থামবো?</h3>
    <p>বেশি tree যোগ করলে Overfitting হয়। Early Stopping validation score কমলে training থামিয়ে দেয়।</p>
    <pre><code">model_es = xgb.XGBRegressor(
    n_estimators=1000,      # অনেক বেশি রাখো
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
model_es.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,   # 50 round-এ improvement না হলে থামো
    verbose=False,
)
print(f"সেরা iteration: {model_es.best_iteration}")
print(f"Test R²: {r2_score(y_test, model_es.predict(X_test)):.4f}")</code></pre>

    <h3>৪. গুরুত্বপূর্ণ Hyperparameter</h3>
    <table>
      <thead><tr><th>Parameter</th><th>কী করে</th><th>সাধারণ range</th></tr></thead>
      <tbody>
        <tr><td><strong>n_estimators</strong></td><td>tree সংখ্যা</td><td>100–1000 (early stopping সহ)</td></tr>
        <tr><td><strong>learning_rate (eta)</strong></td><td>প্রতিটি tree-র প্রভাব</td><td>0.01–0.3</td></tr>
        <tr><td><strong>max_depth</strong></td><td>tree গভীরতা</td><td>3–10</td></tr>
        <tr><td><strong>subsample</strong></td><td>প্রতিটি tree-তে কতটুকু row</td><td>0.6–1.0</td></tr>
        <tr><td><strong>colsample_bytree</strong></td><td>প্রতিটি tree-তে কতটুকু feature</td><td>0.6–1.0</td></tr>
        <tr><td><strong>reg_alpha (L1)</strong></td><td>Lasso regularization</td><td>0–1</td></tr>
        <tr><td><strong>reg_lambda (L2)</strong></td><td>Ridge regularization</td><td>1–10</td></tr>
      </tbody>
    </table>

    <h3>৫. Hyperparameter Tuning</h3>
    <pre><code">from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators':      [100, 200, 500],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'max_depth':         [3, 5, 7, 9],
    'subsample':         [0.6, 0.8, 1.0],
    'colsample_bytree':  [0.6, 0.8, 1.0],
    'reg_alpha':         [0, 0.1, 0.5],
    'reg_lambda':        [1, 2, 5],
}

search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_dist,
    n_iter=30,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"সেরা parameter: {search.best_params_}")
print(f"Test R²: {search.score(X_test, y_test):.4f}")</code></pre>

    <h3>৬. Feature Importance</h3>
    <pre><code">import pandas as pd

imp_df = pd.DataFrame({
    'Feature':    data.feature_names,
    'Importance': model.feature_importances_,
}).sort_values('Importance', ascending=False)

print(imp_df)

# তিন ধরনের importance:
xgb.plot_importance(model, importance_type='weight')   # কতবার split হলো
xgb.plot_importance(model, importance_type='gain')     # গড় gain (সবচেয়ে ভালো)
xgb.plot_importance(model, importance_type='cover')    # কতটুকু ডেটা cover করে</code></pre>

    <h3>৭. তিনটি মডেলের তুলনা</h3>
    <pre><code">from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=100, random_state=42),
}

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for name, m in models.items():
    if name == 'Linear Regression':
        pipe = Pipeline([('scaler', StandardScaler()), ('m', m)])
    else:
        pipe = Pipeline([('m', m)])
    pipe.fit(X_train, y_train)
    r2 = pipe.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
    print(f"{name:20s}: R²={r2:.4f}, RMSE={rmse:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Boosting</td><td>sequential tree — প্রতিটি tree আগেরটার ভুল শোধরায়</td></tr>
        <tr><td>learning_rate</td><td>ছোট হলে বেশি tree দরকার কিন্তু ভালো generalization</td></tr>
        <tr><td>Early Stopping</td><td>Overfitting এড়াতে validation score দেখে থামো</td></tr>
        <tr><td>Feature Importance</td><td>gain-based importance সবচেয়ে নির্ভরযোগ্য</td></tr>
        <tr><td>Scaling দরকার নেই</td><td>tree-based model — feature scale-independent</td></tr>
        <tr><td>কখন XGBoost?</td><td>tabular data-তে প্রায় সবসময় Random Forest বা LR-এর চেয়ে ভালো</td></tr>
      </tbody>
    </table>
  `,
};
