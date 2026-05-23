export const knn_3_regression = {
  title: "KNN Regression: Continuous Output Prediction",
  description: "KNeighborsRegressor দিয়ে continuous value prediction — California housing dataset, MSE, R² score এবং Linear Regression-এর সাথে তুলনা।",
  date: "২৩ মে, ২০২৬",
  category: "কে-নিকটতম প্রতিবেশী",
  readTime: 9,
  slug: "knn-regression",
  content: `
    <h3>১. KNN Regression কিভাবে কাজ করে?</h3>
    <p>
      KNN Regression-এ classification-এর মতো একই algorithm ব্যবহার হয়, শুধু output আলাদা।
      Classification-এ K প্রতিবেশীর majority class নেওয়া হয়, কিন্তু regression-এ K প্রতিবেশীর
      target value-এর <strong>mean (বা weighted mean)</strong> নেওয়া হয়।
    </p>
    <p>
      Prediction formula: ŷ = (1/K) × Σ yi, যেখানে yi হলো K নিকটতম প্রতিবেশীর target value।
    </p>
    <p>
      উদাহরণ: একটি বাড়ির দাম predict করতে হবে। KNN তার ৫ জন নিকটতম প্রতিবেশী বাড়ির
      দাম খোঁজে: [২০ লাখ, ২৫ লাখ, ২২ লাখ, ২৮ লাখ, ২৩ লাখ]। Prediction = mean = ২৩.৬ লাখ।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Manual KNN Regression বোঝার জন্য
# Training data: বাড়ির আয়তন (sqft) ও দাম (লাখ টাকা)
X_train = np.array([[500], [750], [1000], [1200], [1500], [1800], [2000]])
y_train = np.array([15, 20, 28, 33, 42, 55, 65])

# নতুন বাড়ি predict করতে হবে
new_house = np.array([[1100]])

# KNN Regressor (K=3)
knn_reg = KNeighborsRegressor(n_neighbors=3, weights='uniform')
knn_reg.fit(X_train, y_train)

# Prediction
pred_price = knn_reg.predict(new_house)
print(f"বাড়ির আয়তন: {new_house[0][0]} sqft")
print(f"Predicted দাম: {pred_price[0]:.2f} লাখ টাকা")

# K=3 নিকটতম প্রতিবেশী কারা?
distances, indices = knn_reg.kneighbors(new_house)
print(f"\nK=3 নিকটতম প্রতিবেশী:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. আয়তন={X_train[idx][0]} sqft, দাম={y_train[idx]} লাখ, দূরত্ব={dist:.1f}")
print(f"\nMean দাম: {np.mean(y_train[indices[0]]):.2f} লাখ")</code></pre>

    <h3>২. California Housing Dataset-এ KNN Regression</h3>
    <p>
      California housing dataset-এ ঘরের বিভিন্ন বৈশিষ্ট্য থেকে median house value predict করা হবে।
      এটি একটি real-world regression সমস্যার চমৎকার উদাহরণ:
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Dataset Load
housing = fetch_california_housing()
X = housing.data
y = housing.target

print("California Housing Dataset:")
print(f"  Features: {housing.feature_names}")
print(f"  Samples: {X.shape[0]}")
print(f"  Target range: \${y.min()*100000:.0f} - \${y.max()*100000:.0f}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalization (KNN-এর জন্য অত্যাবশ্যক)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Regressor Training
knn_reg = KNeighborsRegressor(
    n_neighbors=10,
    weights='distance',
    metric='euclidean',
    n_jobs=-1  # সব CPU core ব্যবহার
)
knn_reg.fit(X_train_scaled, y_train)

# Prediction ও Evaluation
y_pred = knn_reg.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance (K=10, weights='distance'):")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f} (\${rmse*100000:.0f})")
print(f"  MAE:  {mae:.4f} (\${mae*100000:.0f})")
print(f"  R²:   {r2:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual House Value ($100k)')
plt.ylabel('Predicted House Value ($100k)')
plt.title(f'KNN Regression: Actual vs Predicted (R²={r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৩. Weighted KNN Regression</h3>
    <p>
      <strong>weights='distance'</strong> ব্যবহার করলে কাছের প্রতিবেশীরা বেশি গুরুত্ব পায়।
      এটি সাধারণত regression-এ বেশি accurate কারণ কাছের data point-এর value বেশি relevant:
    </p>
    <pre><code>import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Uniform vs Distance weights তুলনা
results = {}

for weights in ['uniform', 'distance']:
    for k in [5, 10, 15, 20]:
        knn = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[(k, weights)] = {'MSE': mse, 'R2': r2}

# Results DataFrame তৈরি
rows = []
for (k, w), metrics in results.items():
    rows.append({'K': k, 'Weights': w, 'MSE': round(metrics['MSE'], 4),
                 'R2': round(metrics['R2'], 4)})

results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
print("KNN Regression Results:")
print(results_df.to_string(index=False))

# Residual Analysis
best_knn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)
residuals = y_test - y_pred_best

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.3, s=10, color='steelblue')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()</code></pre>

    <h3>৪. Optimal K নির্বাচন Regression-এ</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
mse_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    # neg_mean_squared_error কারণ sklearn maximize করে, আমরা minimize চাই
    scores = cross_val_score(knn, X_train_scaled, y_train,
                             cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(-scores.mean())  # positive MSE

best_k = list(k_range)[np.argmin(mse_scores)]

plt.figure(figsize=(10, 6))
plt.plot(k_range, mse_scores, 'b-o', linewidth=2, markersize=6)
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
            label=f'Best K={best_k}')
plt.xlabel('K মান')
plt.ylabel('Mean Squared Error (CV)')
plt.title('K vs MSE: KNN Regression-এ Optimal K খোঁজা')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Best K: {best_k}")
print(f"Best CV MSE: {mse_scores[best_k-1]:.4f}")</code></pre>

    <h3>৫. KNN Regression vs Linear Regression: তুলনামূলক বিশ্লেষণ</h3>
    <table>
      <thead>
        <tr>
          <th>বৈশিষ্ট্য</th>
          <th>KNN Regression</th>
          <th>Linear Regression</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Model Type</td>
          <td>Non-parametric</td>
          <td>Parametric</td>
        </tr>
        <tr>
          <td>Assumption</td>
          <td>কোনো assumption নেই</td>
          <td>Linear relationship ধরে নেয়</td>
        </tr>
        <tr>
          <td>Training Speed</td>
          <td>O(1) — অত্যন্ত দ্রুত</td>
          <td>O(n·d²) — সাধারণত দ্রুত</td>
        </tr>
        <tr>
          <td>Prediction Speed</td>
          <td>O(n·d) — ধীর</td>
          <td>O(d) — অত্যন্ত দ্রুত</td>
        </tr>
        <tr>
          <td>Memory</td>
          <td>O(n·d) — সব data রাখতে হয়</td>
          <td>O(d) — শুধু coefficients</td>
        </tr>
        <tr>
          <td>Non-linearity</td>
          <td>Handle করতে পারে</td>
          <td>পারে না (polynomial ছাড়া)</td>
        </tr>
        <tr>
          <td>Interpretability</td>
          <td>কম interpretable</td>
          <td>সহজে interpretable</td>
        </tr>
        <tr>
          <td>Outlier Sensitivity</td>
          <td>Distance weight দিলে কম</td>
          <td>বেশি sensitive</td>
        </tr>
      </tbody>
    </table>
  `,
};
