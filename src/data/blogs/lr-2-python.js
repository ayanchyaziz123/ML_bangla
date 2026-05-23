export const lr_2_python = {
  title: "Python-এ লিনিয়ার রিগ্রেশন — সম্পূর্ণ কোড গাইড",
  description: "Scikit-learn দিয়ে লিনিয়ার রিগ্রেশন ইমপ্লিমেন্ট করো — ডেটা প্রস্তুতি, মডেল ট্রেনিং, ভবিষ্যদ্বাণী এবং মূল্যায়ন পর্যন্ত সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 9,
  slug: "lr-python",
  content: `
    <h3>১. প্রয়োজনীয় লাইব্রেরি</h3>
    <pre><code>import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler</code></pre>

    <h3>২. ডেটা তৈরি ও দেখা</h3>
    <pre><code># আমাদের ছোট ডেটাসেট
data = {
    'study_hours': [2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'marks':       [35, 55, 60, 70, 72, 80, 85, 90, 92, 98]
}
df = pd.DataFrame(data)
print(df.head())

# ডেটা দেখো
print(f"মোট ডেটা পয়েন্ট: {len(df)}")
print(f"গড় নম্বর: {df['marks'].mean():.1f}")
print(df.describe())</code></pre>

    <h3>৩. ফিচার এবং টার্গেট আলাদা করা</h3>
    <pre><code>X = df[['study_hours']]  # 2D array দরকার sklearn-এর জন্য
y = df['marks']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Training set: {X_train.shape[0]} টি")
print(f"Test set:     {X_test.shape[0]} টি")</code></pre>

    <h3>৪. মডেল ট্রেনিং</h3>
    <pre><code>model = LinearRegression()
model.fit(X_train, y_train)

# মডেলের শেখা প্যারামিটার দেখো
print(f"Slope (m):     {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")
print(f"সমীকরণ: marks = {model.coef_[0]:.2f} × hours + {model.intercept_:.2f}")</code></pre>

    <h3>৫. ভবিষ্যদ্বাণী এবং মূল্যায়ন</h3>
    <pre><code>y_pred = model.predict(X_test)

# মেট্রিক্স
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# নতুন ডেটায় ভবিষ্যদ্বাণী
new_hours = np.array([[7.5]])
predicted = model.predict(new_hours)
print(f"৭.৫ ঘণ্টা পড়লে নম্বর হবে: {predicted[0]:.1f}")</code></pre>

    <h3>৬. রেজাল্ট তুলনা</h3>
    <pre><code># প্রকৃত vs ভবিষ্যদ্বাণী
results = pd.DataFrame({
    'actual':     y_test.values,
    'predicted':  y_pred.round(1),
    'diff':       (y_test.values - y_pred).round(1)
})
print(results)</code></pre>

    <h3>৭. মাল্টিপল লিনিয়ার রিগ্রেশন</h3>
    <pre><code># একাধিক ফিচার দিয়ে
data_multi = {
    'study_hours':    [2, 4, 6, 8, 10, 3, 5, 7, 9, 11],
    'sleep_hours':    [9, 7, 6, 8, 5,  8, 6, 7, 5, 6],
    'attendance_pct': [60, 70, 80, 90, 95, 65, 75, 85, 90, 95],
    'marks':          [40, 55, 68, 82, 92, 45, 60, 75, 88, 95]
}
df_multi = pd.DataFrame(data_multi)

X_multi = df_multi[['study_hours', 'sleep_hours', 'attendance_pct']]
y_multi = df_multi['marks']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# প্রতিটি ফিচারের গুরুত্ব
for feature, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"{feature}: {coef:.2f}")</code></pre>

    <h3>৮. সম্পূর্ণ পাইপলাইন</h3>
    <pre><code>from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),          # ফিচার স্কেল করা
    ('model',  LinearRegression())         # মডেল
])

pipe.fit(X_train_m, y_train_m)
score = pipe.score(X_test_m, y_test_m)
print(f"Pipeline R²: {score:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>ধাপ</th><th>কোড</th></tr></thead>
      <tbody>
        <tr><td>ডেটা ভাগ</td><td>train_test_split(X, y, test_size=0.2)</td></tr>
        <tr><td>মডেল তৈরি</td><td>LinearRegression()</td></tr>
        <tr><td>ট্রেনিং</td><td>model.fit(X_train, y_train)</td></tr>
        <tr><td>ভবিষ্যদ্বাণী</td><td>model.predict(X_test)</td></tr>
        <tr><td>মূল্যায়ন</td><td>r2_score, mean_squared_error</td></tr>
        <tr><td>Slope/Intercept</td><td>model.coef_, model.intercept_</td></tr>
      </tbody>
    </table>
  `,
};
