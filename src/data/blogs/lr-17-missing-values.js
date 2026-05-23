export const lr_17_missing_values = {
  title: "Missing Values সামলানো — ফাঁকা ডেটার সঙ্গে কী করবো?",
  description: "Missing data কেন হয়, SimpleImputer, KNNImputer, IterativeImputer দিয়ে imputation, এবং কখন row বাদ দেওয়া উচিত — বাংলায় সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-missing-values",
  content: `
    <h3>১. Missing Data কেন হয়?</h3>
    <p>বাস্তব ডেটায় missing values প্রায় সবসময় থাকে। কারণগুলো:</p>
    <ul>
      <li>সার্ভে ফর্মে কিছু প্রশ্ন উত্তর না দেওয়া</li>
      <li>সেন্সর বা যন্ত্রপাতি ত্রুটি</li>
      <li>ডেটা entry mistake বা system error</li>
      <li>কিছু তথ্য প্রযোজ্য নয় (যেমন independent person-এর spouse income)</li>
    </ul>
    <p>লিনিয়ার রিগ্রেশন missing value সরাসরি handle করতে পারে না — আগে পূরণ করতে হবে।</p>

    <h3>২. Missing Data বোঝা</h3>
    <pre><code>import pandas as pd
import numpy as np

data = {
    'বয়স':      [25, np.nan, 35, 40, np.nan, 30, 28],
    'বেতন':     [30000, 45000, np.nan, 60000, 35000, np.nan, 40000],
    'অভিজ্ঞতা': [2, 5, 8, np.nan, 3, 6, np.nan],
    'দাম':      [100, 150, 200, 250, 130, 180, 160],
}
df = pd.DataFrame(data)

# Missing value বিশ্লেষণ
print("Missing value সংখ্যা:")
print(df.isnull().sum())
print()
print("Missing value শতাংশ:")
print((df.isnull().sum() / len(df) * 100).round(1))</code></pre>

    <h3>৩. Missing Data-র তিন ধরন</h3>
    <table>
      <thead><tr><th>ধরন</th><th>অর্থ</th><th>উদাহরণ</th><th>সমাধান</th></tr></thead>
      <tbody>
        <tr><td><strong>MCAR</strong> (সম্পূর্ণ random)</td><td>missing হওয়া কোনো কিছুর উপর নির্ভর করে না</td><td>random sensor failure</td><td>যেকোনো imputation</td></tr>
        <tr><td><strong>MAR</strong> (অন্য column-এর উপর নির্ভর)</td><td>missing হওয়া অন্য observed data-র উপর নির্ভর করে</td><td>বেশি বয়স্ক মানুষ বেতন জানাতে চান না</td><td>Model-based imputation</td></tr>
        <tr><td><strong>MNAR</strong> (নিজের মানের উপর নির্ভর)</td><td>missing হওয়া সেই column-এর মানের উপর নির্ভর করে</td><td>বেশি বেতনের মানুষ বেতন লুকান</td><td>Domain knowledge দরকার</td></tr>
      </tbody>
    </table>

    <h3>৪. SimpleImputer — সহজ পদ্ধতি</h3>
    <pre><code>from sklearn.impute import SimpleImputer

# সংখ্যাত্মক ফিচার: mean বা median দিয়ে পূরণ
imp_mean   = SimpleImputer(strategy='mean')
imp_median = SimpleImputer(strategy='median')

X = df[['বয়স', 'বেতন', 'অভিজ্ঞতা']]

X_mean   = imp_mean.fit_transform(X)
X_median = imp_median.fit_transform(X)

print("Mean imputation:")
print(pd.DataFrame(X_mean, columns=X.columns).round(1))

# Categorical ফিচার: most_frequent দিয়ে
imp_mode = SimpleImputer(strategy='most_frequent')

# Constant দিয়ে পূরণ (যখন missing = আলাদা অর্থ):
imp_zero = SimpleImputer(strategy='constant', fill_value=0)</code></pre>

    <h3>৫. KNNImputer — প্রতিবেশী দিয়ে পূরণ</h3>
    <p>K-Nearest Neighbors ব্যবহার করে — কাছাকাছি row-গুলোর গড় দিয়ে missing value পূরণ করে।</p>
    <pre><code>from sklearn.impute import KNNImputer

knn_imp = KNNImputer(n_neighbors=3)
X_knn   = knn_imp.fit_transform(X)

print("KNN imputation:")
print(pd.DataFrame(X_knn, columns=X.columns).round(1))

# KNN-এর সুবিধা: feature-এর সম্পর্ক বিবেচনা করে
# অসুবিধা: বড় ডেটায় ধীর</code></pre>

    <h3>৬. IterativeImputer — সবচেয়ে স্মার্ট পদ্ধতি</h3>
    <p>প্রতিটি missing feature-কে অন্য feature দিয়ে predict করে — পর্যায়ক্রমে যতক্ষণ না converge করে।</p>
    <pre><code>from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

iter_imp = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42
)
X_iter = iter_imp.fit_transform(X)

print("Iterative imputation:")
print(pd.DataFrame(X_iter, columns=X.columns).round(1))</code></pre>

    <h3>৭. Missing Indicator — missing ছিল সেই তথ্য রাখো</h3>
    <pre><code>from sklearn.impute import MissingIndicator

# missing ছিল কিনা সেটি নতুন binary column হিসেবে রাখো
indicator = MissingIndicator()
missing_mask = indicator.fit_transform(X)

df_indicator = pd.DataFrame(
    missing_mask,
    columns=[f"{col}_missing" for col in X.columns[indicator.features_]]
)
print(df_indicator)

# Pipeline-এ একসাথে ব্যবহার:
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
X_processed = pipe.fit_transform(X)</code></pre>

    <h3>৮. কখন Row বাদ দেওয়া ঠিক?</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>করণীয়</th></tr></thead>
      <tbody>
        <tr><td>missing &lt; 5% এবং MCAR</td><td>row বাদ দাও (listwise deletion)</td></tr>
        <tr><td>missing 5–30%</td><td>imputation ব্যবহার করো</td></tr>
        <tr><td>missing &gt; 30%</td><td>সেই column বাদ দেওয়ার কথা ভাবো</td></tr>
        <tr><td>target column (y) missing</td><td>সেই row সবসময় বাদ দাও</td></tr>
        <tr><td>MNAR</td><td>domain expert-এর সাহায্য নাও</td></tr>
      </tbody>
    </table>
    <pre><code># Missing percentage বেশি column বাদ দেওয়া
threshold = 0.30  # 30% এর বেশি missing → বাদ
missing_frac = df.isnull().mean()
cols_to_drop = missing_frac[missing_frac > threshold].index
df_dropped = df.drop(columns=cols_to_drop)
print(f"বাদ দেওয়া column: {list(cols_to_drop)}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>পদ্ধতি</th><th>কখন</th><th>সুবিধা</th></tr></thead>
      <tbody>
        <tr><td>Mean/Median</td><td>সহজ ক্ষেত্রে, outlier নেই</td><td>দ্রুত, সহজ</td></tr>
        <tr><td>KNNImputer</td><td>feature সম্পর্ক বিবেচনা করতে</td><td>accurate কিন্তু ধীর</td></tr>
        <tr><td>IterativeImputer</td><td>সবচেয়ে সঠিক ফলাফল চাইলে</td><td>সেরা accuracy</td></tr>
        <tr><td>Missing Indicator</td><td>missing হওয়াটাই তথ্য হলে</td><td>তথ্য হারায় না</td></tr>
        <tr><td>Row/Column drop</td><td>খুব কম বা খুব বেশি missing</td><td>সহজ কিন্তু তথ্য নষ্ট</td></tr>
      </tbody>
    </table>
  `,
};
