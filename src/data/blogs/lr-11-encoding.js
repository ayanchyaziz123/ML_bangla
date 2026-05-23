export const lr_11_encoding = {
  title: "Feature Encoding: Categorical ডেটাকে সংখ্যায় রূপান্তর",
  description: "Label Encoding, One-Hot Encoding, Ordinal Encoding, Target Encoding — কোনটা কখন ব্যবহার করতে হয় এবং লিনিয়ার রিগ্রেশনে কেন encoding ছাড়া চলে না।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 11,
  slug: "lr-feature-encoding",
  content: `
    <h3>১. Encoding কেন দরকার?</h3>
    <p>লিনিয়ার রিগ্রেশন (এবং বেশিরভাগ ML মডেল) শুধু <strong>সংখ্যা বুঝতে পারে</strong>। কিন্তু বাস্তব ডেটায় অনেক column থাকে যেগুলো text বা category:</p>
    <ul>
      <li>শহর: "ঢাকা", "চট্টগ্রাম", "সিলেট"</li>
      <li>শিক্ষা: "SSC", "HSC", "স্নাতক", "স্নাতকোত্তর"</li>
      <li>রঙ: "লাল", "নীল", "সবুজ"</li>
    </ul>
    <p>এগুলোকে সংখ্যায় রূপান্তর করার প্রক্রিয়াকে বলে <strong>Encoding</strong>। ভুল encoding করলে মডেল ভুল সম্পর্ক শিখে।</p>

    <h3>২. Label Encoding</h3>
    <p>প্রতিটি unique value-কে একটি করে integer দেওয়া হয়।</p>
    <pre><code>from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({'শহর': ['ঢাকা', 'চট্টগ্রাম', 'সিলেট', 'ঢাকা', 'চট্টগ্রাম']})

le = LabelEncoder()
df['শহর_encoded'] = le.fit_transform(df['শহর'])
print(df)
#       শহর  শহর_encoded
# 0    ঢাকা            2
# 1  চট্টগ্রাম          0
# 2    সিলেট            1
# 3    ঢাকা            2
# 4  চট্টগ্রাম          0

print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))</code></pre>

    <p><strong>সমস্যা:</strong> মডেল ভাবে চট্টগ্রাম (0) &lt; সিলেট (1) &lt; ঢাকা (2) — কিন্তু শহরগুলোর মধ্যে এই ক্রম নেই! তাই nominal (কোনো ক্রম নেই) category-তে Label Encoding ব্যবহার করা <strong>ঠিক না</strong>।</p>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Label Encoding-এর সমস্যা</text>
        <rect x="20" y="30" width="100" height="28" rx="4" fill="#dbeafe"/>
        <text x="70" y="48" text-anchor="middle" font-size="10" fill="#1e40af">চট্টগ্রাম = 0</text>
        <rect x="190" y="30" width="100" height="28" rx="4" fill="#dbeafe"/>
        <text x="240" y="48" text-anchor="middle" font-size="10" fill="#1e40af">সিলেট = 1</text>
        <rect x="360" y="30" width="100" height="28" rx="4" fill="#dbeafe"/>
        <text x="410" y="48" text-anchor="middle" font-size="10" fill="#1e40af">ঢাকা = 2</text>
        <text x="155" y="47" text-anchor="middle" font-size="14" fill="#dc2626">&lt;</text>
        <text x="330" y="47" text-anchor="middle" font-size="14" fill="#dc2626">&lt;</text>
        <text x="240" y="88" text-anchor="middle" font-size="9" fill="#dc2626">মডেল মনে করে ঢাকা সংখ্যাগতভাবে সবচেয়ে "বড়" — যা ভুল!</text>
      </svg>
    </div>

    <h3>৩. One-Hot Encoding</h3>
    <p>প্রতিটি unique value-এর জন্য আলাদা binary column তৈরি করা হয়। Nominal category-র জন্য এটিই সঠিক পদ্ধতি।</p>
    <pre><code>import pandas as pd

df = pd.DataFrame({'শহর': ['ঢাকা', 'চট্টগ্রাম', 'সিলেট', 'ঢাকা']})

# pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['শহর'], drop_first=True)
# drop_first=True → Dummy Variable Trap এড়াতে একটি column বাদ দাও

print(df_encoded)
#    শহর_সিলেট  শহর_ঢাকা
# 0      False      True
# 1      False     False   ← চট্টগ্রাম (reference category)
# 2       True     False
# 3      False      True

# sklearn দিয়ে:
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False)
encoded = ohe.fit_transform(df[['শহর']])
print(ohe.get_feature_names_out())</code></pre>

    <p><strong>Dummy Variable Trap:</strong> যদি 3টি category থাকে এবং 3টি column রাখো, তাহলে তৃতীয় column = 1 − column1 − column2, যা perfect multicollinearity তৈরি করে। তাই <code>drop_first=True</code> ব্যবহার করো।</p>

    <h3>৪. Ordinal Encoding</h3>
    <p>যখন category-গুলোর মধ্যে একটি নির্দিষ্ট <strong>ক্রম</strong> আছে — তখন Ordinal Encoding সঠিক।</p>
    <pre><code>from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({'শিক্ষা': ['SSC', 'স্নাতক', 'HSC', 'স্নাতকোত্তর', 'SSC']})

# ক্রম নির্দিষ্ট করো (ছোট থেকে বড়)
categories = [['SSC', 'HSC', 'স্নাতক', 'স্নাতকোত্তর']]

oe = OrdinalEncoder(categories=categories)
df['শিক্ষা_encoded'] = oe.fit_transform(df[['শিক্ষা']])
print(df)
#           শিক্ষা  শিক্ষা_encoded
# 0           SSC             0.0
# 1        স্নাতক             2.0
# 2           HSC             1.0
# 3  স্নাতকোত্তর             3.0
# 4           SSC             0.0</code></pre>

    <h3>৫. Target Encoding</h3>
    <p>প্রতিটি category-কে সেই category-র target (y) এর গড় দিয়ে replace করা হয়। High cardinality (অনেক unique value) column-এর জন্য কার্যকর।</p>
    <pre><code>import pandas as pd
import numpy as np

df = pd.DataFrame({
    'শহর':   ['ঢাকা', 'চট্টগ্রাম', 'সিলেট', 'ঢাকা', 'সিলেট', 'চট্টগ্রাম'],
    'দাম':   [80,     55,           40,      90,     45,       60]
})

# Training-এ: প্রতিটি শহরের গড় দাম দিয়ে replace
target_mean = df.groupby('শহর')['দাম'].mean()
df['শহর_encoded'] = df['শহর'].map(target_mean)
print(df)
#         শহর  দাম  শহর_encoded
# 0      ঢাকা   80          85.0
# 1  চট্টগ্রাম   55          57.5
# 2    সিলেট   40          42.5
# 3      ঢাকা   90          85.0
# 4    সিলেট   45          42.5
# 5  চট্টগ্রাম   60          57.5

# ⚠ সতর্কতা: Target encoding-এ data leakage হতে পারে।
# Cross-validation-এ সঠিকভাবে ব্যবহার করো।</code></pre>

    <h3>৬. কোন Encoding কখন?</h3>
    <table>
      <thead><tr><th>Encoding</th><th>কখন ব্যবহার করবে</th><th>সতর্কতা</th></tr></thead>
      <tbody>
        <tr><td><strong>Label</strong></td><td>Tree-based model (Decision Tree, Random Forest) বা ordinal data</td><td>Linear model-এ nominal data-তে ব্যবহার করো না</td></tr>
        <tr><td><strong>One-Hot</strong></td><td>Nominal category, linear model, কম unique value</td><td>অনেক unique value থাকলে column অনেক বেড়ে যায়</td></tr>
        <tr><td><strong>Ordinal</strong></td><td>ক্রম আছে এমন category (low/medium/high)</td><td>ক্রম নিজে নির্দিষ্ট করতে হবে</td></tr>
        <tr><td><strong>Target</strong></td><td>High cardinality (৫০+ unique value)</td><td>Data leakage — pipeline-এ সাবধানে ব্যবহার করো</td></tr>
      </tbody>
    </table>

    <h3>৭. সম্পূর্ণ Pipeline উদাহরণ</h3>
    <pre><code>from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# ফিচার ভাগ করো
nominal_cols  = ['শহর', 'ধরন']          # One-Hot
ordinal_cols  = ['শিক্ষা']             # Ordinal
numeric_cols  = ['area', 'age']         # StandardScaler

preprocessor = ColumnTransformer([
    ('ohe',     OneHotEncoder(drop='first'),
                nominal_cols),
    ('ordinal', OrdinalEncoder(
                    categories=[['SSC','HSC','স্নাতক','স্নাতকোত্তর']]),
                ordinal_cols),
    ('scale',   StandardScaler(),
                numeric_cols),
])

pipeline = Pipeline([
    ('prep',  preprocessor),
    ('model', LinearRegression()),
])

pipeline.fit(X_train, y_train)
print("R²:", pipeline.score(X_test, y_test))</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>কেন encoding?</td><td>ML মডেল সংখ্যা ছাড়া কিছু বোঝে না</td></tr>
        <tr><td>Nominal → One-Hot</td><td>ক্রমহীন category-তে এটিই সঠিক</td></tr>
        <tr><td>Ordinal → Ordinal Encoding</td><td>ক্রম আছে — সেই ক্রম নিজে দাও</td></tr>
        <tr><td>Dummy Trap</td><td>drop_first=True দিয়ে multicollinearity এড়াও</td></tr>
        <tr><td>Target Encoding</td><td>High cardinality-তে কার্যকর, data leakage-এ সাবধান</td></tr>
      </tbody>
    </table>
  `,
};
