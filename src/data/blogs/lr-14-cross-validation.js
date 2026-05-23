export const lr_14_cross_validation = {
  title: "Train/Test Split ও Cross-Validation — মডেল সঠিকভাবে মূল্যায়ন করো",
  description: "Train/Test split কেন যথেষ্ট নয়, k-fold Cross-Validation কীভাবে কাজ করে, এবং Overfitting vs Underfitting ধরার সঠিক পদ্ধতি — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 11,
  slug: "lr-cross-validation",
  content: `
    <h3>১. সমস্যাটা কী?</h3>
    <p>ধরো তুমি মডেল বানালে এবং একই ডেটায় test করলে — মডেল R² = 0.98 পেলো। কিন্তু নতুন ডেটায় R² = 0.45! এটাকে বলে <strong>Overfitting</strong> — মডেল training ডেটা মুখস্থ করে ফেলেছে কিন্তু নতুন ডেটায় সাধারণীকরণ করতে পারছে না।</p>
    <p>এই সমস্যা ধরার জন্য দরকার সঠিক মূল্যায়ন পদ্ধতি।</p>

    <h3>২. Train/Test Split</h3>
    <p>ডেটাকে দুটো ভাগে ভাগ করো — একটি দিয়ে মডেল train করো, অন্যটি দিয়ে test করো।</p>
    <pre><code>from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ডেটা তৈরি
np.random.seed(42)
X = np.random.rand(200, 3)
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(200)*0.5

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2  = r2_score(y_test,  model.predict(X_test))

print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")

# Train R² >> Test R² → Overfitting
# Train R² ≈ Test R² কিন্তু দুটোই কম → Underfitting</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Train/Test Split (80/20)</text>
        <rect x="20" y="35" width="352" height="40" rx="4" fill="#dbeafe"/>
        <text x="196" y="60" text-anchor="middle" font-size="10" fill="#1e40af">Training Set (80%) — মডেল শেখে</text>
        <rect x="380" y="35" width="80" height="40" rx="4" fill="#fde68a"/>
        <text x="420" y="60" text-anchor="middle" font-size="10" fill="#92400e">Test (20%)</text>
        <text x="240" y="95" text-anchor="middle" font-size="9" fill="#6b7280">Test set একবারও দেখানো যাবে না — শুধু চূড়ান্ত মূল্যায়নে ব্যবহার করো</text>
      </svg>
    </div>

    <h3>৩. Train/Test Split-এর সমস্যা</h3>
    <p>মাত্র একটি split করলে ফলাফল <strong>lucky বা unlucky</strong> হতে পারে — কোন ডেটা train-এ গেল তার উপর নির্ভর করে। ছোট ডেটাসেটে এই সমস্যা বেশি।</p>

    <h3>৪. K-Fold Cross-Validation</h3>
    <p>ডেটাকে k ভাগে ভাগ করো। k বার model train করো — প্রতিবার একটি ভিন্ন fold test হিসেবে ব্যবহার হয়। সব k বারের স্কোরের গড় নাও।</p>
    <pre><code>from sklearn.model_selection import cross_val_score, KFold

model = LinearRegression()

# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("প্রতিটি fold-এর R²:", scores.round(4))
print(f"গড় R²: {scores.mean():.4f} ± {scores.std():.4f}")

# MSE দিয়েও করা যায়:
mse_scores = cross_val_score(
    model, X, y, cv=5,
    scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-mse_scores)
print(f"গড় RMSE: {rmse_scores.mean():.4f}")</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 135" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">5-Fold Cross-Validation</text>
        <!-- Row labels -->
        <text x="30" y="38" font-size="9" fill="#6b7280">Fold 1</text>
        <text x="30" y="58" font-size="9" fill="#6b7280">Fold 2</text>
        <text x="30" y="78" font-size="9" fill="#6b7280">Fold 3</text>
        <text x="30" y="98" font-size="9" fill="#6b7280">Fold 4</text>
        <text x="30" y="118" font-size="9" fill="#6b7280">Fold 5</text>
        <!-- Fold 1 -->
        <rect x="65" y="27" width="60" height="14" rx="2" fill="#fde68a"/>
        <rect x="130" y="27" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="195" y="27" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="260" y="27" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="325" y="27" width="60" height="14" rx="2" fill="#dbeafe"/>
        <!-- Fold 2 -->
        <rect x="65" y="47" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="130" y="47" width="60" height="14" rx="2" fill="#fde68a"/>
        <rect x="195" y="47" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="260" y="47" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="325" y="47" width="60" height="14" rx="2" fill="#dbeafe"/>
        <!-- Fold 3 -->
        <rect x="65" y="67" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="130" y="67" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="195" y="67" width="60" height="14" rx="2" fill="#fde68a"/>
        <rect x="260" y="67" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="325" y="67" width="60" height="14" rx="2" fill="#dbeafe"/>
        <!-- Fold 4 -->
        <rect x="65" y="87" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="130" y="87" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="195" y="87" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="260" y="87" width="60" height="14" rx="2" fill="#fde68a"/>
        <rect x="325" y="87" width="60" height="14" rx="2" fill="#dbeafe"/>
        <!-- Fold 5 -->
        <rect x="65" y="107" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="130" y="107" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="195" y="107" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="260" y="107" width="60" height="14" rx="2" fill="#dbeafe"/>
        <rect x="325" y="107" width="60" height="14" rx="2" fill="#fde68a"/>
        <rect x="400" y="30" width="12" height="12" rx="2" fill="#fde68a"/>
        <text x="415" y="41" font-size="8" fill="#6b7280">Test</text>
        <rect x="400" y="48" width="12" height="12" rx="2" fill="#dbeafe"/>
        <text x="415" y="59" font-size="8" fill="#6b7280">Train</text>
      </svg>
    </div>

    <h3>৫. Overfitting vs Underfitting চেনা</h3>
    <pre><code>from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

results = []
for degree in [1, 2, 5, 10]:
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=degree)),
        ('model', LinearRegression()),
    ])
    train_score = cross_val_score(pipe, X[:,0:1], y, cv=5,
                                  scoring='r2').mean()
    # train score আলাদাভাবে:
    pipe.fit(X_train[:,0:1], y_train)
    tr = r2_score(y_train, pipe.predict(X_train[:,0:1]))
    results.append((degree, tr, train_score))
    print(f"Degree {degree:2d}: Train={tr:.3f}, CV={train_score:.3f}")

# Degree 1  → Train=0.75, CV=0.74 (Underfitting — দুটোই কম)
# Degree 2  → Train=0.91, CV=0.90 (ভালো — দুটো কাছাকাছি)
# Degree 10 → Train=0.99, CV=0.45 (Overfitting — train >> CV)</code></pre>

    <h3>৬. Learning Curve দিয়ে Bias-Variance নির্ণয়</h3>
    <pre><code>from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1),   label='Validation')
plt.xlabel('Training size')
plt.ylabel('R²')
plt.legend()
plt.title('Learning Curve')
plt.show()

# Train >> Val  → Overfitting (High Variance)
# উভয়ই কম     → Underfitting (High Bias)</code></pre>

    <h3>৭. কখন কোন CV ব্যবহার করবে?</h3>
    <table>
      <thead><tr><th>পদ্ধতি</th><th>কখন</th><th>k-এর মান</th></tr></thead>
      <tbody>
        <tr><td>K-Fold (k=5)</td><td>সাধারণ ক্ষেত্রে</td><td>5 (সবচেয়ে জনপ্রিয়)</td></tr>
        <tr><td>K-Fold (k=10)</td><td>সামান্য বেশি নির্ভরযোগ্যতা দরকার</td><td>10</td></tr>
        <tr><td>Leave-One-Out (LOO)</td><td>খুব ছোট ডেটাসেট (n &lt; 50)</td><td>n (সব)</td></tr>
        <tr><td>Stratified K-Fold</td><td>Classification, imbalanced class</td><td>5–10</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Train/Test Split</td><td>দ্রুত কিন্তু একটি random split — নির্ভরযোগ্যতা কম</td></tr>
        <tr><td>K-Fold CV</td><td>k বার train/test — গড় স্কোর অনেক বেশি নির্ভরযোগ্য</td></tr>
        <tr><td>Overfitting</td><td>Train R² বেশি, CV R² কম → মডেল বেশি জটিল</td></tr>
        <tr><td>Underfitting</td><td>দুটোই কম → মডেল বেশি সরল</td></tr>
        <tr><td>Learning Curve</td><td>bias-variance সমস্যা ভিজুয়ালি দেখতে</td></tr>
      </tbody>
    </table>
  `,
};
