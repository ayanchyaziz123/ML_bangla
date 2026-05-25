export const hyperparam_1_basics = {
  slug: 'hyperparam-1-basics',
  title: 'হাইপারপ্যারামিটার: মডেলের গোপন নিয়ন্ত্রণ',
  description: 'প্যারামিটার বনাম হাইপারপ্যারামিটার, কেন টিউনিং গুরুত্বপূর্ণ, সাধারণ হাইপারপ্যারামিটার এবং ম্যানুয়াল টিউনিংয়ের সমস্যাগুলো।',
  date: 'মে ২০২৫',
  category: 'হাইপারপ্যারামিটার অপটিমাইজেশন',
  readTime: 10,
  content: `
    <h3>১. প্যারামিটার বনাম হাইপারপ্যারামিটার</h3>
    <p>
      মেশিন লার্নিং মডেল তৈরিতে দুটি মূল ধরনের সংখ্যামান থাকে: <strong>প্যারামিটার</strong> এবং <strong>হাইপারপ্যারামিটার</strong>।
      পার্থক্যটা বোঝা অত্যন্ত জরুরি, কারণ এই দুটি সম্পূর্ণ আলাদা উপায়ে নির্ধারিত হয়।
    </p>
    <p>
      <strong>প্যারামিটার</strong> হলো মডেলের অভ্যন্তরীণ মান, যা ট্রেনিং ডেটা থেকে স্বয়ংক্রিয়ভাবে শেখা হয়। যেমন —
      Linear Regression-এ weights ও bias, Neural Network-এ প্রতিটি layer-এর weights। আপনি এগুলো নিজে সেট করেন না;
      gradient descent বা অন্য optimization algorithm এগুলো নির্ধারণ করে।
    </p>
    <p>
      <strong>হাইপারপ্যারামিটার</strong> হলো মডেল ট্রেনিং শুরুর আগে আপনাকে নিজে সেট করতে হয় এমন মান। যেমন —
      learning rate, decision tree-র depth, SVM-এর C ও gamma। এগুলো ডেটা থেকে শেখা যায় না;
      আপনাকেই এগুলো বেছে দিতে হয়।
    </p>
    <table>
      <thead>
        <tr>
          <th>বৈশিষ্ট্য</th>
          <th>প্যারামিটার</th>
          <th>হাইপারপ্যারামিটার</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>কে নির্ধারণ করে?</td>
          <td>Training algorithm (স্বয়ংক্রিয়)</td>
          <td>Data scientist (ম্যানুয়ালি)</td>
        </tr>
        <tr>
          <td>কখন নির্ধারিত হয়?</td>
          <td>Training-এর সময়</td>
          <td>Training শুরুর আগে</td>
        </tr>
        <tr>
          <td>উদাহরণ</td>
          <td>weights, bias, coefficients</td>
          <td>learning rate, depth, C, gamma</td>
        </tr>
        <tr>
          <td>সংরক্ষিত হয়?</td>
          <td>Model-এর সাথে সেভ হয়</td>
          <td>আলাদাভাবে রেকর্ড করতে হয়</td>
        </tr>
      </tbody>
    </table>

    <h3>২. কেন হাইপারপ্যারামিটার টিউনিং গুরুত্বপূর্ণ?</h3>
    <p>
      একটি সাধারণ উদাহরণ দেখা যাক। Random Forest classifier-এ <code>n_estimators</code> (tree সংখ্যা) এবং
      <code>max_depth</code> (গভীরতা) দুটি গুরুত্বপূর্ণ হাইপারপ্যারামিটার।
    </p>
    <pre><code>from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=10, random_state=42)

# Default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
score_default = cross_val_score(rf_default, X, y, cv=5).mean()

# Tuned hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)
score_tuned = cross_val_score(rf_tuned, X, y, cv=5).mean()

print(f"Default CV Score:  {score_default:.4f}")
print(f"Tuned CV Score:    {score_tuned:.4f}")
print(f"Improvement:       {(score_tuned - score_default)*100:.2f}%")</code></pre>
    <p>
      একই অ্যালগরিদম ব্যবহার করেও ভালো হাইপারপ্যারামিটার নির্বাচন মডেলের accuracy উল্লেখযোগ্যভাবে বাড়াতে পারে।
      বাস্তব প্রজেক্টে ২–১০% accuracy improvement প্রায়ই দেখা যায়, যা production system-এ বিশাল পার্থক্য ফেলতে পারে।
    </p>

    <h3>৩. সাধারণ হাইপারপ্যারামিটার ও তাদের প্রভাব</h3>
    <h4>৩.১ Learning Rate (শেখার হার)</h4>
    <p>
      Gradient descent-ভিত্তিক মডেলে (Neural Network, Gradient Boosting) learning rate সবচেয়ে গুরুত্বপূর্ণ হাইপারপ্যারামিটার।
      এটি প্রতিটি step-এ weights কতটুকু আপডেট হবে তা নিয়ন্ত্রণ করে।
    </p>
    <p>
      <strong>বড় learning rate:</strong> দ্রুত converge করে, কিন্তু optimal solution miss করতে পারে বা oscillate করতে পারে।<br/>
      <strong>ছোট learning rate:</strong> ধীরে ধীরে converge করে, অনেক epoch লাগে, কিন্তু ভালো minimum খুঁজে পায়।<br/>
      <strong>Typical range:</strong> 0.0001 থেকে 0.1 পর্যন্ত।
    </p>
    <pre><code>from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

learning_rates = [0.001, 0.01, 0.1, 0.5]
for lr in learning_rates:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        learning_rate_init=lr,
        max_iter=500,
        random_state=42
    )
    score = cross_val_score(mlp, X_scaled, y, cv=3).mean()
    print(f"Learning Rate {lr:.3f} -> CV Score: {score:.4f}")</code></pre>

    <h4>৩.২ Decision Tree: max_depth এবং min_samples_split</h4>
    <p>
      <strong>max_depth:</strong> Tree কতটা গভীর হবে। বেশি গভীর হলে overfitting, কম গভীর হলে underfitting।<br/>
      <strong>min_samples_split:</strong> একটি node split করতে ন্যূনতম কতটি sample দরকার। বড় মান overfitting কমায়।
    </p>
    <pre><code>from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

depths = range(1, 20)
train_scores = []
cv_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X, y)
    train_scores.append(dt.score(X, y))
    cv_score = cross_val_score(dt, X, y, cv=5).mean()
    cv_scores.append(cv_score)

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'b-o', label='Train Score', markersize=5)
plt.plot(depths, cv_scores, 'r-s', label='CV Score', markersize=5)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Tradeoff: max_depth-এর প্রভাব')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_depth = depths[cv_scores.index(max(cv_scores))]
print(f"Best max_depth: {best_depth}, CV Score: {max(cv_scores):.4f}")</code></pre>

    <h4>৩.৩ SVM: C এবং gamma</h4>
    <p>
      SVM (Support Vector Machine)-এ দুটি গুরুত্বপূর্ণ হাইপারপ্যারামিটার:
    </p>
    <p>
      <strong>C (Regularization parameter):</strong> Classification error-এর penalty। বড় C মানে কম regularization
      (overfitting সম্ভব), ছোট C মানে বেশি regularization (underfitting সম্ভব)।<br/>
      <strong>gamma (RBF kernel-এ):</strong> একটি training example-এর influence কতদূর পৌঁছাবে। বড় gamma মানে
      কাছের points-এর উপর বেশি focus (overfitting), ছোট gamma মানে দূরের points-ও গণ্য হয়।
    </p>
    <pre><code>from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = ['scale', 0.001, 0.01, 0.1, 1]

print("C vs gamma CV Scores:")
print(f"{'C':>8}", end='')
for g in gamma_values:
    print(f"  gamma={g}", end='')
print()

for C in C_values:
    print(f"C={C:>5}", end='')
    for gamma in gamma_values:
        svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
        score = cross_val_score(svm, X_scaled, y, cv=3).mean()
        print(f"  {score:.3f}     ", end='')
    print()</code></pre>

    <h3>৪. ম্যানুয়াল টিউনিংয়ের সমস্যা</h3>
    <p>
      অনেক শুরুর দিকের ML practitioners হাইপারপ্যারামিটার ম্যানুয়ালি টিউন করেন — অর্থাৎ একটি মান ব্যবহার করে
      দেখেন, তারপর আরেকটি ব্যবহার করেন। এই পদ্ধতিতে বেশ কিছু মৌলিক সমস্যা আছে:
    </p>
    <p>
      <strong>সমস্যা ১ — Test set-এ overfitting:</strong> যদি আপনি বারবার test set-এ evaluate করে হাইপারপ্যারামিটার
      বেছে নেন, তাহলে আসলে test set-এ overfit করছেন। Model নতুন ডেটায় ভালো করবে না।
    </p>
    <p>
      <strong>সমস্যা ২ — Human bias:</strong> মানুষ intuition দিয়ে parameter space ভালোভাবে explore করতে পারে না।
      আমরা প্রায়ই round numbers বেছে নিই (যেমন 0.1, 1, 10) এবং অনেক ভালো combination miss করে যাই।
    </p>
    <p>
      <strong>সমস্যা ৩ — Interaction effects:</strong> একটি হাইপারপ্যারামিটারের optimal value অন্যটির উপর নির্ভর করে।
      ম্যানুয়াল টিউনিংয়ে এই interaction ধরা মুশকিল।
    </p>
    <p>
      <strong>সমস্যা ৪ — পুনরুৎপাদনযোগ্যতা নেই:</strong> আপনার সিদ্ধান্তের কোনো systematic record থাকে না।
    </p>

    <h3>৫. Validation Set বনাম Test Set</h3>
    <p>
      হাইপারপ্যারামিটার টিউনিংয়ের জন্য ডেটাকে তিন ভাগে ভাগ করতে হয়:
    </p>
    <pre><code>from sklearn.model_selection import train_test_split

# মোট ডেটার 60% train, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    random_state=42, stratify=y_train_val
)

print(f"Train size:      {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test size:       {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")

# Validation set দিয়ে hyperparameter select করুন
best_score = 0
best_depth = None

for depth in range(1, 15):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    val_score = dt.score(X_val, y_val)
    if val_score > best_score:
        best_score = val_score
        best_depth = depth

# শুধুমাত্র FINAL evaluation-এ test set ব্যবহার করুন
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train_val, y_train_val)
test_score = final_model.score(X_test, y_test)

print(f"Best depth (from validation): {best_depth}")
print(f"Validation score:             {best_score:.4f}")
print(f"Final test score:             {test_score:.4f}")</code></pre>

    <h3>৬. Nested Cross-Validation</h3>
    <p>
      ডেটা কম থাকলে train/val/test split-এ প্রতিটি set খুব ছোট হয়ে যায়। সমাধান হলো
      <strong>Nested Cross-Validation</strong>। এতে দুটি cross-validation loop থাকে:
    </p>
    <p>
      <strong>Outer loop:</strong> Generalization error estimate করার জন্য — সত্যিকারের performance দেখায়।<br/>
      <strong>Inner loop:</strong> Hyperparameter select করার জন্য — validation হিসেবে কাজ করে।
    </p>
    <pre><code>from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Inner cross-validation: hyperparameter selection
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}

inner_cv = 5  # inner loop: 5-fold
outer_cv = 5  # outer loop: 5-fold

dt = DecisionTreeClassifier(random_state=42)

# GridSearchCV = inner loop
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

# cross_val_score = outer loop
nested_scores = cross_val_score(
    grid_search, X, y,
    cv=outer_cv,
    scoring='accuracy'
)

print(f"Nested CV scores:  {nested_scores}")
print(f"Mean score:        {nested_scores.mean():.4f}")
print(f"Std deviation:     {nested_scores.std():.4f}")
print()
print("এই score-ই সত্যিকারের generalization performance-এর unbiased estimate।")</code></pre>
    <p>
      Nested CV-তে পাওয়া score সবচেয়ে নির্ভরযোগ্য, কারণ hyperparameter selection এবং performance
      estimation সম্পূর্ণ আলাদা রাখা হয়। এটি বিশেষভাবে গুরুত্বপূর্ণ যখন আপনি academic paper-এ বা
      production deployment-এ unbiased performance estimate দিতে চান।
    </p>

    <h3>৭. সারসংক্ষেপ</h3>
    <p>
      হাইপারপ্যারামিটার অপটিমাইজেশন ML pipeline-এর একটি অপরিহার্য অংশ। মূল বিষয়গুলো মনে রাখুন:
    </p>
    <p>
      প্যারামিটার ট্রেনিং থেকে শেখা হয়, হাইপারপ্যারামিটার আপনাকে নির্ধারণ করতে হয়।
      ম্যানুয়াল টিউনিং অকার্যকর এবং biased।
      Test set শুধুমাত্র final evaluation-এর জন্য রাখুন।
      ছোট ডেটাসেটে nested CV ব্যবহার করুন।
      পরবর্তী পর্বে দেখব Grid Search ও Random Search — systematic hyperparameter tuning-এর প্রথম পদক্ষেপ।
    </p>
  `
};
