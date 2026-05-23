export const lr_5_interview = {
  title: "লিনিয়ার রিগ্রেশন ইন্টারভিউ প্রশ্ন ও উত্তর",
  description: "লিনিয়ার রিগ্রেশন সম্পর্কে সবচেয়ে বেশি জিজ্ঞেস করা ইন্টারভিউ প্রশ্ন এবং তাদের সহজ উত্তর — ML ইন্টারভিউ প্রস্তুতির জন্য।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 12,
  slug: "lr-interview-questions",
  content: `
    <h3>প্রশ্ন ১: লিনিয়ার রিগ্রেশন এবং লজিস্টিক রিগ্রেশনের পার্থক্য কী?</h3>
    <table>
      <thead><tr><th></th><th>লিনিয়ার রিগ্রেশন</th><th>লজিস্টিক রিগ্রেশন</th></tr></thead>
      <tbody>
        <tr><td><strong>আউটপুট</strong></td><td>যেকোনো সংখ্যা (continuous)</td><td>0 থেকে 1 (probability)</td></tr>
        <tr><td><strong>ব্যবহার</strong></td><td>দাম, তাপমাত্রা ভবিষ্যদ্বাণী</td><td>স্প্যাম/না স্প্যাম, রোগ আছে/নেই</td></tr>
        <tr><td><strong>Loss Function</strong></td><td>MSE (Mean Squared Error)</td><td>Binary Cross-Entropy</td></tr>
        <tr><td><strong>Activation</strong></td><td>কোনো activation নেই</td><td>Sigmoid function</td></tr>
      </tbody>
    </table>

    <h3>প্রশ্ন ২: লিনিয়ার রিগ্রেশনের assumptions কী কী?</h3>
    <p>লিনিয়ার রিগ্রেশন ভালো কাজ করতে ৫টি assumption থাকা দরকার:</p>
    <table>
      <thead><tr><th>#</th><th>Assumption</th><th>কীভাবে চেক করবে</th></tr></thead>
      <tbody>
        <tr><td>1</td><td><strong>Linearity:</strong> X এবং Y-এর মধ্যে সরল সম্পর্ক</td><td>Scatter plot</td></tr>
        <tr><td>2</td><td><strong>Independence:</strong> ডেটা পয়েন্টগুলো পরস্পর independent</td><td>Durbin-Watson test</td></tr>
        <tr><td>3</td><td><strong>Homoscedasticity:</strong> residuals-এর variance সমান</td><td>Residual plot</td></tr>
        <tr><td>4</td><td><strong>Normality:</strong> residuals normal distribution অনুসরণ করে</td><td>Q-Q plot, Shapiro-Wilk</td></tr>
        <tr><td>5</td><td><strong>No Multicollinearity:</strong> ফিচারগুলো পরস্পর correlated না</td><td>VIF < 5</td></tr>
      </tbody>
    </table>

    <h3>প্রশ্ন ৩: Gradient Descent কী এবং কীভাবে কাজ করে?</h3>
    <p>Gradient Descent একটি optimization algorithm যেটি step by step cost function কমায়।</p>
    <pre><code># ধাপগুলো:
# ১. m এবং b-এর যেকোনো মান দিয়ে শুরু করো
# ২. gradient (partial derivative) হিসাব করো
# ৩. learning rate দিয়ে আপডেট করো
# ৪. যতক্ষণ না convergence হয় পুনরাবৃত্তি করো

m = m - α × (∂MSE/∂m)
b = b - α × (∂MSE/∂b)

# α খুব বড় হলে → diverge করে (কখনো converge হয় না)
# α খুব ছোট হলে → অনেক সময় লাগে
# সাধারণ ভালো মান: 0.01, 0.001</code></pre>

    <h3>প্রশ্ন ৪: Overfitting এবং Underfitting কী? কীভাবে সমাধান করবে?</h3>
    <table>
      <thead><tr><th></th><th>Underfitting</th><th>Overfitting</th></tr></thead>
      <tbody>
        <tr><td><strong>লক্ষণ</strong></td><td>Training এবং Test দুটোতেই খারাপ</td><td>Training-এ ভালো, Test-এ খারাপ</td></tr>
        <tr><td><strong>কারণ</strong></td><td>মডেল খুব সরল, ফিচার কম</td><td>মডেল খুব জটিল, ফিচার বেশি</td></tr>
        <tr><td><strong>সমাধান</strong></td><td>বেশি ফিচার যোগ করো, জটিল মডেল ব্যবহার করো</td><td>Regularization, বেশি ডেটা, Feature Reduction</td></tr>
      </tbody>
    </table>

    <h3>প্রশ্ন ৫: R² এবং Adjusted R²-এর পার্থক্য কী?</h3>
    <pre><code># R² সমস্যা: ফিচার যোগ করলে সবসময় বাড়ে
# (এমনকি অর্থহীন ফিচারও R² বাড়িয়ে দেয়)

# Adjusted R² এই সমস্যা ঠিক করে — অর্থহীন ফিচার যোগ করলে কমে

# নিয়ম:
# একক ফিচার হলে → R² ব্যবহার করো
# একাধিক ফিচার তুলনায় → Adjusted R² ব্যবহার করো</code></pre>

    <h3>প্রশ্ন ৬: OLS কীভাবে কাজ করে?</h3>
    <p>OLS (Ordinary Least Squares) residuals-এর বর্গের যোগফল minimize করে <strong>analytically</strong> (গণিত দিয়ে সরাসরি) সেরা m এবং b বের করে।</p>
    <pre><code># Matrix form-এ:
β = (XᵀX)⁻¹ Xᵀy

# এটি closed-form solution — iteration দরকার নেই
# কিন্তু বড় ডেটায় matrix inversion ধীর → Gradient Descent ভালো</code></pre>

    <h3>প্রশ্ন ৭: Lasso কীভাবে Feature Selection করে?</h3>
    <p>Lasso-এর L1 penalty-র geometric shape হলো diamond। Optimization করতে গিয়ে solution প্রায়ই diamond-এর কোণে আসে — সেখানে কিছু coefficient ঠিক শূন্য হয়।</p>
    <pre><code># Lasso: কিছু coefficient = 0 → সেই ফিচার বাদ
# Ridge: সব coefficient ছোট কিন্তু কখনো শূন্য নয়

# তাই:
# Feature selection দরকার হলে → Lasso
# সব feature রাখতে চাইলে → Ridge
# দুটোর মিশ্রণ দরকার হলে → ElasticNet</code></pre>

    <h3>প্রশ্ন ৮: কখন লিনিয়ার রিগ্রেশন ব্যবহার করা উচিত নয়?</h3>
    <ul>
      <li>যখন সম্পর্ক non-linear (তখন polynomial regression বা tree-based model)</li>
      <li>যখন outlier অনেক বেশি (robust regression ব্যবহার করো)</li>
      <li>যখন target binary বা categorical (logistic regression ব্যবহার করো)</li>
      <li>যখন ফিচারের চেয়ে ডেটা কম (n &lt; p) — regularization দরকার</li>
    </ul>

    <h3>প্রশ্ন ৯: Regularization কী এবং কেন দরকার?</h3>
    <pre><code># Regularization মডেলকে "শাস্তি" দেয় জটিল হওয়ার জন্য
# এটি overfitting কমায়

# Ridge (L2): Loss + α × Σmᵢ²
# Lasso (L1): Loss + α × Σ|mᵢ|
# ElasticNet:  Loss + α₁ × Σ|mᵢ| + α₂ × Σmᵢ²

from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1, 50% L2
en.fit(X_train, y_train)</code></pre>

    <h3>প্রশ্ন ১০: Feature Scaling কেন দরকার?</h3>
    <pre><code># উদাহরণ:
# বয়স: 20–60
# আয়: 10,000–1,000,000

# Scaling ছাড়া: বড় মানের ফিচার coefficient-কে dominate করে
# Gradient Descent ধীর হয়ে যায়

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# StandardScaler: গড় বাদ দিয়ে std দিয়ে ভাগ
# MinMaxScaler:   0 থেকে 1-এর মধ্যে নিয়ে আসে</code></pre>

    <h3>দ্রুত রিভিশন টেবিল</h3>
    <table>
      <thead><tr><th>প্রশ্ন</th><th>এক লাইনে উত্তর</th></tr></thead>
      <tbody>
        <tr><td>লিনিয়ার রিগ্রেশন কী?</td><td>Y = mX + b — continuous output ভবিষ্যদ্বাণীর মডেল</td></tr>
        <tr><td>Cost function কী?</td><td>MSE — (Actual - Predicted)² এর গড়</td></tr>
        <tr><td>R² মানে কী?</td><td>মডেল data-এর কতটুকু variation ব্যাখ্যা করে (0-1)</td></tr>
        <tr><td>Multicollinearity সমাধান?</td><td>Ridge বা Lasso regularization, VIF দিয়ে চেক</td></tr>
        <tr><td>Ridge vs Lasso পার্থক্য?</td><td>Ridge সব রাখে (ছোট করে), Lasso কিছু শূন্য করে</td></tr>
        <tr><td>Overfitting কমাবো কীভাবে?</td><td>Regularization, বেশি ডেটা, কম ফিচার</td></tr>
        <tr><td>Feature Scaling কেন?</td><td>বড় মানের ফিচার dominance ঠেকাতে</td></tr>
        <tr><td>OLS vs Gradient Descent?</td><td>OLS: ছোট ডেটায়, GD: বড় ডেটায়</td></tr>
      </tbody>
    </table>
  `,
};
