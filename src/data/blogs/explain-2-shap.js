export const explain_2_shap = {
  slug: 'explain-2-shap',
  title: 'SHAP: শ্যাপলি ভ্যালু দিয়ে মডেল ব্যাখ্যা',
  description: 'গেম থিওরির শ্যাপলি ভ্যালু থেকে SHAP, TreeSHAP, KernelSHAP, Force Plots, Summary Plots এবং Beeswarm Plots সহ সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'মডেল ব্যাখ্যাযোগ্যতা',
  readTime: 13,
  content: `
    <h3>১. শ্যাপলি ভ্যালু: গেম থিওরি থেকে ML-এ</h3>
    <p>
      ১৯৫৩ সালে গণিতবিদ <strong>Lloyd Shapley</strong> একটি সমস্যার সমাধান দিয়েছিলেন: একটি সহযোগিতামূলক খেলায় (cooperative game) যদি একাধিক খেলোয়াড় একসাথে পুরস্কার জেতে, তাহলে প্রতিটি খেলোয়াড়ের ন্যায্য ভাগ কত হবে? তাঁর উত্তর ছিল <strong>শ্যাপলি ভ্যালু</strong> — প্রতিটি খেলোয়াড়ের অবদান হিসাব করা হয় তাকে বিভিন্ন সমন্বয়ে (coalition) যুক্ত ও বাদ দিয়ে।
    </p>
    <p>
      ML-এ এই ধারণাটি চমৎকারভাবে প্রযোগ করা যায়। এখানে "খেলোয়াড়" হলো ফিচার, "পুরস্কার" হলো মডেলের ভবিষ্যদ্বাণী। প্রতিটি ফিচারের শ্যাপলি ভ্যালু বলে দেয় — এই নির্দিষ্ট ভবিষ্যদ্বাণীতে এই ফিচারটি গড় ভবিষ্যদ্বাণী থেকে কতটা সরিয়ে দিয়েছে।
    </p>
    <pre><code>import numpy as np
from itertools import combinations

def shapley_value(feature_idx, X_sample, model, feature_names):
    """
    ব্রুট-ফোর্স শ্যাপলি ভ্যালু হিসাব (ছোট ফিচার সেটের জন্য)
    """
    n_features = len(feature_names)
    others = [i for i in range(n_features) if i != feature_idx]

    shapley = 0.0
    baseline = model.predict([X_sample.mean(axis=0)])[0]

    for size in range(len(others) + 1):
        for subset in combinations(others, size):
            # subset ছাড়া prediction
            x_without = X_sample.mean(axis=0).copy()
            for idx in subset:
                x_without[idx] = X_sample[:, idx].mean()
            pred_without = model.predict([x_without])[0]

            # subset + feature_idx সহ prediction
            x_with = x_without.copy()
            x_with[feature_idx] = X_sample[0, feature_idx]
            pred_with = model.predict([x_with])[0]

            # Weight: |S|!(n-|S|-1)!/n!
            s = len(subset)
            weight = (np.math.factorial(s) *
                      np.math.factorial(n_features - s - 1) /
                      np.math.factorial(n_features))
            shapley += weight * (pred_with - pred_without)

    return shapley</code></pre>

    <h3>২. SHAP = SHapley Additive exPlanations</h3>
    <p>
      <strong>SHAP</strong> (Lundberg & Lee, 2017) হলো শ্যাপলি ভ্যালু-ভিত্তিক একটি ইউনিফাইড ফ্রেমওয়ার্ক। SHAP-এর মূল সম্পর্ক:
    </p>
    <p>
      <code>f(x) = E[f(X)] + φ₁ + φ₂ + ... + φₙ</code>
    </p>
    <p>
      যেখানে <code>f(x)</code> হলো নির্দিষ্ট ভবিষ্যদ্বাণী, <code>E[f(X)]</code> হলো সকল ডেটার গড় ভবিষ্যদ্বাণী (baseline), এবং <code>φᵢ</code> হলো i-তম ফিচারের SHAP ভ্যালু।
    </p>
    <p>
      SHAP-এর তিনটি মূল গুণ:
    </p>
    <p>
      <strong>Local Accuracy:</strong> সকল SHAP ভ্যালুর যোগফল = ভবিষ্যদ্বাণী - baseline।<br/>
      <strong>Missingness:</strong> যে ফিচারের মান নেই, তার SHAP ভ্যালু শূন্য।<br/>
      <strong>Consistency:</strong> কোনো মডেলে যদি একটি ফিচারের প্রভাব বাড়ে, তার SHAP ভ্যালু কমতে পারে না।
    </p>

    <h3>৩. TreeSHAP: ট্রি মডেলের জন্য দ্রুত SHAP</h3>
    <p>
      ব্রুট-ফোর্স শ্যাপলি হিসাব <code>O(2^n)</code> সময় নেয়, যা বড় ফিচার সেটে অব্যবহারিক। <strong>TreeSHAP</strong> ট্রি স্ট্রাকচার ব্যবহার করে এটিকে <code>O(TLD²)</code> করে (T = গাছের সংখ্যা, L = পাতার সংখ্যা, D = গভীরতা)।
    </p>
    <pre><code>import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# ডেটা প্রস্তুত
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest মডেল
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")

# TreeSHAP Explainer তৈরি
explainer = shap.TreeExplainer(rf)

# SHAP ভ্যালু হিসাব (class 1 = malignant)
shap_values = explainer.shap_values(X_test)
# shap_values[1] = class 1 এর জন্য SHAP values
print(f"\nSHAP values shape: {np.array(shap_values[1]).shape}")
print(f"Expected value (baseline): {explainer.expected_value[1]:.4f}")

# Verification: SHAP ভ্যালুর যোগফল = prediction - baseline
sample = 0
shap_sum = shap_values[1][sample].sum()
predicted_prob = rf.predict_proba(X_test.iloc[[sample]])[0][1]
print(f"\nSample {sample} verification:")
print(f"  Sum of SHAP values:   {shap_sum:.4f}")
print(f"  Predicted prob:       {predicted_prob:.4f}")
print(f"  Baseline:             {explainer.expected_value[1]:.4f}")
print(f"  Sum + Baseline:       {shap_sum + explainer.expected_value[1]:.4f}")</code></pre>

    <h3>৪. KernelSHAP: মডেল-অ্যাগনস্টিক SHAP</h3>
    <p>
      ট্রি নয় এমন মডেলের জন্য (Logistic Regression, SVM, Neural Network) <strong>KernelSHAP</strong> ব্যবহার করা হয়। এটি LIME-এর মতো কিন্তু বিশেষ কার্নেল ওজন ব্যবহার করে শ্যাপলি ভ্যালুর সকল গুণ নিশ্চিত করে। তুলনামূলক ধীর কিন্তু যেকোনো মডেলে কাজ করে।
    </p>
    <pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Logistic Regression পাইপলাইন (ট্রি নয়)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)
print(f"LR Test Accuracy: {pipe.score(X_test, y_test):.4f}")

# KernelSHAP — background data হিসেবে শুধু ১০০টি sample
background = shap.sample(X_train, 100, random_state=42)
kernel_explainer = shap.KernelExplainer(
    pipe.predict_proba, background
)

# শুধু ২০টি test sample (KernelSHAP ধীর)
X_test_small = X_test.iloc[:20]
kernel_shap_values = kernel_explainer.shap_values(X_test_small)

print(f"KernelSHAP values shape: {np.array(kernel_shap_values[1]).shape}")

# গ্লোবাল গুরুত্ব
mean_abs = np.abs(kernel_shap_values[1]).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': mean_abs
}).sort_values('mean_abs_shap', ascending=False)
print("\nTop 5 features (KernelSHAP):")
print(importance_df.head())</code></pre>

    <h3>৫. Force Plots: একটি ভবিষ্যদ্বাণীর ব্যাখ্যা</h3>
    <p>
      <strong>Force Plot</strong> একটি নির্দিষ্ট ভবিষ্যদ্বাণীর SHAP ভ্যালু ভিজুয়ালি দেখায়। লাল রঙের ফিচারগুলো ভবিষ্যদ্বাণীকে উপরে ঠেলে দিচ্ছে (ক্লাস 1 এর দিকে), নীল রঙের ফিচারগুলো নিচে ঠেলছে।
    </p>
    <pre><code># Single prediction force plot
sample_idx = 0
print(f"Prediction for sample {sample_idx}:")
print(f"  Actual:    {y_test.iloc[sample_idx]}")
print(f"  Predicted: {rf.predict(X_test.iloc[[sample_idx]])[0]}")
print(f"  Probability (malignant): {rf.predict_proba(X_test.iloc[[sample_idx]])[0][1]:.4f}")
print(f"  Baseline: {explainer.expected_value[1]:.4f}")

# SHAP ভ্যালু বিস্তারিত
sample_shap = shap_values[1][sample_idx]
feature_shap_df = pd.DataFrame({
    'feature': X_test.columns,
    'value': X_test.iloc[sample_idx].values,
    'shap_value': sample_shap
}).sort_values('shap_value', key=abs, ascending=False)

print(f"\nTop 10 SHAP contributions for sample {sample_idx}:")
print(feature_shap_df.head(10).to_string(index=False))

# HTML/Notebook-এ Force Plot দেখান
# shap.initjs()
# shap.force_plot(
#     explainer.expected_value[1],
#     shap_values[1][sample_idx],
#     X_test.iloc[sample_idx],
#     feature_names=list(X_test.columns)
# )</code></pre>

    <h3>৬. Summary Plots: গ্লোবাল গুরুত্ব</h3>
    <p>
      <strong>Summary Plot</strong> (বা Dot Plot) সকল টেস্ট স্যাম্পলের SHAP ভ্যালু একসাথে দেখায়। প্রতিটি বিন্দু একটি স্যাম্পল, রঙ ফিচারের মান (লাল = উচ্চ, নীল = নিম্ন), এবং x-অক্ষ SHAP ভ্যালু।
    </p>
    <pre><code>import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Bar plot: গ্লোবাল গুরুত্ব
print("Generating SHAP Summary (Bar) Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values[1], X_test,
    plot_type="bar",
    max_display=10,
    show=False
)
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=150)
plt.close()
print("Saved: shap_bar_plot.png")

# Dot plot: প্রতিটি স্যাম্পলের SHAP ভ্যালু
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values[1], X_test,
    max_display=10,
    show=False
)
plt.title("SHAP Summary Plot (Beeswarm)")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150)
plt.close()
print("Saved: shap_summary_plot.png")</code></pre>

    <h3>৭. Beeswarm Plots এবং Dependence Plots</h3>
    <p>
      <strong>Beeswarm Plot</strong> Summary Plot-এর উন্নত সংস্করণ। প্রতিটি স্যাম্পলের জন্য একটি বিন্দু, ওভারল্যাপ কমাতে বিন্দুগুলো ছড়িয়ে দেওয়া হয়।
    </p>
    <p>
      <strong>Dependence Plot</strong> একটি নির্দিষ্ট ফিচারের মান এবং তার SHAP ভ্যালুর সম্পর্ক দেখায়। এটি non-linear effects ধরতে সক্ষম।
    </p>
    <pre><code># Dependence plot: সবচেয়ে গুরুত্বপূর্ণ ফিচারের জন্য
top_feature = importance_df.iloc[0]['feature'] if 'importance_df' in dir() else 'worst radius'

print(f"\nGenerating Dependence Plot for: {top_feature}")
fig, ax = plt.subplots(figsize=(8, 5))
shap.dependence_plot(
    top_feature,
    shap_values[1],
    X_test,
    ax=ax,
    show=False
)
plt.title(f"SHAP Dependence Plot: {top_feature}")
plt.tight_layout()
plt.savefig("shap_dependence_plot.png", dpi=150)
plt.close()
print("Saved: shap_dependence_plot.png")

# Interaction values (feature interaction)
print("\nComputing SHAP Interaction Values...")
shap_interaction = explainer.shap_interaction_values(X_test.iloc[:50])
print(f"Interaction values shape: {np.array(shap_interaction).shape}")</code></pre>

    <h3>৮. SHAP Waterfall Plot</h3>
    <p>
      <strong>Waterfall Plot</strong> একটি ভবিষ্যদ্বাণীর ধাপে ধাপে ব্যাখ্যা দেয় — baseline থেকে শুরু করে প্রতিটি ফিচারের অবদান যোগ/বিয়োগ করে চূড়ান্ত ভবিষ্যদ্বাণীতে পৌঁছানো হয়।
    </p>
    <pre><code># Waterfall plot (shap >= 0.40)
sample_idx = 3

# Explanation object তৈরি
explanation = shap.Explanation(
    values=shap_values[1][sample_idx],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[sample_idx].values,
    feature_names=list(X_test.columns)
)

fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(explanation, max_display=10, show=False)
plt.title(f"SHAP Waterfall Plot — Sample {sample_idx}")
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150)
plt.close()
print("Saved: shap_waterfall.png")

# সংক্ষেপে: কোন ফিচার কোন দিকে টানছে
pos_features = explanation.values[explanation.values > 0]
neg_features = explanation.values[explanation.values < 0]
print(f"\nFeatures pushing prediction UP:   {len(pos_features)}")
print(f"Features pushing prediction DOWN: {len(neg_features)}")
print(f"Net effect: {explanation.values.sum():.4f}")</code></pre>

    <h3>৯. সারসংক্ষেপ ও পরবর্তী ধাপ</h3>
    <p>
      SHAP হলো মডেল ব্যাখ্যার সবচেয়ে শক্তিশালী এবং তাত্ত্বিকভাবে সুপ্রতিষ্ঠিত পদ্ধতি। মূল বিষয়গুলো:
    </p>
    <p>
      TreeSHAP ট্রি মডেলের জন্য দ্রুত এবং সঠিক। KernelSHAP যেকোনো মডেলে কাজ করে কিন্তু ধীর। Force Plot লোকাল, Summary/Beeswarm Plot গ্লোবাল ব্যাখ্যা দেয়। Waterfall Plot ধাপে ধাপে ব্যাখ্যা করে। Dependence Plot ফিচারের non-linear প্রভাব দেখায়।
    </p>
    <p>
      পরবর্তী পর্বে আমরা LIME দেখব — যা আরও সহজ এবং ভিন্ন দৃষ্টিভঙ্গি থেকে মডেল ব্যাখ্যা করে।
    </p>
  `
};
