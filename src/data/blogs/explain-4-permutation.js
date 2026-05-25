export const explain_4_permutation = {
  slug: 'explain-4-permutation',
  title: 'পারমিউটেশন ইম্পোর্টেন্স ও PDP',
  description: 'পারমিউটেশন ফিচার ইম্পোর্টেন্স, Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE) এবং sklearn ও PDPbox দিয়ে বাস্তব প্রয়োগ।',
  date: 'মে ২০২৬',
  category: 'মডেল ব্যাখ্যাযোগ্যতা',
  readTime: 11,
  content: `
    <h3>১. পারমিউটেশন ফিচার ইম্পোর্টেন্স কী?</h3>
    <p>
      <strong>পারমিউটেশন ফিচার ইম্পোর্টেন্স (Permutation Feature Importance)</strong> একটি সহজ কিন্তু কার্যকর পদ্ধতি। মূল ধারণা: যদি একটি ফিচার সত্যিই গুরুত্বপূর্ণ হয়, তাহলে সেই ফিচারের মানগুলো এলোমেলো (shuffle) করে দিলে মডেলের performance উল্লেখযোগ্যভাবে কমে যাবে।
    </p>
    <p>
      গাণিতিকভাবে: একটি ফিচার <code>j</code>-এর পারমিউটেশন ইম্পোর্টেন্স =
    </p>
    <p>
      <code>PFI_j = error(model, X_permuted_j) - error(model, X_original)</code>
    </p>
    <p>
      যদি <code>PFI_j</code> বড় হয়, সেই ফিচার গুরুত্বপূর্ণ। শূন্য বা ঋণাত্মক হলে সেই ফিচার মডেলের জন্য তেমন দরকারি নয়।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# ডেটা প্রস্তুত
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# মডেল তৈরি
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Baseline Test Accuracy: {baseline_acc:.4f}")

# Manual Permutation Importance
def manual_permutation_importance(model, X_test, y_test, n_repeats=5):
    baseline = accuracy_score(y_test, model.predict(X_test))
    importances = {}

    for col in X_test.columns:
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            permuted_acc = accuracy_score(y_test, model.predict(X_permuted))
            drops.append(baseline - permuted_acc)   # accuracy drop
        importances[col] = np.mean(drops)

    return pd.Series(importances).sort_values(ascending=False)

pfi = manual_permutation_importance(rf, X_test, y_test, n_repeats=5)
print("\nManual Permutation Importance (top 10):")
print(pfi.head(10).to_string())</code></pre>

    <h3>২. sklearn-এর permutation_importance</h3>
    <p>
      sklearn 0.22 থেকে <code>permutation_importance</code> বিল্ট-ইন হিসেবে পাওয়া যায়। এটি অনেক বেশি efficient এবং বিভিন্ন scoring metric সাপোর্ট করে।
    </p>
    <pre><code>from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sklearn's permutation_importance
result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,          # প্রতিটি ফিচার ১০ বার shuffle
    random_state=42,
    scoring='accuracy',    # metric
    n_jobs=-1              # parallel processing
)

# ফলাফল সংগঠিত করুন
pfi_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

print("Permutation Feature Importance (sklearn):")
print(pfi_df.head(10).to_string(index=False))

# বিল্ট-ইন Feature Importance-এর সাথে তুলনা
builtin_fi = pd.DataFrame({
    'feature': X_test.columns,
    'builtin': rf.feature_importances_
}).sort_values('builtin', ascending=False)

# Comparison
merged = pfi_df.merge(builtin_fi, on='feature')
print("\nComparison: Permutation vs Built-in Importance (top 10):")
print(merged.head(10)[['feature', 'importance_mean', 'builtin']].to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Permutation importance (horizontal bar with error bars)
top10 = pfi_df.head(10)
axes[0].barh(top10['feature'][::-1], top10['importance_mean'][::-1],
             xerr=top10['importance_std'][::-1], color='steelblue', alpha=0.8)
axes[0].set_xlabel("Accuracy Drop")
axes[0].set_title("Permutation Feature Importance")

# Built-in importance
top10_bi = builtin_fi.head(10)
axes[1].barh(top10_bi['feature'][::-1], top10_bi['builtin'][::-1],
             color='coral', alpha=0.8)
axes[1].set_xlabel("Importance")
axes[1].set_title("Built-in Feature Importance")

plt.tight_layout()
plt.savefig("permutation_vs_builtin.png", dpi=150)
plt.close()
print("\nSaved: permutation_vs_builtin.png")</code></pre>

    <h3>৩. কখন পারমিউটেশন ইম্পোর্টেন্স বিল্ট-ইন থেকে ভালো?</h3>
    <p>
      ট্রি মডেলের বিল্ট-ইন ফিচার ইম্পোর্টেন্সে একটি সমস্যা আছে — এটি <strong>ট্রেনিং ডেটায়</strong> হিসাব করা হয় এবং উচ্চ-cardinality ফিচার (অনেক unique মান আছে এমন) এবং পারস্পরিক সম্পর্কযুক্ত ফিচারের ক্ষেত্রে biased হতে পারে। পারমিউটেশন ইম্পোর্টেন্স <strong>টেস্ট ডেটায়</strong> হিসাব করা হয়, তাই এটি generalization performance প্রতিফলিত করে।
    </p>
    <pre><code># Correlated features-এর সমস্যা প্রদর্শন
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# দুটি পারস্পরিক সম্পর্কযুক্ত ফিচার তৈরি
X_corr, y_corr = make_classification(
    n_samples=1000, n_features=5, n_informative=3,
    n_redundant=2, random_state=42
)
feature_names_corr = [f'feature_{i}' for i in range(5)]
X_corr_df = pd.DataFrame(X_corr, columns=feature_names_corr)

# Correlation matrix
print("Correlation Matrix:")
print(X_corr_df.corr().round(2).to_string())

rf_corr = RandomForestClassifier(n_estimators=100, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_corr, y_corr, test_size=0.2, random_state=42)
rf_corr.fit(X_tr, y_tr)

# Built-in vs Permutation
print("\nBuilt-in importance:")
for name, imp in zip(feature_names_corr, rf_corr.feature_importances_):
    print(f"  {name}: {imp:.4f}")

perm = permutation_importance(rf_corr, X_te, y_te, n_repeats=10, random_state=42)
print("\nPermutation importance:")
for name, imp in zip(feature_names_corr, perm.importances_mean):
    print(f"  {name}: {imp:.4f}")</code></pre>

    <h3>৪. Partial Dependence Plots (PDP)</h3>
    <p>
      <strong>PDP</strong> গ্লোবাল ব্যাখ্যার একটি শক্তিশালী টুল। এটি দেখায় — একটি ফিচারের মান পরিবর্তন হলে (অন্য সব ফিচার স্থির রেখে গড় করে) মডেলের ভবিষ্যদ্বাণী কীভাবে পরিবর্তিত হয়।
    </p>
    <p>
      গাণিতিক সংজ্ঞা: ফিচার <code>S</code>-এর জন্য PDP:
    </p>
    <p>
      <code>PDP(x_S) = E_{x_C}[f(x_S, x_C)] = (1/n) Σ f(x_S, x_C^(i))</code>
    </p>
    <p>
      যেখানে <code>x_C</code> হলো অন্যান্য (complement) ফিচার।
    </p>
    <pre><code>from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# একটি ফিচারের PDP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDP for top 2 features
top2_features = list(pfi_df['feature'].head(2))
print(f"Plotting PDP for: {top2_features}")

features_idx = [list(X_test.columns).index(f) for f in top2_features]

PartialDependenceDisplay.from_estimator(
    rf,
    X_test,
    features=features_idx,
    feature_names=list(X_test.columns),
    kind='average',    # PDP (averaged)
    ax=axes,
    grid_resolution=50,
    response_method='predict_proba'
)

axes[0].set_title(f"PDP: {top2_features[0]}")
axes[1].set_title(f"PDP: {top2_features[1]}")
plt.suptitle("Partial Dependence Plots", y=1.02)
plt.tight_layout()
plt.savefig("pdp_plot.png", dpi=150)
plt.close()
print("Saved: pdp_plot.png")

# 2D PDP (interaction between two features)
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    rf,
    X_test,
    features=[(features_idx[0], features_idx[1])],
    feature_names=list(X_test.columns),
    kind='average',
    ax=[ax],
    response_method='predict_proba'
)
plt.title(f"2D PDP: {top2_features[0]} vs {top2_features[1]}")
plt.tight_layout()
plt.savefig("pdp_2d.png", dpi=150)
plt.close()
print("Saved: pdp_2d.png")</code></pre>

    <h3>৫. Individual Conditional Expectation (ICE)</h3>
    <p>
      PDP সকল স্যাম্পলের গড় দেখায়। কিন্তু গড় misleading হতে পারে যদি বিভিন্ন স্যাম্পলে ফিচারের প্রভাব ভিন্ন হয় (subgroup effects)। <strong>ICE (Individual Conditional Expectation)</strong> প্রতিটি স্যাম্পলের জন্য আলাদা লাইন আঁকে।
    </p>
    <pre><code># ICE plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ICE (kind='individual') + PDP (kind='both')
PartialDependenceDisplay.from_estimator(
    rf,
    X_test,
    features=features_idx[:2],
    feature_names=list(X_test.columns),
    kind='both',       # ICE + PDP একসাথে
    subsample=50,      # ৫০টি ICE লাইন
    ax=axes,
    grid_resolution=50,
    response_method='predict_proba',
    alpha=0.3          # ICE লাইনের transparency
)

axes[0].set_title(f"ICE + PDP: {top2_features[0]}")
axes[1].set_title(f"ICE + PDP: {top2_features[1]}")
plt.suptitle("Individual Conditional Expectation (ICE) Plots", y=1.02)
plt.tight_layout()
plt.savefig("ice_plot.png", dpi=150)
plt.close()
print("Saved: ice_plot.png")

# Centered ICE (c-ICE): প্রতিটি লাইনকে x=min-এ শূন্য থেকে শুরু করানো
PartialDependenceDisplay.from_estimator(
    rf,
    X_test,
    features=features_idx[:1],
    feature_names=list(X_test.columns),
    kind='individual',
    subsample=50,
    centered=True,     # Centered ICE
    grid_resolution=50,
    response_method='predict_proba',
    alpha=0.3
)
plt.title(f"Centered ICE: {top2_features[0]}")
plt.tight_layout()
plt.savefig("centered_ice.png", dpi=150)
plt.close()
print("Saved: centered_ice.png")</code></pre>

    <h3>৬. PDPbox দিয়ে উন্নত ভিজুয়ালাইজেশন</h3>
    <p>
      <strong>PDPbox</strong> লাইব্রেরি আরও সমৃদ্ধ PDP ভিজুয়ালাইজেশন প্রদান করে — হিস্টোগ্রাম, confidence intervals এবং interaction plots সহ।
    </p>
    <pre><code># PDPbox (pip install pdpbox)
try:
    from pdpbox import pdp, info_plots
    print("PDPbox available")

    # PDPbox-এর জন্য DataFrame দরকার
    X_train_df = pd.DataFrame(X_train, columns=X_test.columns)

    # PDPbox: Isolated PDP
    pdp_isolate = pdp.pdp_isolate(
        model=rf,
        dataset=X_train_df,
        model_features=list(X_test.columns),
        feature=top2_features[0],
        num_grid_points=20
    )

    fig, axes = pdp.pdp_plot(
        pdp_isolate,
        feature_name=top2_features[0],
        plot_lines=True,
        frac_to_plot=0.1,   # ১০% ICE লাইন
        center=True,
        plot_pts_dist=True  # Distribution histogram
    )
    plt.savefig("pdpbox_isolated.png", dpi=150)
    plt.close()
    print("Saved: pdpbox_isolated.png")

    # PDPbox: Interaction PDP
    pdp_interact = pdp.pdp_interact(
        model=rf,
        dataset=X_train_df,
        model_features=list(X_test.columns),
        features=top2_features[:2]
    )
    fig, axes = pdp.pdp_interact_plot(
        pdp_interact,
        feature_names=top2_features[:2],
        plot_type='contour'
    )
    plt.savefig("pdpbox_interact.png", dpi=150)
    plt.close()
    print("Saved: pdpbox_interact.png")

except ImportError:
    print("PDPbox not installed. Using sklearn's PartialDependenceDisplay instead.")
    print("Install with: pip install pdpbox")</code></pre>

    <h3>৭. PDP-এর সীমাবদ্ধতা</h3>
    <p>
      PDP-এর একটি মূল সীমাবদ্ধতা হলো <strong>feature correlation</strong>। যদি দুটি ফিচার পারস্পরিক সম্পর্কযুক্ত হয়, PDP অবাস্তব ফিচার সমন্বয় তৈরি করতে পারে। এই সমস্যার সমাধান হলো <strong>Accumulated Local Effects (ALE) Plots</strong>।
    </p>
    <pre><code># ALE Plots (alibi লাইব্রেরি প্রয়োজন)
try:
    from alibi.explainers import ALE, plot_ale

    ale_explainer = ALE(
        rf.predict_proba,
        feature_names=list(X_test.columns),
        target_names=data.target_names
    )
    ale_exp = ale_explainer.explain(X_test.values)
    print("ALE computed successfully.")
    print(f"ALE values shape: {ale_exp.ale_values[0].shape}")

except ImportError:
    print("alibi not installed. Install with: pip install alibi")
    print("ALE is an alternative to PDP that handles correlated features better.")

# Summary: পদ্ধতিগুলোর তুলনা
summary = pd.DataFrame({
    'Method':     ['PDP', 'ICE', 'ALE', 'Permutation Importance'],
    'Type':       ['Global', 'Local+Global', 'Global', 'Global'],
    'Handles Correlation': ['No', 'No', 'Yes', 'Moderate'],
    'Shows Distribution':  ['No', 'Yes (per sample)', 'No', 'No'],
    'Speed':      ['Fast', 'Medium', 'Fast', 'Medium'],
})
print("\nComparison of Methods:")
print(summary.to_string(index=False))</code></pre>

    <h3>৮. সব পদ্ধতি একসাথে: সংক্ষিপ্ত পাইপলাইন</h3>
    <pre><code>from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np

# ১. মডেল তৈরি ও মূল্যায়ন
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")

# ২. Permutation Importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
pfi_series = pd.Series(perm.importances_mean, index=X_test.columns)
top5 = pfi_series.nlargest(5)
print("\nTop 5 Permutation Importance features:")
print(top5.to_string())

# ৩. PDP + ICE for top feature
top_feat_idx = list(X_test.columns).index(top5.index[0])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
PartialDependenceDisplay.from_estimator(
    rf, X_test, features=[top_feat_idx],
    feature_names=list(X_test.columns),
    kind='both', subsample=40, ax=[axes[0]],
    response_method='predict_proba', alpha=0.3
)
PartialDependenceDisplay.from_estimator(
    rf, X_test, features=[top_feat_idx],
    feature_names=list(X_test.columns),
    kind='average', ax=[axes[1]],
    response_method='predict_proba'
)
axes[0].set_title(f"ICE + PDP: {top5.index[0]}")
axes[1].set_title(f"PDP only: {top5.index[0]}")
plt.tight_layout()
plt.savefig("complete_pipeline.png", dpi=150)
plt.close()
print("Saved: complete_pipeline.png")</code></pre>

    <h3>৯. সারসংক্ষেপ</h3>
    <p>
      পারমিউটেশন ইম্পোর্টেন্স এবং PDP/ICE হলো গ্লোবাল ব্যাখ্যার দুটি সহজ কিন্তু শক্তিশালী পদ্ধতি। পারমিউটেশন ইম্পোর্টেন্স বলে কোন ফিচার গুরুত্বপূর্ণ, PDP বলে কীভাবে প্রভাব ফেলছে, এবং ICE দেখায় বিভিন্ন স্যাম্পলে প্রভাব আলাদা কিনা। পরবর্তী পর্বে আমরা একটি সম্পূর্ণ প্রজেক্টে এই সব পদ্ধতি একসাথে প্রয়োগ করব।
    </p>
  `
};
