export const dt_7_feature_importance = {
  title: "Feature Importance ও SHAP — মডেল কেন এই সিদ্ধান্ত নিলো?",
  description: "Gini Importance, Permutation Importance, SHAP Values — Decision Tree ও Ensemble মডেলের সিদ্ধান্ত ব্যাখ্যা করার সব পদ্ধতি বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "ডিসিশন ট্রি",
  readTime: 11,
  slug: "dt-feature-importance-shap",
  content: `
    <h3>১. কেন Feature Importance জানা দরকার?</h3>
    <p>শুধু accuracy দিয়ে মডেল বোঝা যায় না। জানতে হবে:</p>
    <ul>
      <li>কোন ফিচার সবচেয়ে গুরুত্বপূর্ণ?</li>
      <li>একটি নির্দিষ্ট prediction কেন হলো?</li>
      <li>মডেল কোনো ভুল pattern শিখেছে কিনা?</li>
    </ul>
    <p>এই প্রশ্নগুলোর উত্তর দেওয়াকে বলে <strong>Model Interpretability</strong>।</p>

    <h3>২. Gini-based Importance (MDI)</h3>
    <pre><code">import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# MDI (Mean Decrease in Impurity)
mdi_imp = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': rf.feature_importances_,
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(9, 6))
plt.barh(mdi_imp['Feature'][:10], mdi_imp['Importance'][:10], color='steelblue')
plt.xlabel('MDI Feature Importance')
plt.title('Top 10 Features (Gini-based)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("সমস্যা: high cardinality feature-কে overestimate করে!")</code></pre>

    <h3>৩. Permutation Importance</h3>
    <p>একটি ফিচারের মান randomly shuffle করো — accuracy কতটুকু কমলো সেটি হলো সেই ফিচারের গুরুত্ব।</p>
    <pre><code">from sklearn.inspection import permutation_importance

# Test set-এ calculate করো
perm = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,        # ৩০ বার shuffle করে গড় নাও
    random_state=42,
    n_jobs=-1,
)

perm_imp = pd.DataFrame({
    'Feature': feature_names,
    'Mean':    perm.importances_mean,
    'Std':     perm.importances_std,
}).sort_values('Mean', ascending=False)

# Error bar সহ plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(perm_imp['Feature'][:10], perm_imp['Mean'][:10],
        xerr=perm_imp['Std'][:10], color='coral', capsize=4)
ax.set_xlabel('Permutation Importance (accuracy drop)')
ax.set_title('Top 10 Features (Permutation-based)')
ax.invert_yaxis()
plt.tight_layout()
plt.show()</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Permutation Importance — কীভাবে কাজ করে</text>
        <rect x="15" y="28" width="120" height="60" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="75" y="48" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">Original Data</text>
        <text x="75" y="62" text-anchor="middle" font-size="8" fill="#3b82f6">Accuracy = 0.97</text>
        <text x="75" y="80" text-anchor="middle" font-size="8" fill="#6b7280">(baseline)</text>
        <text x="143" y="60" font-size="14" fill="#94a3b8">→</text>
        <rect x="158" y="28" width="140" height="60" rx="4" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="228" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Feature X shuffle করো</text>
        <text x="228" y="58" text-anchor="middle" font-size="8" fill="#d97706">X: [1,3,2,5] → [3,1,5,2]</text>
        <text x="228" y="72" text-anchor="middle" font-size="8" fill="#d97706">Accuracy = 0.82</text>
        <text x="306" y="60" font-size="14" fill="#94a3b8">→</text>
        <rect x="322" y="28" width="145" height="60" rx="4" fill="#dcfce7" stroke="#86efac"/>
        <text x="394" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Importance(X)</text>
        <text x="394" y="60" text-anchor="middle" font-size="10" fill="#16a34a">= 0.97 − 0.82</text>
        <text x="394" y="76" text-anchor="middle" font-size="10" fill="#16a34a">= 0.15</text>
      </svg>
    </div>

    <h3>৪. SHAP Values — সর্বোত্তম ব্যাখ্যা পদ্ধতি</h3>
    <p>SHAP (SHapley Additive exPlanations) প্রতিটি prediction-এর জন্য প্রতিটি ফিচারের contribution বলে দেয়।</p>
    <pre><code"># pip install shap
import shap

# TreeExplainer — tree-based model-এর জন্য অত্যন্ত দ্রুত
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Classification-এ shap_values হলো list: [class_0, class_1]
# shap_values[1] = class_1 (positive)-এর জন্য

# ১. Global Feature Importance
shap.summary_plot(shap_values[1], X_test,
                  feature_names=feature_names, plot_type='bar')</code></pre>

    <h3>৫. SHAP Summary Plot — Feature ও Direction</h3>
    <pre><code"># প্রতিটি feature কোন দিকে কতটুকু প্রভাব রাখে
shap.summary_plot(shap_values[1], X_test,
                  feature_names=feature_names)

# পড়ার উপায়:
# x-axis: SHAP value (+ = positive class-এর দিকে ঠেলে, − = negative)
# color: feature value (লাল=বেশি, নীল=কম)
# dot: একটি sample

# উদাহরণ:
# worst_radius-এর SHAP value বেশি positive → radius বেশি → malignant বেশি সম্ভব</code></pre>

    <h3>৬. একটি Prediction ব্যাখ্যা করো (Waterfall Plot)</h3>
    <pre><code"># একটি নির্দিষ্ট sample-এর prediction কেন হলো?
sample_idx = 5

# shap.Explanation object তৈরি করো
explanation = shap.Explanation(
    values=shap_values[1][sample_idx],
    base_values=explainer.expected_value[1],
    data=X_test[sample_idx],
    feature_names=feature_names,
)

# Waterfall plot: base → final prediction পথ
shap.plots.waterfall(explanation)

# Force plot: একটি সারিতে দেখো
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_idx],
    X_test[sample_idx],
    feature_names=feature_names,
)</code></pre>

    <h3>৭. SHAP Dependence Plot</h3>
    <pre><code"># একটি feature-এর SHAP value কীভাবে বদলায়?
shap.dependence_plot(
    'worst radius',         # feature নাম বা index
    shap_values[1],
    X_test,
    feature_names=feature_names,
    interaction_index='auto'  # interaction feature auto-detect
)
# x-axis: feature value
# y-axis: SHAP value (এই feature-এর contribution)
# color: interaction feature</code></pre>

    <h3>৮. তিনটি পদ্ধতির তুলনা</h3>
    <table>
      <thead><tr><th>পদ্ধতি</th><th>কী দেয়</th><th>সুবিধা</th><th>সীমাবদ্ধতা</th></tr></thead>
      <tbody>
        <tr><td><strong>MDI</strong></td><td>Global importance</td><td>দ্রুত, built-in</td><td>high-cardinality bias</td></tr>
        <tr><td><strong>Permutation</strong></td><td>Global importance</td><td>নির্ভরযোগ্য, model-agnostic</td><td>correlated feature-এ সমস্যা</td></tr>
        <tr><td><strong>SHAP</strong></td><td>Global + Local (per sample)</td><td>সবচেয়ে সঠিক, additive</td><td>ধীর (non-tree model-এ)</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>MDI</td><td>দ্রুত কিন্তু high cardinality-তে bias আছে</td></tr>
        <tr><td>Permutation</td><td>shuffle করে accuracy drop মাপে — বেশি নির্ভরযোগ্য</td></tr>
        <tr><td>SHAP</td><td>প্রতিটি prediction-এর per-feature contribution — সেরা পদ্ধতি</td></tr>
        <tr><td>Waterfall</td><td>একটি sample-এর prediction step-by-step ব্যাখ্যা</td></tr>
        <tr><td>Dependence Plot</td><td>feature value বদলালে SHAP কীভাবে বদলায় তা দেখায়</td></tr>
      </tbody>
    </table>
  `,
};
