export const explain_5_project = {
  slug: 'explain-5-project',
  title: 'মডেল ব্যাখ্যা প্রজেক্ট: ঋণ অনুমোদন মডেল',
  description: 'SHAP ও LIME দিয়ে একটি Random Forest ঋণ মডেলের সিদ্ধান্ত অ-প্রযুক্তিবিদ স্টেকহোল্ডারদের কাছে ব্যাখ্যা করুন',
  date: 'মে ২০২৫',
  category: 'মডেল ব্যাখ্যাযোগ্যতা',
  readTime: 13,
  content: `
<h3>প্রজেক্ট লক্ষ্য</h3>
<p>একটি ব্যাংক ঋণ অনুমোদন মডেল তৈরি করে SHAP ও LIME দিয়ে ব্যাখ্যা করব। EU AI Act অনুযায়ী ঋণ প্রত্যাখ্যানের কারণ ব্যাখ্যা করা আইনত বাধ্যতামূলক।</p>

<h3>মডেল তৈরি</h3>

<pre><code class="language-python">import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# সিনথেটিক ঋণ ডেটা
np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'বয়স': np.random.randint(22, 65, n),
    'আয়': np.random.randint(20000, 150000, n),
    'ঋণ_পরিমাণ': np.random.randint(50000, 1000000, n),
    'ক্রেডিট_স্কোর': np.random.randint(300, 850, n),
    'কর্মসংস্থান_বছর': np.random.randint(0, 30, n),
    'বিদ্যমান_ঋণ': np.random.randint(0, 5, n),
    'শিক্ষা': np.random.choice(['উচ্চ মাধ্যমিক', 'স্নাতক', 'স্নাতকোত্তর'], n),
})

# লেবেল: ক্রেডিট স্কোর ও আয়ের উপর ভিত্তি করে
df['অনুমোদন'] = (
    (df['ক্রেডিট_স্কোর'] > 600) &
    (df['আয'] > 40000) &
    (df['ঋণ_পরিমাণ'] < df['আয'] * 5)
).astype(int)

# এনকোডিং
le = LabelEncoder()
df['শিক্ষা_কোড'] = le.fit_transform(df['শিক্ষা'])

feature_cols = ['বয়স', 'আয়', 'ঋণ_পরিমাণ', 'ক্রেডিট_স্কোর',
                'কর্মসংস্থান_বছর', 'বিদ্যমান_ঋণ', 'শিক্ষা_কোড']
X = df[feature_cols].values
y = df['অনুমোদন'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# মডেল ট্রেনিং
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test),
      target_names=['প্রত্যাখ্যান', 'অনুমোদন']))
</code></pre>

<h3>SHAP বিশ্লেষণ</h3>

<pre><code class="language-python"># TreeSHAP ব্যাখ্যাকারী
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# শ্রেণী ১ (অনুমোদন) এর SHAP মান
shap_values_approved = shap_values[1]

# বৈশ্বিক গুরুত্ব — বার প্লট
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_approved, X_test,
                  feature_names=feature_cols,
                  plot_type="bar",
                  show=False)
plt.title("গ্লোবাল ফিচার গুরুত্ব (SHAP)")
plt.tight_layout()
plt.savefig("shap_global.png")
plt.show()

# বিস্তারিত বিতরণ — beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_approved, X_test,
                  feature_names=feature_cols,
                  show=False)
plt.title("SHAP বিতরণ প্লট")
plt.tight_layout()
plt.show()
</code></pre>

<h4>স্থানীয় ব্যাখ্যা — নির্দিষ্ট আবেদনকারী</h4>

<pre><code class="language-python"># একটি প্রত্যাখ্যাত আবেদনকারী ব্যাখ্যা
rejected_idx = np.where(y_test == 0)[0][0]
rejected_applicant = X_test[rejected_idx]

print("আবেদনকারীর তথ্য:")
for feat, val in zip(feature_cols, rejected_applicant):
    print(f"  {feat}: {val:.0f}")

print(f"\\nমডেলের সিদ্ধান্ত: প্রত্যাখ্যান")
print(f"অনুমোদনের সম্ভাবনা: {model.predict_proba([rejected_applicant])[0][1]:.2%}")

# SHAP force plot
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][rejected_idx],
    rejected_applicant,
    feature_names=feature_cols,
    matplotlib=True
)
</code></pre>

<h3>LIME বিশ্লেষণ</h3>

<pre><code class="language-python"># LIME ব্যাখ্যাকারী
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_cols,
    class_names=['প্রত্যাখ্যান', 'অনুমোদন'],
    mode='classification'
)

# নির্দিষ্ট আবেদনকারীর LIME ব্যাখ্যা
lime_exp = lime_explainer.explain_instance(
    rejected_applicant,
    model.predict_proba,
    num_features=7
)

lime_exp.show_in_notebook(show_all=False)

# HTML রিপোর্ট
lime_exp.save_to_file('lime_explanation.html')

# টেক্সট ব্যাখ্যা
print("\\nLIME ব্যাখ্যা:")
for feat, weight in lime_exp.as_list():
    direction = "বাড়িয়েছে" if weight > 0 else "কমিয়েছে"
    print(f"  {feat}: অনুমোদনের সম্ভাবনা {direction} ({weight:+.4f})")
</code></pre>

<h3>ব্যবহারকারী-বান্ধব ব্যাখ্যা</h3>

<pre><code class="language-python">def explain_decision(applicant_data, model, explainer, shap_values_all, feature_cols):
    """অ-প্রযুক্তিবিদদের জন্য সরল ব্যাখ্যা"""
    prediction = model.predict([applicant_data])[0]
    probability = model.predict_proba([applicant_data])[0]

    decision = "✅ অনুমোদিত" if prediction == 1 else "❌ প্রত্যাখ্যাত"
    print(f"সিদ্ধান্ত: {decision}")
    print(f"আত্মবিশ্বাস: {max(probability):.1%}")
    print()

    # SHAP দিয়ে কারণ খুঁজি
    idx = 0  # ব্যাখ্যার জন্য ইন্ডেক্স
    shap_val = shap_values_all[1][idx]

    top_factors = sorted(zip(feature_cols, shap_val),
                        key=lambda x: abs(x[1]), reverse=True)[:3]

    if prediction == 0:
        print("প্রত্যাখ্যানের প্রধান কারণ:")
        for feat, val in top_factors:
            if val < 0:
                print(f"  • {feat}: এই মান ঋণ অনুমোদনে বাধা দিচ্ছে")
    else:
        print("অনুমোদনের প্রধান কারণ:")
        for feat, val in top_factors:
            if val > 0:
                print(f"  • {feat}: এই মান ঋণ অনুমোদনে সাহায্য করেছে")

explain_decision(rejected_applicant, model, explainer,
                 shap_values, feature_cols)
</code></pre>

<h4>প্রধান শিক্ষা</h4>
<ul>
<li>ক্রেডিট স্কোর সবচেয়ে গুরুত্বপূর্ণ ফিচার (SHAP বিশ্লেষণে)</li>
<li>SHAP গ্লোবাল প্যাটার্ন দেখায়, LIME স্থানীয় সিদ্ধান্ত ব্যাখ্যা করে</li>
<li>দুটি পদ্ধতি একত্রে ব্যবহার সর্বোত্তম</li>
<li>নিয়ামক সম্মতির জন্য SHAP অডিট ট্রেইল রাখুন</li>
</ul>
`
};
