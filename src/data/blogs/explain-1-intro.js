export const explain_1_intro = {
  slug: 'explain-1-intro',
  title: 'মডেল ব্যাখ্যাযোগ্যতা: কেন দরকার?',
  description: 'ব্ল্যাক বক্স বনাম ইন্টারপ্রিটেবল মডেল, গ্লোবাল ও লোকাল ব্যাখ্যা, EU AI Act এবং SHAP/LIME/Permutation Importance-এর সংক্ষিপ্ত পরিচিতি।',
  date: 'মে ২০২৬',
  category: 'মডেল ব্যাখ্যাযোগ্যতা',
  readTime: 10,
  content: `
    <h3>১. মডেল ব্যাখ্যাযোগ্যতা কী?</h3>
    <p>
      মেশিন লার্নিং মডেল আজকাল ঋণ অনুমোদন, রোগ নির্ণয়, চাকরির বাছাই থেকে শুরু করে অনেক গুরুত্বপূর্ণ সিদ্ধান্ত নিচ্ছে। কিন্তু একটি প্রশ্ন সবার মনে জাগে — মডেল কেন এই সিদ্ধান্ত নিল? কোন ফিচার বা তথ্য মডেলের সিদ্ধান্তকে সবচেয়ে বেশি প্রভাবিত করল?
    </p>
    <p>
      <strong>মডেল ব্যাখ্যাযোগ্যতা (Model Explainability)</strong> বা <strong>XAI (Explainable Artificial Intelligence)</strong> হলো সেই পদ্ধতি এবং কৌশলের সমষ্টি, যা মডেলের সিদ্ধান্ত গ্রহণ প্রক্রিয়াকে মানুষের বোধগম্য করে তোলে। এটি শুধু প্রযুক্তিগত বিষয় নয় — এটি নৈতিকতা, আইন এবং বিশ্বাসের প্রশ্নও।
    </p>

    <h3>২. ব্ল্যাক বক্স বনাম ইন্টারপ্রিটেবল মডেল</h3>
    <p>
      মেশিন লার্নিং মডেলগুলোকে মোটামুটি দুই ভাগে ভাগ করা যায়:
    </p>
    <h4>ইন্টারপ্রিটেবল (Interpretable) মডেল</h4>
    <p>
      এই মডেলগুলো সরাসরি বোঝা যায়। যেমন — <strong>লিনিয়ার রিগ্রেশন</strong>-এ প্রতিটি ফিচারের একটি সহগ (coefficient) থাকে যা সরাসরি বলে দেয় সেই ফিচার আউটপুটকে কতটা প্রভাবিত করে। <strong>ডিসিশন ট্রি</strong>-তে if-else নিয়মের মাধ্যমে সিদ্ধান্ত নেওয়া হয়, যা সহজেই ট্রেস করা যায়।
    </p>
    <p>
      এই মডেলগুলো স্বচ্ছ, কিন্তু জটিল ডেটায় নির্ভুলতা কম হতে পারে। এটাকে বলা হয় <strong>accuracy-interpretability trade-off</strong>।
    </p>
    <h4>ব্ল্যাক বক্স (Black Box) মডেল</h4>
    <p>
      <strong>নিউরাল নেটওয়ার্ক</strong>, <strong>র‍্যান্ডম ফরেস্ট</strong>, <strong>গ্রেডিয়েন্ট বুস্টিং</strong> — এগুলো অত্যন্ত শক্তিশালী এবং নির্ভুল, কিন্তু এদের ভেতরের কার্যকলাপ সরাসরি বোঝা কঠিন। একটি ডিপ নিউরাল নেটওয়ার্কে লক্ষ লক্ষ প্যারামিটার থাকে — কোন প্যারামিটার কোন সিদ্ধান্তে কতটা ভূমিকা রাখল তা সরাসরি বলা প্রায় অসম্ভব।
    </p>
    <pre><code># মডেলের তুলনা: interpretability vs accuracy
models = {
    'Linear Regression':    {'interpretability': 'High',   'accuracy': 'Low-Medium'},
    'Decision Tree':        {'interpretability': 'High',   'accuracy': 'Medium'},
    'Logistic Regression':  {'interpretability': 'High',   'accuracy': 'Medium'},
    'Random Forest':        {'interpretability': 'Low',    'accuracy': 'High'},
    'Gradient Boosting':    {'interpretability': 'Low',    'accuracy': 'High'},
    'Neural Network':       {'interpretability': 'Very Low','accuracy': 'Very High'},
}

for model, props in models.items():
    print(f"{model:25} | Interpretability: {props['interpretability']:9} | Accuracy: {props['accuracy']}")</code></pre>

    <h3>৩. গ্লোবাল বনাম লোকাল ব্যাখ্যা</h3>
    <p>
      মডেল ব্যাখ্যার দুটি স্তর আছে:
    </p>
    <h4>গ্লোবাল ব্যাখ্যা (Global Explanation)</h4>
    <p>
      পুরো মডেল কীভাবে কাজ করে তা বোঝানো। যেমন — সামগ্রিকভাবে কোন ফিচারগুলো মডেলের সিদ্ধান্তে সবচেয়ে বেশি প্রভাব ফেলে? <strong>Permutation Feature Importance</strong> এবং <strong>SHAP Summary Plots</strong> গ্লোবাল ব্যাখ্যার উদাহরণ।
    </p>
    <h4>লোকাল ব্যাখ্যা (Local Explanation)</h4>
    <p>
      একটি নির্দিষ্ট ভবিষ্যদ্বাণী কেন করা হলো তা বোঝানো। যেমন — এই নির্দিষ্ট গ্রাহকের ঋণ কেন প্রত্যাখ্যান করা হলো? <strong>SHAP Force Plots</strong> এবং <strong>LIME</strong> লোকাল ব্যাখ্যার উদাহরণ।
    </p>
    <pre><code>import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# ডেটা লোড
data = load_breast_cancer()
X, y = data.data, data.target

# মডেল তৈরি
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# SHAP দিয়ে গ্লোবাল ব্যাখ্যা
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# Global: কোন ফিচার সবচেয়ে গুরুত্বপূর্ণ?
print("Global Feature Importance (mean |SHAP|):")
mean_shap = np.abs(shap_values[1]).mean(axis=0)
for i, name in enumerate(data.feature_names):
    print(f"  {name:35}: {mean_shap[i]:.4f}")

# Local: একটি নির্দিষ্ট ভবিষ্যদ্বাণীর ব্যাখ্যা
sample_idx = 5
print(f"\nLocal explanation for sample {sample_idx}:")
print(f"  Predicted class: {clf.predict([X[sample_idx]])[0]}")
print(f"  Top contributing features:")
local_shap = shap_values[1][sample_idx]
top_idx = np.argsort(np.abs(local_shap))[::-1][:5]
for idx in top_idx:
    print(f"    {data.feature_names[idx]:35}: {local_shap[idx]:.4f}")</code></pre>

    <h3>৪. মডেল-স্পেসিফিক বনাম মডেল-অ্যাগনস্টিক পদ্ধতি</h3>
    <h4>মডেল-স্পেসিফিক (Model-Specific)</h4>
    <p>
      নির্দিষ্ট ধরনের মডেলের জন্য ডিজাইন করা। যেমন — <strong>TreeSHAP</strong> শুধু ট্রি-ভিত্তিক মডেলের জন্য (Random Forest, XGBoost)। <strong>Integrated Gradients</strong> নিউরাল নেটওয়ার্কের জন্য। এগুলো দ্রুত কিন্তু নির্দিষ্ট মডেলেই সীমাবদ্ধ।
    </p>
    <h4>মডেল-অ্যাগনস্টিক (Model-Agnostic)</h4>
    <p>
      যেকোনো মডেলে প্রযোগ করা যায়। <strong>LIME</strong> এবং <strong>KernelSHAP</strong> মডেল-অ্যাগনস্টিক। এরা মডেলকে একটি ব্ল্যাক বক্স হিসেবে দেখে এবং শুধু ইনপুট-আউটপুট সম্পর্ক বিশ্লেষণ করে। নমনীয় কিন্তু কিছুটা ধীরগতি।
    </p>

    <h3>৫. EU AI Act এবং নিয়ন্ত্রক চাপ</h3>
    <p>
      ২০২৪ সালে কার্যকর হওয়া <strong>EU AI Act</strong> বিশ্বের প্রথম ব্যাপক AI আইন। এই আইনে "high-risk" AI সিস্টেমগুলোর জন্য ব্যাখ্যাযোগ্যতা বাধ্যতামূলক করা হয়েছে। ঋণ সিদ্ধান্ত, নিয়োগ প্রক্রিয়া, চিকিৎসা নির্ণয় — এই ক্ষেত্রগুলোতে AI মডেলকে অবশ্যই ব্যাখ্যা করতে পারতে হবে।
    </p>
    <p>
      এছাড়া <strong>GDPR (General Data Protection Regulation)</strong>-এ "right to explanation" অন্তর্ভুক্ত আছে — কোনো ব্যক্তি স্বয়ংক্রিয় সিদ্ধান্তের বিরুদ্ধে ব্যাখ্যা চাইতে পারেন।
    </p>
    <pre><code># Risk-based classification (EU AI Act)
risk_categories = {
    'Unacceptable Risk (Banned)': [
        'Social scoring systems',
        'Real-time biometric surveillance in public',
        'Subliminal manipulation',
    ],
    'High Risk (Requires Explainability)': [
        'Credit scoring',
        'Employment recruitment',
        'Medical diagnosis assistance',
        'Education assessment',
        'Law enforcement predictive tools',
    ],
    'Limited Risk': [
        'Chatbots (must disclose AI)',
        'Deepfake generation',
    ],
    'Minimal Risk': [
        'Spam filters',
        'AI in video games',
        'Recommendation systems',
    ],
}

for category, examples in risk_categories.items():
    print(f"\n{category}:")
    for ex in examples:
        print(f"  - {ex}")</code></pre>

    <h3>৬. AI-এর উপর বিশ্বাস কেন গুরুত্বপূর্ণ</h3>
    <p>
      একটি মডেল যতই নির্ভুল হোক না কেন, যদি ব্যবহারকারীরা এটির সিদ্ধান্ত বুঝতে না পারেন, তাহলে তারা এটি গ্রহণ করবেন না। একজন ডাক্তার AI-এর রোগ নির্ণয় তখনই বিশ্বাস করবেন যখন তিনি বুঝতে পারবেন কোন লক্ষণগুলো সেই রোগ নির্দেশ করছে।
    </p>
    <p>
      ব্যাখ্যাযোগ্যতা আরও একটি গুরুত্বপূর্ণ কাজ করে — <strong>bias detection</strong>। যদি একটি ঋণ মডেল বর্ণ বা লিঙ্গের ভিত্তিতে বৈষম্য করে, SHAP বা LIME দিয়ে সেটি ধরা যায়।
    </p>

    <h3>৭. প্রধান XAI পদ্ধতিগুলোর সংক্ষিপ্ত পরিচয়</h3>
    <h4>SHAP (SHapley Additive exPlanations)</h4>
    <p>
      গেম থিওরির শ্যাপলি ভ্যালু ব্যবহার করে প্রতিটি ফিচারের অবদান হিসাব করে। তাত্ত্বিকভাবে সবচেয়ে শক্তিশালী এবং সামঞ্জস্যপূর্ণ। TreeSHAP ট্রি মডেলের জন্য অত্যন্ত দ্রুত। সামনে আমরা বিস্তারিত দেখব।
    </p>
    <h4>LIME (Local Interpretable Model-agnostic Explanations)</h4>
    <p>
      একটি জটিল মডেলের কাছে স্থানীয়ভাবে (locally) একটি সহজ লিনিয়ার মডেল ফিট করে। টেক্সট, ছবি এবং সারণি ডেটায় কাজ করে। মডেল-অ্যাগনস্টিক।
    </p>
    <h4>Permutation Feature Importance</h4>
    <p>
      একটি ফিচারের মান এলোমেলো করে দিলে মডেলের performance কতটা কমে তা পরিমাপ করে। সহজ এবং কার্যকর, sklearn-এ বিল্ট-ইন।
    </p>
    <h4>Partial Dependence Plots (PDP)</h4>
    <p>
      একটি ফিচারের মান পরিবর্তন করলে মডেলের আউটপুট কীভাবে পরিবর্তিত হয় তা ভিজুয়ালি দেখায়। গ্লোবাল ব্যাখ্যার জন্য চমৎকার।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# মডেল তৈরি
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ১. Built-in Feature Importance (model-specific)
print("Built-in Feature Importance (top 5):")
fi = pd.Series(model.feature_importances_, index=data.feature_names)
print(fi.nlargest(5).to_string())

# ২. Permutation Importance (model-agnostic)
print("\nPermutation Importance (top 5):")
perm_imp = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42
)
perm_series = pd.Series(
    perm_imp.importances_mean, index=data.feature_names
)
print(perm_series.nlargest(5).to_string())</code></pre>

    <h3>৮. কখন কোন পদ্ধতি ব্যবহার করবেন</h3>
    <p>
      নিচের গাইডলাইন অনুসরণ করুন:
    </p>
    <p>
      <strong>ট্রি-ভিত্তিক মডেল (RF, XGBoost, LightGBM)?</strong> — TreeSHAP সেরা বিকল্প। দ্রুত এবং সঠিক।<br/>
      <strong>নিউরাল নেটওয়ার্ক?</strong> — KernelSHAP বা LIME ব্যবহার করুন, অথবা Integrated Gradients।<br/>
      <strong>একটি নির্দিষ্ট ভবিষ্যদ্বাণী ব্যাখ্যা করতে হবে?</strong> — LIME বা SHAP Force Plot।<br/>
      <strong>সামগ্রিক মডেল আচরণ বুঝতে হবে?</strong> — SHAP Summary Plot বা Permutation Importance।<br/>
      <strong>ফিচারের প্রভাব ভিজুয়ালি দেখতে হবে?</strong> — Partial Dependence Plot (PDP)।
    </p>

    <h3>৯. সারসংক্ষেপ</h3>
    <p>
      মডেল ব্যাখ্যাযোগ্যতা আর ঐচ্ছিক নয় — এটি আধুনিক ML প্র্যাকটিসের অপরিহার্য অংশ। EU AI Act, GDPR এবং নৈতিক AI-এর চাপে ব্যাখ্যাযোগ্যতার গুরুত্ব ক্রমশ বাড়ছে। SHAP, LIME এবং Permutation Importance — এই তিনটি পদ্ধতি শিখলে আপনি যেকোনো মডেলকে ব্যাখ্যা করতে পারবেন। পরবর্তী পর্বে আমরা SHAP বিস্তারিতভাবে দেখব।
    </p>
  `
};
