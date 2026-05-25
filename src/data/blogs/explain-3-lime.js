export const explain_3_lime = {
  slug: 'explain-3-lime',
  title: 'LIME: স্থানীয় মডেল দিয়ে ব্যাখ্যা',
  description: 'LIME কীভাবে ইনপুট পার্টার্ব করে স্থানীয় লিনিয়ার মডেল ফিট করে, টেক্সট/সারণি/ছবি ভবিষ্যদ্বাণী ব্যাখ্যা করে এবং lime লাইব্রেরি দিয়ে বাস্তব প্রয়োগ।',
  date: 'মে ২০২৬',
  category: 'মডেল ব্যাখ্যাযোগ্যতা',
  readTime: 12,
  content: `
    <h3>১. LIME কী এবং কেন দরকার?</h3>
    <p>
      <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> ২০১৬ সালে Ribeiro, Singh ও Guestrin প্রস্তাব করেন। এর মূল ধারণা সহজ: যদিও একটি জটিল মডেল বৈশ্বিকভাবে (globally) বোঝা কঠিন, কিন্তু একটি নির্দিষ্ট ইনপুটের কাছাকাছি স্থানীয়ভাবে (locally) এটি প্রায় লিনিয়ার আচরণ করে।
    </p>
    <p>
      LIME সেই স্থানীয় আচরণকে একটি সহজ <strong>লিনিয়ার মডেল</strong> দিয়ে approximate করে এবং সেই লিনিয়ার মডেলের সহগগুলোই ব্যাখ্যা হিসেবে উপস্থাপন করে।
    </p>
    <p>
      LIME-এর তিনটি মূল বৈশিষ্ট্য:
    </p>
    <p>
      <strong>Local:</strong> শুধু একটি নির্দিষ্ট ভবিষ্যদ্বাণীর কাছাকাছি কাজ করে।<br/>
      <strong>Interpretable:</strong> আউটপুট হিসেবে সহজ লিনিয়ার মডেল দেয়।<br/>
      <strong>Model-agnostic:</strong> যেকোনো মডেলে কাজ করে — শুধু predict function দরকার।
    </p>

    <h3>২. LIME-এর কার্যপদ্ধতি</h3>
    <p>
      LIME নিম্নলিখিত ধাপে কাজ করে:
    </p>
    <p>
      <strong>ধাপ ১ — Perturbation:</strong> মূল ইনপুটের কাছাকাছি অনেকগুলো পার্টার্বড স্যাম্পল তৈরি করা হয়। সারণি ডেটায় ফিচারের মান পরিবর্তন করা হয়, টেক্সটে শব্দ বাদ দেওয়া হয়, ছবিতে সুপারপিক্সেল ঢেকে দেওয়া হয়।
    </p>
    <p>
      <strong>ধাপ ২ — Black-box Predictions:</strong> পার্টার্বড স্যাম্পলগুলো মূল মডেলে পাঠিয়ে ভবিষ্যদ্বাণী নেওয়া হয়।
    </p>
    <p>
      <strong>ধাপ ৩ — Weighting:</strong> মূল ইনপুট থেকে যত কাছে, তত বেশি ওজন।
    </p>
    <p>
      <strong>ধাপ ৪ — Local Linear Model:</strong> ওজনযুক্ত ডেটায় একটি লিনিয়ার মডেল ফিট করা হয়।
    </p>
    <p>
      <strong>ধাপ ৫ — Explanation:</strong> লিনিয়ার মডেলের সহগ = ব্যাখ্যা।
    </p>
    <pre><code>import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances

def simple_lime_tabular(model, instance, X_train, n_samples=1000, kernel_width=0.25):
    """
    সারণি ডেটার জন্য সরলীকৃত LIME বাস্তবায়ন
    """
    n_features = instance.shape[0]

    # ধাপ ১: Perturbation — random samples তৈরি
    perturbed = np.random.normal(0, 1, (n_samples, n_features))
    # মূল ডেটার scale অনুযায়ী adjust
    perturbed = perturbed * X_train.std(axis=0) + X_train.mean(axis=0)

    # ধাপ ২: Black-box predictions
    predictions = model.predict_proba(perturbed)[:, 1]

    # ধাপ ৩: Kernel weighting (exponential kernel)
    distances = np.sqrt(np.sum((perturbed - instance) ** 2, axis=1))
    kernel = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

    # ধাপ ৪: Weighted Ridge Regression
    local_model = Ridge(alpha=1.0)
    local_model.fit(perturbed, predictions, sample_weight=kernel)

    # ধাপ ৫: Coefficients = Explanation
    return dict(zip(range(n_features), local_model.coef_))

print("Simple LIME implementation ready.")</code></pre>

    <h3>৩. lime লাইব্রেরি দিয়ে সারণি ডেটা ব্যাখ্যা</h3>
    <p>
      বাস্তবে আমরা <code>lime</code> লাইব্রেরি ব্যবহার করব যা অনেক বেশি robust এবং বিভিন্ন ডেটা টাইপ সাপোর্ট করে।
    </p>
    <pre><code>import numpy as np
import pandas as pd
from lime import lime_tabular
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# ডেটা প্রস্তুত
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# Gradient Boosting মডেল
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)
print(f"Test Accuracy: {gbc.score(X_test, y_test):.4f}")

# LIME Tabular Explainer তৈরি
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=data.feature_names,
    class_names=data.target_names,   # ['malignant', 'benign']
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# একটি নির্দিষ্ট ভবিষ্যদ্বাণীর ব্যাখ্যা
sample_idx = 10
instance = X_test[sample_idx]
true_label = y_test[sample_idx]
predicted = gbc.predict([instance])[0]
predicted_prob = gbc.predict_proba([instance])[0]

print(f"\nSample {sample_idx}:")
print(f"  True label:  {data.target_names[true_label]}")
print(f"  Predicted:   {data.target_names[predicted]}")
print(f"  Probability: malignant={predicted_prob[0]:.3f}, benign={predicted_prob[1]:.3f}")

# LIME ব্যাখ্যা তৈরি
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=gbc.predict_proba,
    num_features=10,           # শীর্ষ ১০টি ফিচার
    num_samples=1000           # perturbation samples
)

# ব্যাখ্যা দেখুন
print("\nLIME Explanation (top features):")
for feature, weight in explanation.as_list():
    direction = "pushes UP" if weight > 0 else "pushes DOWN"
    print(f"  {feature:50} weight={weight:+.4f}  ({direction})")</code></pre>

    <h3>৪. টেক্সট ভবিষ্যদ্বাণী ব্যাখ্যা</h3>
    <p>
      LIME টেক্সট ক্লাসিফিকেশনের জন্যও চমৎকার কাজ করে। এখানে শব্দগুলোকে অন/অফ করে (একটি শব্দ বাদ দিলে ভবিষ্যদ্বাণী কতটা পরিবর্তন হয়) ব্যাখ্যা তৈরি করা হয়।
    </p>
    <pre><code>from lime import lime_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups

# সীমিত ডেটা (দ্রুত ডেমোর জন্য)
categories = ['sci.med', 'sci.space']
newsgroups = fetch_20newsgroups(
    subset='train', categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# TF-IDF + Logistic Regression পাইপলাইন
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
text_pipeline.fit(newsgroups.data, newsgroups.target)

# Test accuracy
from sklearn.datasets import fetch_20newsgroups
test_data = fetch_20newsgroups(
    subset='test', categories=categories,
    remove=('headers', 'footers', 'quotes')
)
print(f"Test Accuracy: {text_pipeline.score(test_data.data, test_data.target):.4f}")

# LIME Text Explainer
text_explainer = lime_text.LimeTextExplainer(
    class_names=categories,
    split_expression=r'\W+',   # শব্দ বিভাজক
    random_state=42
)

# একটি ডকুমেন্টের ব্যাখ্যা
doc_idx = 5
doc = test_data.data[doc_idx]
true_cat = categories[test_data.target[doc_idx]]
pred_proba = text_pipeline.predict_proba([doc])[0]

print(f"\nDocument {doc_idx} (first 150 chars):")
print(f"  {doc[:150]}...")
print(f"  True category: {true_cat}")
print(f"  P(sci.med)={pred_proba[0]:.3f}, P(sci.space)={pred_proba[1]:.3f}")

text_exp = text_explainer.explain_instance(
    text_instance=doc,
    classifier_fn=text_pipeline.predict_proba,
    num_features=8,
    num_samples=500
)

print("\nLIME Text Explanation (top words):")
for word, weight in text_exp.as_list():
    direction = "sci.space" if weight > 0 else "sci.med"
    print(f"  '{word}' -> {direction} (weight={weight:+.4f})")</code></pre>

    <h3>৫. ছবি ভবিষ্যদ্বাণী ব্যাখ্যা</h3>
    <p>
      ছবির জন্য LIME <strong>superpixel</strong> ব্যবহার করে। ছবিকে অর্থপূর্ণ অঞ্চলে ভাগ করে, একটি একটি করে অঞ্চল লুকিয়ে দেখা হয় কোন অঞ্চল মডেলের সিদ্ধান্তে সবচেয়ে বেশি প্রভাব ফেলছে।
    </p>
    <pre><code>from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

# Pre-trained MobileNet ব্যবহার
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image

# মডেল লোড
mobilenet = MobileNetV2(weights='imagenet')
print("MobileNetV2 loaded.")

def predict_fn(images):
    """LIME-এর জন্য predict function"""
    # images: (N, H, W, 3), uint8
    processed = preprocess_input(images.astype(np.float32))
    return mobilenet.predict(processed, verbose=0)

# একটি ছবি লোড ও resize
img_path = 'sample_cat.jpg'  # আপনার ছবির path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# MobileNet ভবিষ্যদ্বাণী
pred = mobilenet.predict(
    preprocess_input(np.expand_dims(img_array, 0)), verbose=0
)
top_preds = decode_predictions(pred, top=3)[0]
print("\nTop predictions:")
for _, label, prob in top_preds:
    print(f"  {label}: {prob:.4f}")

# LIME Image Explainer
image_explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = image_explainer.explain_instance(
    img_array.astype(np.double),
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=300  # বেশি = বেশি নির্ভুল কিন্তু ধীর
)

# গুরুত্বপূর্ণ অঞ্চল দেখুন
from skimage.io import imsave
temp_img, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)
marked = mark_boundaries(temp_img / 255.0, mask)
imsave('lime_image_explanation.png', (marked * 255).astype(np.uint8))
print("Saved: lime_image_explanation.png")</code></pre>

    <h3>৬. LIME-এর সীমাবদ্ধতা</h3>
    <p>
      LIME-এর কিছু গুরুত্বপূর্ণ সীমাবদ্ধতা আছে:
    </p>
    <p>
      <strong>অস্থিরতা (Instability):</strong> একই ইনপুটে বারবার চালালে ব্যাখ্যা কিছুটা ভিন্ন হতে পারে (কারণ perturbation র‍্যান্ডম)।<br/>
      <strong>সীমানা নির্ধারণ:</strong> "স্থানীয়" মানে কতটুকু? kernel_width নির্বাচন ফলাফলকে প্রভাবিত করে।<br/>
      <strong>উচ্চমাত্রার ডেটা:</strong> অনেক ফিচার থাকলে perturbation কম কার্যকর।
    </p>
    <pre><code># LIME-এর stability পরীক্ষা
from lime import lime_tabular

stability_results = []
for seed in range(5):
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=gbc.predict_proba,
        num_features=5,
        num_samples=500,
        random_state=seed
    )
    top_features = [feat for feat, _ in exp.as_list()[:5]]
    stability_results.append(top_features)

print("LIME Stability Check (top 5 features across 5 runs):")
for i, result in enumerate(stability_results):
    print(f"  Run {i+1}: {result}")

# Feature overlap
all_features = [set(r) for r in stability_results]
common = all_features[0].intersection(*all_features[1:])
print(f"\nConsistently top features: {common}")</code></pre>

    <h3>৭. LIME বনাম SHAP: কোনটি বেছে নেবেন?</h3>
    <p>
      <strong>LIME বেছে নিন যখন:</strong> দ্রুত ব্যাখ্যা দরকার, টেক্সট বা ছবির ব্যাখ্যা দরকার, সহজ কোড বাস্তবায়ন দরকার।
    </p>
    <p>
      <strong>SHAP বেছে নিন যখন:</strong> তাত্ত্বিক শক্তিশালী ব্যাখ্যা দরকার, ট্রি মডেল ব্যবহার করছেন (TreeSHAP দ্রুত), গ্লোবাল এবং লোকাল উভয় ব্যাখ্যা দরকার, consistency গ্যারান্টি দরকার।
    </p>
    <pre><code># তুলনামূলক সারসংক্ষেপ
comparison = {
    'Method':              ['LIME',           'SHAP (TreeSHAP)',  'SHAP (Kernel)'],
    'Model-agnostic':      ['Yes',            'No (tree only)',   'Yes'],
    'Speed':               ['Fast',           'Very Fast',        'Slow'],
    'Stability':           ['Medium',         'High',             'High'],
    'Global Explanation':  ['No',             'Yes',              'Yes'],
    'Text/Image Support':  ['Yes',            'Limited',          'No'],
    'Theoretical basis':   ['Approximation',  'Shapley values',   'Shapley values'],
}

import pandas as pd
df = pd.DataFrame(comparison)
df = df.set_index('Method')
print(df.to_string())</code></pre>

    <h3>৮. সারসংক্ষেপ</h3>
    <p>
      LIME একটি শক্তিশালী এবং নমনীয় ব্যাখ্যা পদ্ধতি। এর মডেল-অ্যাগনস্টিক প্রকৃতি এবং টেক্সট/ছবি সাপোর্ট এটিকে বিশেষভাবে NLP ও কম্পিউটার ভিশনে মূল্যবান করে তোলে। তবে স্থিরতার জন্য SHAP-এর তুলনায় কিছুটা দুর্বল। বাস্তব প্রজেক্টে প্রায়ই দুটি পদ্ধতি একসাথে ব্যবহার করা হয়। পরবর্তী পর্বে আমরা Permutation Importance এবং PDP দেখব।
    </p>
  `
};
