export const nb_7_spam_project = {
  title: "End-to-End Project: SMS Spam Detection — Naive Bayes",
  description: "SMS Spam Collection dataset দিয়ে সম্পূর্ণ Spam Detector তৈরি করো — EDA, preprocessing, Naive Bayes comparison, ও deployment-ready pipeline বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 14,
  slug: "nb-spam-detection-project",
  content: `
    <h3>১. Project Overview</h3>
    <p>UCI SMS Spam Collection Dataset ব্যবহার করে একটি spam detector তৈরি করবো যা বলবে একটি SMS spam নাকি ham (legitimate)।</p>
    <table>
      <thead><tr><th>Dataset Info</th><th>মান</th></tr></thead>
      <tbody>
        <tr><td>মোট SMS</td><td>5,574</td></tr>
        <tr><td>Spam</td><td>747 (13.4%)</td></tr>
        <tr><td>Ham</td><td>4,827 (86.6%)</td></tr>
        <tr><td>Target</td><td>binary: spam / ham</td></tr>
        <tr><td>Feature</td><td>raw SMS text</td></tr>
      </tbody>
    </table>

    <h3>২. ডেটা লোড ও EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Dataset লোড (UCI repository থেকে)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df  = pd.read_csv(url, sep='\\t', header=None, names=['label', 'text'])

print(df.shape)
print(df['label'].value_counts())
print("\\nExample spam:")
print(df[df['label']=='spam']['text'].iloc[0])
print("\\nExample ham:")
print(df[df['label']=='ham']['text'].iloc[0])

# Text statistics
df['text_length']  = df['text'].str.len()
df['word_count']   = df['text'].str.split().str.len()
df['has_number']   = df['text'].str.contains(r'\\d').astype(int)
df['has_currency'] = df['text'].str.contains(r'[£$€]').astype(int)
df['has_url']      = df['text'].str.contains(r'http|www|\.com').astype(int)
df['exclamations'] = df['text'].str.count('!')
df['capitals_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
)

# Spam vs Ham statistics
print("\\n=== Spam vs Ham Feature Comparison ===")
for col in ['text_length', 'word_count', 'has_number', 'has_currency',
            'has_url', 'exclamations', 'capitals_ratio']:
    spam_val = df[df['label']=='spam'][col].mean()
    ham_val  = df[df['label']=='ham'][col].mean()
    print(f"{col:20s}  spam={spam_val:.3f}  ham={ham_val:.3f}")

# Length distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[df['label']=='spam']['text_length'].hist(ax=axes[0], bins=30,
    color='crimson', alpha=0.7, label='spam')
df[df['label']=='ham']['text_length'].hist(ax=axes[0], bins=30,
    color='steelblue', alpha=0.7, label='ham')
axes[0].set_title('SMS Length Distribution')
axes[0].legend()

# Word frequency
spam_words = ' '.join(df[df['label']=='spam']['text']).lower().split()
ham_words  = ' '.join(df[df['label']=='ham']['text']).lower().split()
spam_top   = Counter(spam_words).most_common(15)
pd.DataFrame(spam_top, columns=['word','count']).plot.barh(
    x='word', y='count', ax=axes[1], color='crimson', legend=False)
axes[1].set_title('Top 15 Spam Words')
plt.tight_layout()
plt.show()
</code></pre>

    <h3>৩. Text Preprocessing</h3>
    <pre><code>import re
import string

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # URL remove
    text = re.sub(r'http\\S+|www\\S+', 'URL', text)
    # Phone number
    text = re.sub(r'\\b\\d{10,}\\b', 'PHONE', text)
    # Currency amount
    text = re.sub(r'[£$€]\\d+', 'MONEY', text)
    # Special characters
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    # Multiple spaces
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Test preprocessing
examples = [
    "FREE entry Win £1000 cash! Call 07712345678 now!!!",
    "Hey, are you coming to the meeting tomorrow?",
]
for ex in examples:
    print(f"Original:    {ex}")
    print(f"Preprocessed: {preprocess_text(ex)}")
    print()

# Apply to dataset
df['text_clean'] = df['text'].apply(preprocess_text)
df['label_bin']  = (df['label'] == 'spam').astype(int)
</code></pre>

    <h3>৪. Train/Test Split ও Feature Engineering</h3>
    <pre><code>from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import scipy.sparse as sp

X = df['text_clean']
y = df['label_bin']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train spam ratio: {y_train.mean():.3f}")
print(f"Test spam ratio:  {y_test.mean():.3f}")

# TF-IDF features
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"\\nTF-IDF features: {X_train_tfidf.shape[1]}")
</code></pre>

    <h3>৫. তিনটি Naive Bayes মডেল তুলনা</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, ConfusionMatrixDisplay)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

models = {
    'MultinomialNB (Count)': Pipeline([
        ('vec', CountVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)),
        ('clf', MultinomialNB(alpha=0.1)),
    ]),
    'MultinomialNB (TF-IDF)': Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                sublinear_tf=True, min_df=2)),
        ('clf', MultinomialNB(alpha=0.1)),
    ]),
    'BernoulliNB': Pipeline([
        ('vec', CountVectorizer(binary=True, max_features=5000)),
        ('clf', BernoulliNB(alpha=0.5)),
    ]),
    'ComplementNB': Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                sublinear_tf=True, min_df=2)),
        ('clf', ComplementNB(alpha=0.1)),
    ]),
    'Logistic Regression': Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                sublinear_tf=True, min_df=2)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    results[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'F1 (spam)': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_prob),
    }

import pandas as pd
results_df = pd.DataFrame(results).T.sort_values('F1 (spam)', ascending=False)
print(results_df.round(4))
</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Spam Detection Pipeline</text>
        <rect x="10" y="28" width="85" height="70" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="52" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">Raw SMS</text>
        <text x="52" y="64" text-anchor="middle" font-size="8" fill="#3b82f6">"FREE win"</text>
        <text x="52" y="78" text-anchor="middle" font-size="8" fill="#6b7280">input</text>
        <text x="102" y="66" font-size="11" fill="#94a3b8">→</text>
        <rect x="112" y="28" width="90" height="70" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="157" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">Preprocess</text>
        <text x="157" y="60" text-anchor="middle" font-size="8" fill="#d97706">lowercase</text>
        <text x="157" y="73" text-anchor="middle" font-size="8" fill="#d97706">clean text</text>
        <text x="209" y="66" font-size="11" fill="#94a3b8">→</text>
        <rect x="219" y="28" width="90" height="70" rx="5" fill="#fce7f3" stroke="#f9a8d4"/>
        <text x="264" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#9d174d">TF-IDF</text>
        <text x="264" y="60" text-anchor="middle" font-size="8" fill="#be185d">text →</text>
        <text x="264" y="73" text-anchor="middle" font-size="8" fill="#be185d">features</text>
        <text x="316" y="66" font-size="11" fill="#94a3b8">→</text>
        <rect x="326" y="28" width="85" height="70" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="368" y="47" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">NB Model</text>
        <text x="368" y="63" text-anchor="middle" font-size="11" fill="#16a34a">SPAM</text>
        <text x="368" y="79" text-anchor="middle" font-size="8" fill="#6b7280">or ham</text>
        <text x="416" y="66" font-size="11" fill="#94a3b8">✓</text>
      </svg>
    </div>

    <h3>৬. সেরা মডেল Hyperparameter Tuning</h3>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV
import numpy as np

best_pipe = Pipeline([
    ('vec', TfidfVectorizer(sublinear_tf=True)),
    ('clf', MultinomialNB()),
])

param_dist = {
    'vec__max_features':  [3000, 5000, 10000],
    'vec__ngram_range':   [(1,1), (1,2), (1,3)],
    'vec__min_df':        [1, 2, 3],
    'vec__max_df':        [0.9, 0.95, 1.0],
    'clf__alpha':         np.logspace(-3, 1, 20),
}

search = RandomizedSearchCV(
    best_pipe, param_dist,
    n_iter=30, cv=5,
    scoring='f1',
    random_state=42, n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

print(f"সেরা params: {search.best_params_}")
print(f"CV F1:       {search.best_score_:.4f}")

final_model = search.best_estimator_
y_final = final_model.predict(X_test)

print("\\n=== Final Model Performance ===")
print(classification_report(y_test, y_final, target_names=['ham', 'spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_final)
ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam']).plot(colorbar=False)
plt.title('Spam Detector — Confusion Matrix')
plt.show()
</code></pre>

    <h3>৭. Error Analysis — ভুল prediction দেখো</h3>
    <pre><code># Misclassified examples বিশ্লেষণ
y_pred_final = final_model.predict(X_test)
X_test_arr   = X_test.reset_index(drop=True)
y_test_arr   = y_test.reset_index(drop=True)

errors = [(X_test_arr[i], y_test_arr[i], y_pred_final[i])
          for i in range(len(y_test_arr))
          if y_test_arr[i] != y_pred_final[i]]

print(f"মোট ভুল: {len(errors)}")

# False Positives: ham → spam বলেছে
fp = [(t, a, p) for t, a, p in errors if a == 0 and p == 1]
print(f"\\nFalse Positives (ham → spam): {len(fp)}")
for text, actual, pred in fp[:3]:
    print(f"  '{text[:60]}'")

# False Negatives: spam → ham বলেছে
fn = [(t, a, p) for t, a, p in errors if a == 1 and p == 0]
print(f"\\nFalse Negatives (spam → ham): {len(fn)}")
for text, actual, pred in fn[:3]:
    print(f"  '{text[:60]}'")
</code></pre>

    <h3>৮. Production-ready Prediction Function</h3>
    <pre><code>import pickle

# মডেল save করো
with open('spam_detector.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Load ও predict
def predict_spam(sms_text: str, model_path: str = 'spam_detector.pkl'):
    """
    Returns: dict with 'label', 'confidence', 'is_spam'
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    clean = preprocess_text(sms_text)
    prob  = model.predict_proba([clean])[0]
    pred  = model.predict([clean])[0]

    return {
        'label':      'SPAM' if pred == 1 else 'HAM',
        'is_spam':    bool(pred),
        'confidence': float(max(prob)),
        'p_spam':     float(prob[1]),
        'p_ham':      float(prob[0]),
    }

# Test
test_messages = [
    "WINNER!! You've been selected for a FREE prize worth £500!",
    "Hey, can we reschedule tomorrow's meeting to 3pm?",
    "Congratulations! You have won a 1 week FREE holiday!",
    "I'll be home by 7. Should I pick up dinner?",
]

print("=== Spam Detector Results ===")
for msg in test_messages:
    result = predict_spam(msg)
    bar = '█' * int(result['p_spam'] * 20)
    print(f"\\n'{msg[:50]}...'")
    print(f"  → {result['label']:4s}  P(spam)={result['p_spam']:.3f}  [{bar}]")
</code></pre>

    <h3>সারসংক্ষেপ — মডেল Performance তুলনা</h3>
    <table>
      <thead><tr><th>মডেল</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>
        <tr><td>MultinomialNB (Count)</td><td>~98%</td><td>~97%</td><td>~93%</td><td>~95%</td></tr>
        <tr><td>MultinomialNB (TF-IDF)</td><td>~98%</td><td>~98%</td><td>~94%</td><td>~96%</td></tr>
        <tr><td>BernoulliNB</td><td>~97%</td><td>~95%</td><td>~91%</td><td>~93%</td></tr>
        <tr><td>ComplementNB</td><td>~98%</td><td>~97%</td><td>~95%</td><td>~96%</td></tr>
        <tr><td>Logistic Regression</td><td>~99%</td><td>~99%</td><td>~96%</td><td>~97%</td></tr>
      </tbody>
    </table>

    <table style="margin-top:1rem;">
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Dataset</td><td>5574 SMS, 13% spam — imbalanced</td></tr>
        <tr><td>Preprocessing</td><td>lowercase, URL/phone/money normalization</td></tr>
        <tr><td>সেরা NB</td><td>ComplementNB বা MultinomialNB (TF-IDF)</td></tr>
        <tr><td>F1 score</td><td>accuracy-র চেয়ে ভালো metric imbalanced-এ</td></tr>
        <tr><td>False Negative</td><td>spam miss করা বেশি costly — Recall গুরুত্বপূর্ণ</td></tr>
        <tr><td>Production</td><td>pickle দিয়ে save, preprocess function সহ deploy</td></tr>
      </tbody>
    </table>
  `,
};
