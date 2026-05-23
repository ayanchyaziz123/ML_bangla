export const nb_5_text_classification = {
  title: "Text Classification — TF-IDF ও CountVectorizer দিয়ে NLP",
  description: "CountVectorizer, TF-IDF, n-gram, pipeline — Naive Bayes দিয়ে বাংলা ও ইংরেজি text classification-এর সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 11,
  slug: "nb-text-classification-tfidf",
  content: `
    <h3>১. Text → Numbers: কেন দরকার?</h3>
    <p>Machine learning মডেল সংখ্যা বোঝে, text নয়। তাই text-কে numeric vector-এ রূপান্তর করতে হয়। তিনটি প্রধান পদ্ধতি:</p>
    <pre><code>১. Bag of Words (CountVectorizer)
   "I love python" → [0, 1, 0, 1, 0, 1, ...]  (প্রতিটি শব্দের count)

২. Binary Bag of Words
   "I love python" → [0, 1, 0, 1, 0, 1, ...]  (0 বা 1 শুধু)

৩. TF-IDF (Term Frequency - Inverse Document Frequency)
   TF(t,d)  = (t শব্দটি d document-এ কতবার) / (d-তে মোট শব্দ)
   IDF(t)   = log(মোট document / t শব্দ যত document-এ আছে)
   TF-IDF   = TF × IDF

   সাধারণ শব্দ (the, is, a) → low IDF → কম গুরুত্ব
   বিরল কিন্তু গুরুত্বপূর্ণ শব্দ → high IDF → বেশি গুরুত্ব</code></pre>

    <h3>২. CountVectorizer বিস্তারিত</h3>
    <pre><code>from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

docs = [
    "I love machine learning",
    "machine learning is great",
    "python is awesome for machine learning",
    "deep learning and machine learning differ",
]

# Basic CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(docs)

# Vocabulary দেখো
print("Vocabulary:")
vocab = cv.get_feature_names_out()
print(vocab)

# Document-term matrix
df = pd.DataFrame(X.toarray(), columns=vocab)
print("\\nDocument-Term Matrix:")
print(df)

# n-gram: unigram + bigram
cv_ngram = CountVectorizer(ngram_range=(1, 2))
X_ngram = cv_ngram.fit_transform(docs)
print(f"\\nUnigram features: {len(cv.vocabulary_)}")
print(f"Bigram features:  {len(cv_ngram.vocabulary_)}")

# min_df, max_df: rare/common শব্দ ফেলো
cv_filtered = CountVectorizer(
    min_df=2,          # কম-পক্ষে ২ document-এ থাকতে হবে
    max_df=0.9,        # ৯০%-এর বেশি document-এ থাকলে ফেলো (too common)
    stop_words='english',
)
X_filtered = cv_filtered.fit_transform(docs)
print(f"Filtered features: {len(cv_filtered.vocabulary_)}")
</code></pre>

    <h3>৩. TF-IDF Vectorizer</h3>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs = [
    "python data science machine learning",
    "machine learning deep learning neural",
    "cricket bangladesh match win trophy",
    "football goal match world cup",
]
labels = ['tech', 'tech', 'sports', 'sports']

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=20,    # সর্বোচ্চ ২০টি feature রাখো
    ngram_range=(1, 2),
    sublinear_tf=True,  # TF-এর log নাও: 1 + log(tf) — skewness কমায়
)
X_tfidf = tfidf.fit_transform(docs)

# TF-IDF score দেখো
feature_names = tfidf.get_feature_names_out()
for i, doc in enumerate(docs):
    tfidf_scores = X_tfidf[i].toarray()[0]
    top_idx = np.argsort(tfidf_scores)[-3:][::-1]
    print(f"\\n'{doc[:30]}...'")
    print("  Top TF-IDF terms:", [(feature_names[j], round(tfidf_scores[j], 3))
                                   for j in top_idx])
</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">TF-IDF — কেন সাধারণ শব্দের গুরুত্ব কম?</text>
        <rect x="10" y="26" width="140" height="80" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="80" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">"the" শব্দ</text>
        <text x="80" y="59" text-anchor="middle" font-size="8" fill="#3b82f6">TF = high</text>
        <text x="80" y="72" text-anchor="middle" font-size="8" fill="#3b82f6">IDF = low (সব doc-এ)</text>
        <text x="80" y="87" text-anchor="middle" font-size="8" fill="#6b7280">TF-IDF = low</text>
        <rect x="170" y="26" width="140" height="80" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="240" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">"python" শব্দ</text>
        <text x="240" y="59" text-anchor="middle" font-size="8" fill="#d97706">TF = medium</text>
        <text x="240" y="72" text-anchor="middle" font-size="8" fill="#d97706">IDF = high (কম doc-এ)</text>
        <text x="240" y="87" text-anchor="middle" font-size="8" fill="#6b7280">TF-IDF = high</text>
        <rect x="330" y="26" width="140" height="80" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="400" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">উদ্দেশ্য</text>
        <text x="400" y="59" text-anchor="middle" font-size="8" fill="#16a34a">গুরুত্বপূর্ণ ও</text>
        <text x="400" y="72" text-anchor="middle" font-size="8" fill="#16a34a">বিরল শব্দকে</text>
        <text x="400" y="87" text-anchor="middle" font-size="8" fill="#16a34a">বেশি weight দাও</text>
      </svg>
    </div>

    <h3>৪. সম্পূর্ণ NLP Pipeline</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Multi-class news classification dataset
news = [
    ("python tensorflow keras deep learning model", "tech"),
    ("data science machine learning algorithm",     "tech"),
    ("neural network computer vision image",        "tech"),
    ("javascript react frontend web development",   "tech"),
    ("bangladesh cricket match win series",         "sports"),
    ("football world cup final goal penalty",       "sports"),
    ("tennis wimbledon champion grand slam",        "sports"),
    ("olympics medal athlete record",               "sports"),
    ("economy gdp inflation budget deficit",        "finance"),
    ("stock market investment portfolio",           "finance"),
    ("bank interest rate loan credit",              "finance"),
    ("cryptocurrency bitcoin blockchain",           "finance"),
    ("election vote parliament government",         "politics"),
    ("prime minister cabinet policy reform",        "politics"),
    ("opposition party democracy protest",          "politics"),
    ("foreign policy diplomacy trade deal",         "politics"),
]

texts  = [t for t, _ in news]
labels = [l for _, l in news]
label_set = sorted(set(labels))

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# Pipeline: TF-IDF + MultinomialNB
pipe_mnb = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2))),
    ('clf',   MultinomialNB()),
])

# ComplementNB — imbalanced class-এ ভালো
pipe_cnb = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2))),
    ('clf',   ComplementNB()),
])

for name, pipe in [("MultinomialNB", pipe_mnb), ("ComplementNB", pipe_cnb)]:
    pipe.fit(X_train, y_train)
    print(f"\\n=== {name} ===")
    print(classification_report(y_test, pipe.predict(X_test)))

# GridSearch
param_grid = {
    'tfidf__max_features': [100, 500, None],
    'tfidf__ngram_range':  [(1,1), (1,2)],
    'clf__alpha':          [0.1, 0.5, 1.0],
}
grid = GridSearchCV(pipe_mnb, param_grid, cv=3, scoring='f1_macro')
grid.fit(X_train, y_train)
print(f"\\nBest params: {grid.best_params_}")
</code></pre>

    <h3>৫. বাংলা Text-এ সমস্যা ও সমাধান</h3>
    <pre><code># বাংলা text classification-এর চ্যালেঞ্জ:
# ১. Unicode normalization
# ২. Stop words বাংলায়
# ৩. Stemming/Lemmatization বাংলার জন্য সীমিত

import re

def preprocess_bangla(text):
    # Unicode normalization
    text = text.strip()
    # punctuation remove
    text = re.sub(r'[।!?,;:"\'()\\[\\]{}]', ' ', text)
    # multiple spaces
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# বাংলা stop words
bangla_stop_words = [
    'এবং', 'বা', 'কিন্তু', 'তবে', 'যে', 'এই', 'এর',
    'একটি', 'একটা', 'করা', 'হয়', 'হয়েছে', 'করে', 'করেন',
    'আছে', 'ছিল', 'থেকে', 'জন্য', 'দিয়ে', 'হলো',
]

# Analyzer function: text → tokens
def bangla_analyzer(text):
    text = preprocess_bangla(text)
    tokens = text.split()
    # Stop words remove
    tokens = [t for t in tokens if t not in bangla_stop_words and len(t) > 1]
    return tokens

# CountVectorizer-এ custom analyzer
cv_bangla = TfidfVectorizer(
    analyzer=bangla_analyzer,
    min_df=1,
)

bangla_docs = [
    "বাংলাদেশ ক্রিকেট দল জয়ী হয়েছে",
    "ক্রিকেট ম্যাচে বাংলাদেশ সিরিজ জিতেছে",
    "সংসদে বাজেট পেশ করা হয়েছে",
    "সরকার নতুন নীতি ঘোষণা করেছে",
]
bangla_labels = ['sports', 'sports', 'politics', 'politics']

X_bangla = cv_bangla.fit_transform(bangla_docs)
print("বাংলা features:", cv_bangla.get_feature_names_out())
</code></pre>

    <h3>৬. CountVectorizer vs TF-IDF তুলনা</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>CountVectorizer</th><th>TF-IDF</th></tr></thead>
      <tbody>
        <tr><td>ফিচার মান</td><td>Raw count (0, 1, 2, ...)</td><td>Normalized score (0.0–1.0)</td></tr>
        <tr><td>সাধারণ শব্দ</td><td>বেশি count → বেশি গুরুত্ব</td><td>কম গুরুত্ব (IDF কমিয়ে দেয়)</td></tr>
        <tr><td>Document length</td><td>দীর্ঘ document-এ bias</td><td>normalize করে bias কমায়</td></tr>
        <tr><td>NB compatibility</td><td>MultinomialNB — ভালো</td><td>MultinomialNB-এ ব্যবহার করা যায়</td></tr>
        <tr><td>কখন ব্যবহার</td><td>Short, similar-length docs</td><td>Varying-length docs, news, articles</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>CountVectorizer</td><td>text → word count matrix</td></tr>
        <tr><td>TF-IDF</td><td>frequent-but-rare শব্দকে বেশি গুরুত্ব দেয়</td></tr>
        <tr><td>ngram_range</td><td>bigram/trigram context ধরতে পারে</td></tr>
        <tr><td>sublinear_tf</td><td>1+log(tf) — count-এর skewness কমায়</td></tr>
        <tr><td>ComplementNB</td><td>imbalanced dataset-এ MultinomialNB-এর চেয়ে ভালো</td></tr>
      </tbody>
    </table>
  `,
};
