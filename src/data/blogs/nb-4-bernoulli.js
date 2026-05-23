export const nb_4_bernoulli = {
  title: "Bernoulli Naive Bayes — Binary ফিচার ও শব্দের উপস্থিতি",
  description: "Bernoulli NB কীভাবে binary feature নিয়ে কাজ করে, Multinomial NB-এর সাথে পার্থক্য, এবং sklearn-এ spam detection — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 9,
  slug: "nb-bernoulli-naive-bayes",
  content: `
    <h3>১. Bernoulli NB কী?</h3>
    <p>Bernoulli NB ব্যবহার করে যখন প্রতিটি feature <strong>binary (0 বা 1)</strong> — শব্দটি document-এ আছে কিনা, কিন্তু কতবার আছে সেটা গণনা করে না।</p>
    <pre><code>Bernoulli NB Likelihood:
P(xᵢ | class) = P(i | class)^xᵢ × (1 − P(i | class))^(1−xᵢ)

যেখানে:
  xᵢ = 1 যদি feature i উপস্থিত থাকে, 0 যদি অনুপস্থিত
  P(i|class) = class-এ feature i উপস্থিত থাকার probability

গুরুত্বপূর্ণ পার্থক্য:
  Multinomial: অনুপস্থিত শব্দ উপেক্ষা করে
  Bernoulli:   অনুপস্থিত শব্দও likelihood-এ contribute করে!</code></pre>

    <h3>২. হাতে-কলমে Bernoulli NB</h3>
    <pre><code>import numpy as np
from collections import defaultdict

# Training data: (text, label)
emails = [
    ("free money prize win lottery",    "spam"),
    ("win cash free offer prize",       "spam"),
    ("free gift claim now money",       "spam"),
    ("meeting project deadline work",   "ham"),
    ("office lunch tomorrow calendar",  "ham"),
    ("team update project schedule",    "ham"),
]

# Vocabulary তৈরি
vocab = sorted(set(w for text, _ in emails for w in text.split()))
word_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary ({len(vocab)} words):", vocab[:8], "...")

# Binary feature matrix: শব্দ আছে (1) বা নেই (0)
def text_to_binary(text, vocab, word_to_idx):
    vec = np.zeros(len(vocab), dtype=int)
    for word in text.split():
        if word in word_to_idx:
            vec[word_to_idx[word]] = 1  # count নয়, শুধু presence
    return vec

X = np.array([text_to_binary(t, vocab, word_to_idx) for t, _ in emails])
y = np.array([1 if l == "spam" else 0 for _, l in emails])

print("Feature matrix shape:", X.shape)
print("First email (spam):", X[0])

# Training: প্রতিটি class-এ প্রতিটি feature কতটা সম্ভব?
alpha = 1  # Laplace smoothing
for cls in [0, 1]:
    label = "spam" if cls == 1 else "ham"
    X_cls = X[y == cls]
    # P(feature_i = 1 | class)
    p_feat_given_cls = (X_cls.sum(axis=0) + alpha) / (len(X_cls) + 2 * alpha)
    print(f"\\n{label} — top present features:")
    top_idx = np.argsort(p_feat_given_cls)[-5:][::-1]
    for i in top_idx:
        print(f"  '{vocab[i]}': {p_feat_given_cls[i]:.2f}")
</code></pre>

    <h3>৩. Bernoulli vs Multinomial — মূল পার্থক্য</h3>
    <pre><code>from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = [
    "buy cheap medicine now free",
    "win win win lottery free money",
    "project meeting tomorrow office",
    "schedule update team lunch",
    "free free free win cash",
]
labels = [1, 1, 0, 0, 1]  # 1=spam, 0=ham

# CountVectorizer (count-based) — MultinomialNB-এর জন্য
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(texts)

# binary=True → Bernoulli-র জন্য
binary_vec = CountVectorizer(binary=True)
X_binary = binary_vec.fit_transform(texts)

# পার্থক্য দেখো
text = "win win win"
print("Count vectorizer:", count_vec.transform([text]).toarray()[0])
# win শব্দের count = 3
print("Binary vectorizer:", binary_vec.transform([text]).toarray()[0])
# win শব্দের value = 1 (শুধু উপস্থিতি)

# মডেল training
mnb = MultinomialNB(alpha=1)
bnb = BernoulliNB(alpha=1)

mnb.fit(X_count, labels)
bnb.fit(X_binary, labels)

test = ["free cash now"]
print("\\nMultinomialNB:", mnb.predict(count_vec.transform(test)))
print("BernoulliNB:  ", bnb.predict(binary_vec.transform(test)))
</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Bernoulli vs Multinomial — Feature Encoding</text>
        <text x="20" y="36" font-size="9" fill="#6b7280">Text: "win win free lottery"</text>
        <rect x="10" y="44" width="220" height="80" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="120" y="62" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">Multinomial (Count)</text>
        <text x="120" y="78" text-anchor="middle" font-size="8" fill="#3b82f6">win=2  free=1  lottery=1</text>
        <text x="120" y="94" text-anchor="middle" font-size="8" fill="#3b82f6">কতবার এসেছে গণনা করে</text>
        <text x="120" y="110" text-anchor="middle" font-size="8" fill="#6b7280">অনুপস্থিত শব্দ = উপেক্ষা</text>
        <rect x="250" y="44" width="220" height="80" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="360" y="62" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Bernoulli (Binary)</text>
        <text x="360" y="78" text-anchor="middle" font-size="8" fill="#16a34a">win=1  free=1  lottery=1</text>
        <text x="360" y="94" text-anchor="middle" font-size="8" fill="#16a34a">শুধু উপস্থিত/অনুপস্থিত</text>
        <text x="360" y="110" text-anchor="middle" font-size="8" fill="#6b7280">অনুপস্থিত শব্দও count করে</text>
      </svg>
    </div>

    <h3>৪. sklearn BernoulliNB — সম্পূর্ণ Pipeline</h3>
    <pre><code>from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# SMS Spam-এর মতো dataset তৈরি করি
spam_texts = [
    "WINNER!! You've been selected for a FREE prize call now",
    "Congratulations! Claim your free gift voucher",
    "FREE entry in 2 weekly comp to win FA Cup",
    "You have won £1000 cash! Call to claim now",
    "URGENT: Your mobile account has a special offer",
    "Win a brand new car text WIN to 87121",
]
ham_texts = [
    "Hey are you coming to the party tonight",
    "I'll be late for the meeting tomorrow",
    "Can you pick up some groceries on the way",
    "The project deadline is next Friday",
    "Let me know when you're free to talk",
    "Happy birthday! Hope you have a great day",
]

texts  = spam_texts + ham_texts
labels = [1]*6 + [0]*6  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# BernoulliNB pipeline — binary=True গুরুত্বপূর্ণ!
pipe = Pipeline([
    ('vec', CountVectorizer(
        binary=True,       # Bernoulli-র জন্য binary=True
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
    )),
    ('clf', BernoulliNB(alpha=1.0)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred,
      target_names=['ham', 'spam']))

# CV score
cv = cross_val_score(pipe, texts, labels, cv=3, scoring='f1')
print(f"CV F1: {cv.mean():.3f} ± {cv.std():.3f}")

# নতুন SMS predict
new_sms = [
    "FREE ticket win call now claim prize",
    "see you tomorrow at the office",
]
preds = pipe.predict(new_sms)
probs = pipe.predict_proba(new_sms)
for sms, pred, prob in zip(new_sms, preds, probs):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"\\n'{sms[:40]}...'")
    print(f"  → {label}  (P(spam)={prob[1]:.3f})")
</code></pre>

    <h3>৫. binarize parameter</h3>
    <pre><code># BernoulliNB-এর binarize: continuous input থেকে binary তৈরি করতে পারে
# যদি CountVectorizer ব্যবহার না করো

from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Continuous features (যেমন TF-IDF scores)
X_continuous = np.array([
    [0.0, 0.5, 0.8, 0.0, 0.3],
    [0.4, 0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.6, 0.0, 0.0],
])
y = [0, 1, 0]

# binarize=0.0 → 0-এর চেয়ে বেশি হলে 1, নইলে 0
bnb = BernoulliNB(alpha=1.0, binarize=0.0)
bnb.fit(X_continuous, y)
print("Binarized internally — continuous input-ও কাজ করে")
</code></pre>

    <h3>৬. কখন কোনটা ব্যবহার করবে?</h3>
    <table>
      <thead><tr><th>পরিস্থিতি</th><th>সেরা পছন্দ</th></tr></thead>
      <tbody>
        <tr><td>Short documents (SMS, tweet)</td><td>Bernoulli NB — শব্দের উপস্থিতিই যথেষ্ট</td></tr>
        <tr><td>Long documents (article, email)</td><td>Multinomial NB — frequency গুরুত্বপূর্ণ</td></tr>
        <tr><td>TF-IDF features ব্যবহার করলে</td><td>Multinomial বা Complement NB</td></tr>
        <tr><td>Binary features (non-text)</td><td>Bernoulli NB</td></tr>
        <tr><td>Vocabulary খুব বড়</td><td>Bernoulli — absent words penalize করে overfitting কমায়</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Bernoulli NB</td><td>binary feature — শব্দ আছে (1) বা নেই (0)</td></tr>
        <tr><td>Multinomial NB-এর পার্থক্য</td><td>অনুপস্থিত শব্দও likelihood-এ contribute করে</td></tr>
        <tr><td>binary=True</td><td>CountVectorizer-এ এটি set করতে হবে</td></tr>
        <tr><td>binarize</td><td>continuous input-কে threshold দিয়ে binary করে</td></tr>
        <tr><td>সেরা ব্যবহার</td><td>short text, SMS spam, keyword presence/absence</td></tr>
      </tbody>
    </table>
  `,
};
