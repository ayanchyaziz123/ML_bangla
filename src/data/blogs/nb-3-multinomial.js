export const nb_3_multinomial = {
  title: "Multinomial Naive Bayes — Text ও Count Data",
  description: "Multinomial NB কীভাবে শব্দ গণনা করে text classify করে, sklearn-এর MultinomialNB ও alpha smoothing — বাংলায় সম্পূর্ণ গাইড।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 10,
  slug: "nb-multinomial-naive-bayes",
  content: `
    <h3>১. Multinomial NB কী?</h3>
    <p>Gaussian NB continuous feature-এর জন্য। Multinomial NB ব্যবহার হয় যখন feature হলো <strong>count বা frequency</strong> — যেমন একটি document-এ কোন শব্দ কতবার এসেছে। Text classification-এর জন্য এটি সবচেয়ে জনপ্রিয় NB ভেরিয়্যান্ট।</p>
    <pre><code>Multinomial NB-এর Likelihood:
P(xᵢ | class) = (count(word_i, class) + α) / (total_words_in_class + α × |V|)

যেখানে:
  count(word_i, class) = class-এ word_i কতবার এসেছে
  α                    = Laplace smoothing parameter (default=1)
  |V|                  = vocabulary size (মোট unique শব্দ)</code></pre>

    <h3>২. হাতে-কলমে Multinomial NB</h3>
    <pre><code>import numpy as np
from collections import defaultdict, Counter

# Training corpus
docs = [
    ("python data science machine learning", "tech"),
    ("deep learning neural network python",  "tech"),
    ("data analysis pandas numpy",           "tech"),
    ("cricket bangladesh win match",         "sports"),
    ("football world cup goal",              "sports"),
    ("cricket team player match win",        "sports"),
]

# ক্লাস ও vocabulary তৈরি
classes = list(set(label for _, label in docs))
vocab   = set(w for text, _ in docs for w in text.split())
vocab_size = len(vocab)

print(f"Classes:    {classes}")
print(f"Vocabulary: {vocab_size} unique words")

# Prior probability
class_counts = Counter(label for _, label in docs)
total_docs   = len(docs)
log_priors   = {c: np.log(n / total_docs) for c, n in class_counts.items()}

# Word counts per class
word_counts = {c: defaultdict(int) for c in classes}
total_words = {c: 0 for c in classes}

for text, label in docs:
    for word in text.split():
        word_counts[label][word] += 1
        total_words[label] += 1

# Laplace smoothing দিয়ে log likelihood
alpha = 1  # Laplace smoothing

def log_likelihood(word, cls):
    count = word_counts[cls][word] + alpha
    total = total_words[cls] + alpha * vocab_size
    return np.log(count / total)

# Predict
test_doc = "python machine learning data"
for cls in classes:
    score = log_priors[cls]
    for word in test_doc.split():
        score += log_likelihood(word, cls)
    print(f"log P({cls} | doc) = {score:.3f}")
</code></pre>

    <h3>৩. sklearn দিয়ে MultinomialNB</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Sample dataset
texts = [
    "python programming code software",
    "machine learning neural network ai",
    "data science analysis statistics",
    "deep learning tensorflow keras",
    "cricket football sports game",
    "bangladesh win match cricket",
    "goal football player team",
    "sports tournament championship",
    "stock market finance economy",
    "bank investment money profit",
    "tax revenue budget government",
    "economy inflation interest rate",
]
labels = [0,0,0,0, 1,1,1,1, 2,2,2,2]  # 0=tech, 1=sports, 2=finance
label_names = ['tech', 'sports', 'finance']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# CountVectorizer: text → word count matrix
vectorizer = CountVectorizer(
    min_df=1,         # কম-পক্ষে ১ document-এ থাকতে হবে
    ngram_range=(1,2) # unigram + bigram
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Feature count: {X_train_vec.shape[1]}")

# MultinomialNB
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_vec, y_train)

y_pred = mnb.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=label_names))

# নতুন text predict করো
new_texts = ["python data analysis code", "cricket match win goal"]
new_vec   = vectorizer.transform(new_texts)
preds     = mnb.predict(new_vec)
probs     = mnb.predict_proba(new_vec)

for text, pred, prob in zip(new_texts, preds, probs):
    print(f"\\n'{text}'")
    print(f"  Prediction: {label_names[pred]}")
    for i, cls in enumerate(label_names):
        print(f"  P({cls}) = {prob[i]:.3f}")
</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Multinomial NB — Text Classification Pipeline</text>
        <rect x="10" y="28" width="95" height="80" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="57" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#1e40af">Raw Text</text>
        <text x="57" y="65" text-anchor="middle" font-size="8" fill="#3b82f6">"python data</text>
        <text x="57" y="78" text-anchor="middle" font-size="8" fill="#3b82f6">science code"</text>
        <text x="57" y="95" text-anchor="middle" font-size="8" fill="#6b7280">input</text>
        <text x="112" y="72" font-size="12" fill="#94a3b8">→</text>
        <rect x="122" y="28" width="110" height="80" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="177" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#92400e">CountVectorizer</text>
        <text x="177" y="65" text-anchor="middle" font-size="8" fill="#d97706">python: 1</text>
        <text x="177" y="78" text-anchor="middle" font-size="8" fill="#d97706">data: 1, code: 1</text>
        <text x="177" y="91" text-anchor="middle" font-size="8" fill="#d97706">cricket: 0 ...</text>
        <text x="239" y="72" font-size="12" fill="#94a3b8">→</text>
        <rect x="249" y="28" width="110" height="80" rx="5" fill="#fce7f3" stroke="#f9a8d4"/>
        <text x="304" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#9d174d">MultinomialNB</text>
        <text x="304" y="65" text-anchor="middle" font-size="8" fill="#be185d">P(tech|doc)</text>
        <text x="304" y="78" text-anchor="middle" font-size="8" fill="#be185d">P(sports|doc)</text>
        <text x="304" y="91" text-anchor="middle" font-size="8" fill="#be185d">P(finance|doc)</text>
        <text x="366" y="72" font-size="12" fill="#94a3b8">→</text>
        <rect x="376" y="28" width="95" height="80" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="423" y="50" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">argmax</text>
        <text x="423" y="67" text-anchor="middle" font-size="11" fill="#16a34a">tech</text>
        <text x="423" y="85" text-anchor="middle" font-size="8" fill="#6b7280">predicted class</text>
      </svg>
    </div>

    <h3>৪. alpha (Smoothing) Tuning</h3>
    <pre><code>from sklearn.model_selection import GridSearchCV
import numpy as np

# alpha=0 মানে কোনো smoothing নেই (zero probability সমস্যা!)
# alpha=1 মানে Laplace smoothing
# alpha>1 মানে বেশি smoothing (uniform distribution-এর দিকে যায়)

param_grid = {'alpha': np.logspace(-3, 1, 50)}

grid = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='accuracy',
)
grid.fit(X_train_vec, y_train)

print(f"সেরা alpha:    {grid.best_params_['alpha']:.4f}")
print(f"Test Accuracy: {grid.score(X_test_vec, y_test):.4f}")

# alpha-এর প্রভাব দেখো
import matplotlib.pyplot as plt
alphas     = np.logspace(-3, 1, 50)
cv_scores  = []
for a in alphas:
    s = GridSearchCV(MultinomialNB(alpha=a), {}, cv=5, scoring='accuracy')
    s.fit(X_train_vec, y_train)
    cv_scores.append(s.best_score_)

plt.semilogx(alphas, cv_scores)
plt.xlabel('alpha (log scale)')
plt.ylabel('CV Accuracy')
plt.title('Multinomial NB: alpha tuning')
plt.axvline(grid.best_params_['alpha'], color='red', linestyle='--', label='best')
plt.legend()
plt.show()
</code></pre>

    <h3>৫. Top Predictive Words দেখো</h3>
    <pre><code># প্রতিটি class-এর সবচেয়ে গুরুত্বপূর্ণ শব্দ
feature_names = vectorizer.get_feature_names_out()

for i, cls in enumerate(label_names):
    top_idx = np.argsort(mnb.feature_log_prob_[i])[-10:][::-1]
    top_words = [(feature_names[j], np.exp(mnb.feature_log_prob_[i][j]))
                 for j in top_idx]
    print(f"\\n{cls} — Top 10 Words:")
    for word, prob in top_words:
        print(f"  {word:20s}  P={prob:.4f}")
</code></pre>

    <h3>৬. Multinomial vs Gaussian NB</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>Multinomial NB</th><th>Gaussian NB</th></tr></thead>
      <tbody>
        <tr><td>Feature ধরন</td><td>Count / frequency (integers ≥ 0)</td><td>Continuous (real numbers)</td></tr>
        <tr><td>Likelihood model</td><td>Multinomial distribution</td><td>Normal distribution</td></tr>
        <tr><td>সেরা ব্যবহার</td><td>Text classification, NLP</td><td>Medical data, sensor data</td></tr>
        <tr><td>Negative values</td><td>সমর্থন করে না</td><td>সমর্থন করে</td></tr>
        <tr><td>Smoothing</td><td>alpha (Laplace)</td><td>var_smoothing</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Multinomial NB</td><td>count data-র জন্য — প্রতিটি শব্দ কতবার এসেছে</td></tr>
        <tr><td>CountVectorizer</td><td>text → word count matrix</td></tr>
        <tr><td>alpha</td><td>Laplace smoothing — zero probability এড়ায়</td></tr>
        <tr><td>feature_log_prob_</td><td>প্রতিটি class-এর প্রতিটি word-এর log probability</td></tr>
        <tr><td>সুবিধা</td><td>text classification-এ দ্রুত ও কার্যকর baseline</td></tr>
      </tbody>
    </table>
  `,
};
