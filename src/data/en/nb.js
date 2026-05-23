export const nbEn = [
  {
    slug: "nb-bayes-theorem-math",
    title: "Bayes Theorem and the Mathematics of Naive Bayes",
    description: "Prior, Likelihood, Posterior — understand Bayes Theorem and how Naive Bayes classifies with complete math and Python code.",
    category: "Naive Bayes",
    readTime: 12,
    content: `
    <h3>1. Basic Probability</h3>
    <p>For two events A and B, the key relationships are:</p>
    <pre><code>P(A)        = probability of A occurring (prior)
P(B|A)      = probability of B given A (conditional / likelihood)
P(A ∩ B)    = P(B|A) × P(A) = P(A|B) × P(B)</code></pre>

    <h3>2. Bayes Theorem</h3>
    <p>Rearranging the joint probability identity gives us Bayes Theorem:</p>
    <pre><code>P(A|B) = [P(B|A) × P(A)] / P(B)

Medical example:
  A = patient is actually sick
  B = test came back positive

P(sick | positive) = P(positive | sick) × P(sick) / P(positive)

import numpy as np

sensitivity = 0.99   # P(positive | sick)
prevalence  = 0.01   # P(sick)
specificity = 0.95   # P(negative | healthy)

p_positive = sensitivity * prevalence + (1-specificity) * (1-prevalence)
p_sick_given_positive = (sensitivity * prevalence) / p_positive
print(f"P(sick | positive test): {p_sick_given_positive:.2%}")
# Only ~16.7%! Because the disease is rare (prevalence = 1%)</code></pre>

    <h3>3. Naive Bayes Classification</h3>
    <p>To classify an email as spam, Bayes Theorem gives us the posterior for each class. The "Naive" assumption is that all features are conditionally independent given the class.</p>
    <pre><code>P(spam | X) ∝ P(spam) × Π P(xᵢ | spam)
P(ham  | X) ∝ P(ham)  × Π P(xᵢ | ham)

# Whichever class has the higher score wins.</code></pre>

    <h3>4. Manual Implementation</h3>
    <pre><code>import numpy as np
from collections import defaultdict

emails = [
    ("free money win prize", "spam"),
    ("win free cash prize",  "spam"),
    ("free lottery winner",  "spam"),
    ("meeting tomorrow work","ham"),
    ("project deadline work","ham"),
    ("lunch meeting office", "ham"),
]

spam_count = sum(1 for _, l in emails if l == "spam")
ham_count  = sum(1 for _, l in emails if l == "ham")
total      = len(emails)

p_spam = spam_count / total
p_ham  = ham_count  / total

spam_words = defaultdict(int)
ham_words  = defaultdict(int)

for text, label in emails:
    for word in text.split():
        if label == "spam": spam_words[word] += 1
        else:               ham_words[word]  += 1

total_spam_words = sum(spam_words.values())
total_ham_words  = sum(ham_words.values())

test_email = "free win money"
log_p_spam = np.log(p_spam)
log_p_ham  = np.log(p_ham)
vocab_size = len(set(spam_words) | set(ham_words))

for word in test_email.split():
    p_w_spam = (spam_words[word] + 1) / (total_spam_words + vocab_size)
    p_w_ham  = (ham_words[word]  + 1) / (total_ham_words  + vocab_size)
    log_p_spam += np.log(p_w_spam)
    log_p_ham  += np.log(p_w_ham)

print(f"Prediction: {'spam' if log_p_spam > log_p_ham else 'ham'}")</code></pre>

    <h3>5. Why "Naive"?</h3>
    <p>In reality words are not independent — "Bangladesh" and "cricket" tend to co-occur. But even with this incorrect assumption, Naive Bayes works well in practice because we only care about which class ranks highest, not the exact probabilities.</p>

    <h3>6. Three Variants of Naive Bayes</h3>
    <table>
      <thead><tr><th>Variant</th><th>Feature Type</th><th>Likelihood</th><th>Use Case</th></tr></thead>
      <tbody>
        <tr><td><strong>Gaussian NB</strong></td><td>Continuous (real numbers)</td><td>Normal Distribution</td><td>Iris, medical data</td></tr>
        <tr><td><strong>Multinomial NB</strong></td><td>Count (non-negative integers)</td><td>Multinomial Distribution</td><td>Text classification, word counts</td></tr>
        <tr><td><strong>Bernoulli NB</strong></td><td>Binary (0 or 1)</td><td>Bernoulli Distribution</td><td>Spam (word present/absent)</td></tr>
      </tbody>
    </table>

    <h3>Summary</h3>
    <table>
      <thead><tr><th>Concept</th><th>Key Point</th></tr></thead>
      <tbody>
        <tr><td>Bayes Theorem</td><td>Posterior ∝ Likelihood × Prior</td></tr>
        <tr><td>Naive assumption</td><td>Features are conditionally independent — wrong but effective</td></tr>
        <tr><td>Classification</td><td>Compute posterior for each class, pick the highest</td></tr>
        <tr><td>Log probability</td><td>Convert product to sum to avoid underflow</td></tr>
        <tr><td>Laplace smoothing</td><td>Add 1 to counts to avoid zero probability</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-gaussian-naive-bayes",
    title: "Gaussian Naive Bayes — Normal Distribution for Continuous Features",
    description: "How Naive Bayes uses Normal Distribution for continuous features, Python implementation, and when Gaussian NB is the best choice.",
    category: "Naive Bayes",
    readTime: 10,
    content: `
    <h3>1. How Gaussian NB Works</h3>
    <p>Multinomial NB counts words. But temperature, height, weight — these continuous values cannot be counted. Gaussian NB assumes each feature follows a <strong>Normal (Gaussian) Distribution</strong> for each class.</p>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Gaussian PDF: P(x | class) = (1/√(2πσ²)) × exp(−(x−μ)²/2σ²)

mu_male,   sigma_male   = 170, 8
mu_female, sigma_female = 158, 7

x = np.linspace(130, 200, 200)
plt.plot(x, norm.pdf(x, mu_male,   sigma_male),   label='Male')
plt.plot(x, norm.pdf(x, mu_female, sigma_female), label='Female')
plt.xlabel('Height (cm)')
plt.ylabel('Probability Density')
plt.legend(); plt.show()

p_male   = norm.pdf(165, mu_male,   sigma_male)
p_female = norm.pdf(165, mu_female, sigma_female)
print(f"P(height=165 | Male)   = {p_male:.5f}")
print(f"P(height=165 | Female) = {p_female:.5f}")</code></pre>

    <h3>2. Training: Computing μ and σ</h3>
    <pre><code>import numpy as np

X_train = np.array([[170,70],[175,80],[168,65],[158,55],[160,52],[155,48]])
y_train = np.array([1, 1, 1, 0, 0, 0])  # 1=Male, 0=Female

for cls in [0, 1]:
    label = "Male" if cls == 1 else "Female"
    X_cls = X_train[y_train == cls]
    print(f"{label}: mean={X_cls.mean(axis=0)}, std={X_cls.std(axis=0)}")</code></pre>

    <h3>3. Prediction: Likelihood via Gaussian PDF</h3>
    <pre><code>from scipy.stats import norm

x_new = np.array([162, 56])
log_prior_male   = np.log(0.5)
log_prior_female = np.log(0.5)
log_likelihood_male = log_likelihood_female = 0

for feat_idx, x_val in enumerate(x_new):
    X_male   = X_train[y_train == 1][:, feat_idx]
    mu_m, s_m = X_male.mean(), X_male.std()
    log_likelihood_male += np.log(norm.pdf(x_val, mu_m, s_m) + 1e-10)

    X_female = X_train[y_train == 0][:, feat_idx]
    mu_f, s_f = X_female.mean(), X_female.std()
    log_likelihood_female += np.log(norm.pdf(x_val, mu_f, s_f) + 1e-10)

log_post_male   = log_prior_male   + log_likelihood_male
log_post_female = log_prior_female + log_likelihood_female
print(f"Prediction: {'Male' if log_post_male > log_post_female else 'Female'}")</code></pre>

    <h3>4. sklearn GaussianNB</h3>
    <pre><code>from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")
print(f"CV Accuracy:   {cross_val_score(gnb, X, y, cv=5).mean():.4f}")
print(classification_report(y_test, gnb.predict(X_test),
      target_names=data.target_names))</code></pre>

    <h3>5. var_smoothing Tuning</h3>
    <pre><code>from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {'var_smoothing': np.logspace(-12, 0, 100)}
grid = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best var_smoothing: {grid.best_params_['var_smoothing']:.2e}")
print(f"Test Accuracy:      {grid.score(X_test, y_test):.4f}")</code></pre>

    <h3>6. When to Use Gaussian NB</h3>
    <table>
      <thead><tr><th>Scenario</th><th>Gaussian NB Suitable?</th></tr></thead>
      <tbody>
        <tr><td>Continuous numeric features</td><td>Yes — best fit</td></tr>
        <tr><td>Normally distributed features</td><td>Yes — assumption holds</td></tr>
        <tr><td>Small training data</td><td>Yes — works well with limited data</td></tr>
        <tr><td>Real-time prediction needed</td><td>Yes — extremely fast</td></tr>
        <tr><td>Strongly correlated features</td><td>No — independence assumption violated</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-multinomial-naive-bayes",
    title: "Multinomial Naive Bayes — Text and Count Data",
    description: "How Multinomial NB counts words to classify text, sklearn MultinomialNB, and alpha smoothing explained.",
    category: "Naive Bayes",
    readTime: 10,
    content: `
    <h3>1. What is Multinomial NB?</h3>
    <p>Multinomial NB is used when features are <strong>counts or frequencies</strong> — such as how many times each word appears in a document. It is the most popular NB variant for text classification.</p>
    <pre><code>P(xᵢ | class) = (count(word_i, class) + α) / (total_words_in_class + α × |V|)
# α = Laplace smoothing, |V| = vocabulary size</code></pre>

    <h3>2. Manual Multinomial NB</h3>
    <pre><code>import numpy as np
from collections import defaultdict, Counter

docs = [
    ("python data science machine learning", "tech"),
    ("deep learning neural network python",  "tech"),
    ("cricket bangladesh win match",         "sports"),
    ("football world cup goal",              "sports"),
]

classes   = list(set(l for _, l in docs))
vocab     = set(w for t, _ in docs for w in t.split())
vocab_size = len(vocab)

class_counts = Counter(l for _, l in docs)
log_priors   = {c: np.log(n/len(docs)) for c, n in class_counts.items()}

word_counts = {c: defaultdict(int) for c in classes}
total_words = {c: 0 for c in classes}
for text, label in docs:
    for word in text.split():
        word_counts[label][word] += 1
        total_words[label] += 1

alpha = 1
def log_likelihood(word, cls):
    return np.log((word_counts[cls][word] + alpha) /
                  (total_words[cls] + alpha * vocab_size))

test_doc = "python machine learning"
for cls in classes:
    score = log_priors[cls] + sum(log_likelihood(w, cls) for w in test_doc.split())
    print(f"log P({cls} | doc) = {score:.3f}")</code></pre>

    <h3>3. sklearn MultinomialNB</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts  = ["python machine learning ai", "deep learning tensorflow",
          "cricket match win", "football world cup",
          "stock market investment", "economy inflation"]
labels = [0, 0, 1, 1, 2, 2]
label_names = ['tech', 'sports', 'finance']

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42)

vectorizer   = CountVectorizer(ngram_range=(1,2))
X_train_vec  = vectorizer.fit_transform(X_train)
X_test_vec   = vectorizer.transform(X_test)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_vec, y_train)
print(classification_report(y_test, mnb.predict(X_test_vec), target_names=label_names))</code></pre>

    <h3>4. alpha Tuning</h3>
    <pre><code>from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {'alpha': np.logspace(-3, 1, 50)}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_vec, y_train)
print(f"Best alpha: {grid.best_params_['alpha']:.4f}")</code></pre>

    <h3>5. Top Predictive Words</h3>
    <pre><code>feature_names = vectorizer.get_feature_names_out()
for i, cls in enumerate(label_names):
    top_idx   = np.argsort(mnb.feature_log_prob_[i])[-10:][::-1]
    top_words = [(feature_names[j], np.exp(mnb.feature_log_prob_[i][j]))
                 for j in top_idx]
    print(f"{cls}: {top_words}")</code></pre>

    <h3>Summary</h3>
    <table>
      <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
      <tbody>
        <tr><td>Multinomial NB</td><td>For count data — how often each word appears</td></tr>
        <tr><td>CountVectorizer</td><td>Converts text to word count matrix</td></tr>
        <tr><td>alpha</td><td>Laplace smoothing — prevents zero probability</td></tr>
        <tr><td>Best use</td><td>Text classification — fast and effective baseline</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-bernoulli-naive-bayes",
    title: "Bernoulli Naive Bayes — Binary Features and Word Presence",
    description: "How Bernoulli NB handles binary features, the difference from Multinomial NB, and sklearn spam detection.",
    category: "Naive Bayes",
    readTime: 9,
    content: `
    <h3>1. What is Bernoulli NB?</h3>
    <p>Bernoulli NB is used when each feature is <strong>binary (0 or 1)</strong> — whether a word appears in the document, not how many times. The key distinction from Multinomial NB is that absent features also contribute to the likelihood.</p>
    <pre><code>P(xᵢ | class) = P(i|class)^xᵢ × (1 − P(i|class))^(1−xᵢ)
# xᵢ=1 if feature i is present, 0 if absent</code></pre>

    <h3>2. Bernoulli vs Multinomial — Key Difference</h3>
    <pre><code>from sklearn.feature_extraction.text import CountVectorizer

text = "win win win lottery"

# CountVectorizer — counts occurrences
cv = CountVectorizer()
cv.fit([text])
print("Count:", cv.transform([text]).toarray())   # win=3

# Binary — presence only
bv = CountVectorizer(binary=True)
bv.fit([text])
print("Binary:", bv.transform([text]).toarray())  # win=1</code></pre>

    <h3>3. Full BernoulliNB Pipeline</h3>
    <pre><code>from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

texts = [
    "FREE entry Win £1000 cash! Call now!!",
    "Congratulations! Claim your free gift voucher",
    "FREE ticket prize call now claim",
    "Hey, are you coming to the party tonight?",
    "The project deadline is next Friday",
    "Let me know when you are free to talk",
]
labels = [1, 1, 1, 0, 0, 0]

pipe = Pipeline([
    ('vec', CountVectorizer(binary=True, lowercase=True, stop_words='english')),
    ('clf', BernoulliNB(alpha=1.0)),
])
pipe.fit(texts, labels)

new_sms = ["FREE cash win call now", "see you tomorrow at the office"]
preds = pipe.predict(new_sms)
probs = pipe.predict_proba(new_sms)
for sms, pred, prob in zip(new_sms, preds, probs):
    print(f"'{sms[:40]}' → {'SPAM' if pred else 'HAM'}  P(spam)={prob[1]:.3f}")</code></pre>

    <h3>4. When to Use Each Variant</h3>
    <table>
      <thead><tr><th>Scenario</th><th>Best Choice</th></tr></thead>
      <tbody>
        <tr><td>Short documents (SMS, tweet)</td><td>Bernoulli NB — word presence is enough</td></tr>
        <tr><td>Long documents (article, email body)</td><td>Multinomial NB — frequency matters</td></tr>
        <tr><td>TF-IDF features</td><td>Multinomial or Complement NB</td></tr>
        <tr><td>Non-text binary features</td><td>Bernoulli NB</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-text-classification-tfidf",
    title: "Text Classification — TF-IDF and CountVectorizer for NLP",
    description: "CountVectorizer, TF-IDF, n-gram, pipeline — complete guide to text classification with Naive Bayes.",
    category: "Naive Bayes",
    readTime: 11,
    content: `
    <h3>1. Why Convert Text to Numbers?</h3>
    <p>Machine learning models work with numbers, not text. Three main approaches: Bag of Words (word counts), Binary (presence/absence), and TF-IDF (weighted by rarity).</p>
    <pre><code>TF(t,d)  = occurrences of t in d / total words in d
IDF(t)   = log(total docs / docs containing t)
TF-IDF   = TF × IDF

# Common words (the, is, a) → low IDF → low weight
# Rare but meaningful words → high IDF → high weight</code></pre>

    <h3>2. CountVectorizer</h3>
    <pre><code>from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

docs = ["I love machine learning",
        "machine learning is great",
        "python is awesome for machine learning"]

cv = CountVectorizer(ngram_range=(1,2), min_df=1)
X  = cv.fit_transform(docs)
print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))</code></pre>

    <h3>3. TF-IDF Vectorizer</h3>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs   = ["python data science machine",
          "machine learning deep neural",
          "cricket bangladesh match win",
          "football world cup goal"]
labels = ['tech','tech','sports','sports']

tfidf     = TfidfVectorizer(max_features=20, ngram_range=(1,2), sublinear_tf=True)
X_tfidf   = tfidf.fit_transform(docs)
names     = tfidf.get_feature_names_out()
# sublinear_tf=True applies 1+log(tf) to reduce count skewness</code></pre>

    <h3>4. Full NLP Classification Pipeline</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

texts  = ["python tensorflow keras deep","data science machine learning",
          "cricket match win series",    "football world cup final",
          "economy gdp inflation",       "stock market investment",
          "election parliament vote",    "prime minister policy reform"]
labels = [0,0,1,1,2,2,3,3]
label_names = ['tech','sports','finance','politics']

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2))),
    ('clf',   MultinomialNB()),
])

param_grid = {
    'tfidf__max_features': [100, 500, None],
    'tfidf__ngram_range':  [(1,1),(1,2)],
    'clf__alpha':          [0.1, 0.5, 1.0],
}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro')
grid.fit(texts, labels)
print(f"Best params: {grid.best_params_}")</code></pre>

    <h3>5. CountVectorizer vs TF-IDF</h3>
    <table>
      <thead><tr><th>Topic</th><th>CountVectorizer</th><th>TF-IDF</th></tr></thead>
      <tbody>
        <tr><td>Feature value</td><td>Raw count (0, 1, 2 ...)</td><td>Normalized score (0.0–1.0)</td></tr>
        <tr><td>Common words</td><td>High count → high weight</td><td>IDF lowers their weight</td></tr>
        <tr><td>Document length</td><td>Long docs are biased</td><td>Normalizes the bias</td></tr>
        <tr><td>Best for</td><td>Short, similar-length docs</td><td>Varying-length articles</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-laplace-smoothing",
    title: "Laplace Smoothing and the Zero Probability Problem",
    description: "How zero probability breaks Naive Bayes, Laplace and Lidstone smoothing math and Python.",
    category: "Naive Bayes",
    readTime: 9,
    content: `
    <h3>1. The Zero Probability Problem</h3>
    <p>In Naive Bayes all feature probabilities are multiplied together. If any one feature has P = 0, the entire posterior becomes 0 — even if every other feature strongly points to that class.</p>
    <pre><code># If "lottery" never appeared in ham training data:
P("lottery" | ham) = 0

P(ham | email) ∝ P(ham) × ... × P("lottery"|ham) × ...
               = P(ham) × ... × 0 × ... = 0
# Even if every other word is ham-like, prediction fails!</code></pre>

    <h3>2. Laplace Smoothing (Add-1 Smoothing)</h3>
    <pre><code>import numpy as np

# Add α to every count so nothing is ever 0
# P(word | class) = (count + α) / (total + α × |V|)

def laplace_prob(word, word_counts, total, vocab_size, alpha=1):
    return (word_counts.get(word, 0) + alpha) / (total + alpha * vocab_size)

spam_words = {"free":10,"win":8,"prize":6}
ham_words  = {"meeting":9,"project":7,"office":8}
vocab_size = len(set(spam_words)|set(ham_words))

# Now "prize" in ham → (0+1)/(24+|V|) > 0
print(laplace_prob("prize", ham_words, sum(ham_words.values()), vocab_size))</code></pre>

    <h3>3. Lidstone Smoothing</h3>
    <pre><code>import numpy as np
# alpha=1  → Laplace
# alpha<1  → less smoothing, closer to raw counts
# alpha>1  → more smoothing, approaches uniform

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
for a in alphas:
    # See how prediction changes with different alpha values
    print(f"alpha={a:.3f}  P(word|class) shifts toward uniform")</code></pre>

    <h3>4. Log Probability — Underflow Prevention</h3>
    <pre><code>import numpy as np

# With 1000 features, multiplying tiny probs → 0.0 (underflow)
product = 1.0
for _ in range(1000):
    product *= 0.001
print(f"Direct product: {product}")  # 0.0!

# Use log-sum instead:
log_sum = sum(np.log(0.001) for _ in range(1000))
print(f"Log sum: {log_sum:.2f}")  # finite!

# For classification, just compare which log-posterior is larger</code></pre>

    <h3>5. sklearn Alpha Parameter</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

pipe = Pipeline([('vec', CountVectorizer()), ('clf', MultinomialNB())])
grid = GridSearchCV(pipe, {'clf__alpha': np.logspace(-3,1,30)}, cv=5, scoring='f1')
grid.fit(texts, labels)
print(f"Best alpha: {grid.best_params_['clf__alpha']:.4f}")</code></pre>

    <h3>Summary</h3>
    <table>
      <thead><tr><th>Topic</th><th>Key Point</th></tr></thead>
      <tbody>
        <tr><td>Zero Probability</td><td>Unseen word → P=0 → posterior=0</td></tr>
        <tr><td>Laplace Smoothing</td><td>Add α to every count</td></tr>
        <tr><td>alpha=1</td><td>Laplace; alpha&lt;1 = Lidstone</td></tr>
        <tr><td>Log Probability</td><td>Convert product to log-sum to avoid underflow</td></tr>
      </tbody>
    </table>
  `,
  },
  {
    slug: "nb-spam-detection-project",
    title: "End-to-End Project: SMS Spam Detection with Naive Bayes",
    description: "Build a complete spam detector using the SMS Spam Collection dataset — EDA, preprocessing, model comparison, and production pipeline.",
    category: "Naive Bayes",
    readTime: 14,
    content: `
    <h3>1. Project Overview</h3>
    <p>We use the UCI SMS Spam Collection dataset to build a classifier that detects whether an SMS is spam or legitimate (ham).</p>
    <table>
      <thead><tr><th>Dataset Info</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Total SMS</td><td>5,574</td></tr>
        <tr><td>Spam</td><td>747 (13.4%)</td></tr>
        <tr><td>Ham</td><td>4,827 (86.6%)</td></tr>
        <tr><td>Target</td><td>binary: spam / ham</td></tr>
      </tbody>
    </table>

    <h3>2. Data Loading and EDA</h3>
    <pre><code>import pandas as pd
import numpy as np
from collections import Counter

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df  = pd.read_csv(url, sep='\\t', header=None, names=['label','text'])

df['text_length']    = df['text'].str.len()
df['word_count']     = df['text'].str.split().str.len()
df['has_currency']   = df['text'].str.contains(r'[£$€]').astype(int)
df['capitals_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x),1))

print(df.groupby('label')[['text_length','capitals_ratio','has_currency']].mean())</code></pre>

    <h3>3. Text Preprocessing</h3>
    <pre><code>import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+', 'URL', text)
    text = re.sub(r'\\b\\d{10,}\\b', 'PHONE', text)
    text = re.sub(r'[£$€]\\d+', 'MONEY', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(preprocess_text)
df['label_bin']  = (df['label'] == 'spam').astype(int)</code></pre>

    <h3>4. Model Comparison</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X = df['text_clean']; y = df['label_bin']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'MultinomialNB (Count)': Pipeline([
        ('vec', CountVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)),
        ('clf', MultinomialNB(alpha=0.1))]),
    'MultinomialNB (TF-IDF)': Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)),
        ('clf', MultinomialNB(alpha=0.1))]),
    'BernoulliNB': Pipeline([
        ('vec', CountVectorizer(binary=True, max_features=5000)),
        ('clf', BernoulliNB(alpha=0.5))]),
    'ComplementNB': Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
        ('clf', ComplementNB(alpha=0.1))]),
}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    auc  = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    acc  = pipe.score(X_test, y_test)
    print(f"{name:28s}  AUC={auc:.4f}  Acc={acc:.4f}")</code></pre>

    <h3>5. Hyperparameter Tuning</h3>
    <pre><code>from sklearn.model_selection import RandomizedSearchCV

best_pipe  = Pipeline([
    ('vec', TfidfVectorizer(sublinear_tf=True)),
    ('clf', MultinomialNB()),
])
param_dist = {
    'vec__max_features': [3000,5000,10000],
    'vec__ngram_range':  [(1,1),(1,2),(1,3)],
    'clf__alpha':        np.logspace(-3,1,20),
}
search = RandomizedSearchCV(best_pipe, param_dist, n_iter=30, cv=5,
                            scoring='f1', random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(classification_report(y_test, search.predict(X_test),
      target_names=['ham','spam']))</code></pre>

    <h3>6. Production Prediction Function</h3>
    <pre><code>import pickle

with open('spam_detector.pkl', 'wb') as f:
    pickle.dump(search.best_estimator_, f)

def predict_spam(sms_text):
    with open('spam_detector.pkl','rb') as f:
        model = pickle.load(f)
    clean = preprocess_text(sms_text)
    prob  = model.predict_proba([clean])[0]
    pred  = model.predict([clean])[0]
    return {'label': 'SPAM' if pred else 'HAM',
            'p_spam': float(prob[1]),
            'p_ham':  float(prob[0])}

# Test
print(predict_spam("FREE entry Win £500 cash! Call now!!!"))
print(predict_spam("Are you free for lunch tomorrow?"))</code></pre>

    <h3>Model Performance Summary</h3>
    <table>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>
        <tr><td>MultinomialNB (Count)</td><td>~98%</td><td>~97%</td><td>~93%</td><td>~95%</td></tr>
        <tr><td>MultinomialNB (TF-IDF)</td><td>~98%</td><td>~98%</td><td>~94%</td><td>~96%</td></tr>
        <tr><td>BernoulliNB</td><td>~97%</td><td>~95%</td><td>~91%</td><td>~93%</td></tr>
        <tr><td>ComplementNB</td><td>~98%</td><td>~97%</td><td>~95%</td><td>~96%</td></tr>
      </tbody>
    </table>
  `,
  },
];
