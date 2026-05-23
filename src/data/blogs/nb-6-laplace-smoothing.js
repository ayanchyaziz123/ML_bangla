export const nb_6_laplace_smoothing = {
  title: "Laplace Smoothing ও Zero Probability সমস্যা",
  description: "Naive Bayes-এ zero probability কীভাবে সব কিছু নষ্ট করে, Laplace ও Lidstone smoothing-এর গণিত ও Python — বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 9,
  slug: "nb-laplace-smoothing",
  content: `
    <h3>১. Zero Probability সমস্যাটা কী?</h3>
    <p>Naive Bayes-এ সব feature-এর probability গুণ করা হয়। যদি যেকোনো একটি feature-এর P = 0 হয়, তাহলে পুরো posterior = 0 হয়ে যায়!</p>
    <pre><code>উদাহরণ:
Training data-তে "lottery" শব্দটি শুধু spam-এ আছে।
কিন্তু test email-এ "lottery" আছে ham-এ এমন context-এ।

P("lottery" | ham) = 0  (training-এ দেখা যায়নি)

P(ham | email) ∝ P(ham) × P(w1|ham) × P(w2|ham) × P("lottery"|ham) × ...
                = P(ham) × P(w1|ham) × P(w2|ham) × 0 × ...
                = 0

এমনকি বাকি সব শব্দ ham-এর মতো হলেও prediction fail করবে!
এটাকে বলে "Zero Frequency Problem"।</code></pre>

    <h3>২. কেন এটা গুরুতর সমস্যা?</h3>
    <pre><code>import numpy as np

# Training corpus
spam_words = {"free": 10, "win": 8, "prize": 6, "money": 5}
ham_words  = {"meeting": 9, "project": 7, "office": 8, "lunch": 5}

vocab = set(spam_words) | set(ham_words)
total_spam = sum(spam_words.values())  # 29
total_ham  = sum(ham_words.values())   # 29

# নতুন email: "free meeting prize"
# "prize" আছে শুধু spam-এ → P("prize"|ham) = 0
test_words = ["free", "meeting", "prize"]

# Smoothing ছাড়া
log_p_spam = np.log(0.5)  # prior
log_p_ham  = np.log(0.5)

for word in test_words:
    # spam
    count_spam = spam_words.get(word, 0)
    p_spam = count_spam / total_spam if count_spam > 0 else 0
    if p_spam > 0:
        log_p_spam += np.log(p_spam)
    else:
        log_p_spam = -np.inf  # log(0) = -∞

    # ham
    count_ham = ham_words.get(word, 0)
    p_ham = count_ham / total_ham if count_ham > 0 else 0
    if p_ham > 0:
        log_p_ham += np.log(p_ham)
    else:
        log_p_ham = -np.inf

print(f"log P(spam | email) = {log_p_spam}")  # -inf!
print(f"log P(ham  | email) = {log_p_ham}")   # -inf!
print("দুটোই -inf → prediction অনিশ্চিত!")
</code></pre>

    <h3>৩. Laplace Smoothing (Add-1 Smoothing)</h3>
    <pre><code>import numpy as np

# Laplace Smoothing Formula:
# P(word | class) = (count(word, class) + α) / (total_words_in_class + α × |V|)
#
# α = 1 (Laplace smoothing)
# |V| = vocabulary size

def laplace_prob(word, word_counts, total_words, vocab_size, alpha=1):
    count = word_counts.get(word, 0)
    return (count + alpha) / (total_words + alpha * vocab_size)

spam_words = {"free": 10, "win": 8, "prize": 6, "money": 5}
ham_words  = {"meeting": 9, "project": 7, "office": 8, "lunch": 5}
vocab = set(spam_words) | set(ham_words)
vocab_size = len(vocab)

total_spam = sum(spam_words.values())
total_ham  = sum(ham_words.values())

# এখন "prize"-এর P(prize|ham) = 0+1 / 29+8 > 0!
test_words = ["free", "meeting", "prize"]

log_p_spam = np.log(0.5)
log_p_ham  = np.log(0.5)

for word in test_words:
    log_p_spam += np.log(laplace_prob(word, spam_words, total_spam, vocab_size))
    log_p_ham  += np.log(laplace_prob(word, ham_words,  total_ham,  vocab_size))

print(f"log P(spam | email) = {log_p_spam:.4f}")
print(f"log P(ham  | email) = {log_p_ham:.4f}")
print(f"Prediction: {'spam' if log_p_spam > log_p_ham else 'ham'}")
</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Laplace Smoothing — Before vs After</text>
        <rect x="10" y="26" width="220" height="90" rx="5" fill="#fee2e2" stroke="#fca5a5"/>
        <text x="120" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#991b1b">Smoothing ছাড়া</text>
        <text x="120" y="59" text-anchor="middle" font-size="8" fill="#dc2626">P("lottery"|ham) = 0/29 = 0</text>
        <text x="120" y="73" text-anchor="middle" font-size="8" fill="#dc2626">P(ham|doc) = ... × 0 = 0</text>
        <text x="120" y="87" text-anchor="middle" font-size="8" fill="#dc2626">log P(ham) = −∞</text>
        <text x="120" y="101" text-anchor="middle" font-size="8" fill="#991b1b">Prediction অসম্ভব!</text>
        <rect x="250" y="26" width="220" height="90" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="360" y="44" text-anchor="middle" font-size="9" font-weight="600" fill="#166534">Laplace Smoothing (α=1)</text>
        <text x="360" y="59" text-anchor="middle" font-size="8" fill="#16a34a">P("lottery"|ham) = (0+1)/(29+|V|)</text>
        <text x="360" y="73" text-anchor="middle" font-size="8" fill="#16a34a">= 1/37 = 0.027 &gt; 0</text>
        <text x="360" y="87" text-anchor="middle" font-size="8" fill="#16a34a">log P(ham) = finite!</text>
        <text x="360" y="101" text-anchor="middle" font-size="8" fill="#166534">Prediction সম্ভব</text>
      </svg>
    </div>

    <h3>৪. Lidstone Smoothing (alpha ≠ 1)</h3>
    <pre><code>import numpy as np

# Laplace (alpha=1) সবসময় সেরা নয়।
# Lidstone smoothing: alpha যেকোনো positive number

# alpha ছোট → training data-র উপর বেশি নির্ভর
# alpha বড়  → uniform distribution-এর দিকে যায়
# alpha=0   → no smoothing (zero probability সমস্যা!)

def compute_score(test_words, word_counts, total, vocab_size, alpha):
    score = 0
    for word in test_words:
        count = word_counts.get(word, 0)
        score += np.log((count + alpha) / (total + alpha * vocab_size))
    return score

spam_words = {"free": 10, "win": 8, "prize": 6, "money": 5, "lottery": 4}
ham_words  = {"meeting": 9, "project": 7, "office": 8, "lunch": 5}
vocab = set(spam_words) | set(ham_words)
vocab_size = len(vocab)

test_unseen = ["lottery", "meeting"]
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

print(f"{'alpha':>8}  {'P(spam|doc)':>14}  {'P(ham|doc)':>13}  Pred")
for a in alphas:
    s = compute_score(test_unseen, spam_words, sum(spam_words.values()), vocab_size, a)
    h = compute_score(test_unseen, ham_words,  sum(ham_words.values()),  vocab_size, a)
    pred = "spam" if s > h else "ham"
    print(f"{a:>8.3f}  {s:>14.4f}  {h:>13.4f}  {pred}")
</code></pre>

    <h3>৫. sklearn-এ Smoothing</h3>
    <pre><code>from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

texts = [
    "free money win prize",
    "win lottery cash free",
    "meeting project work",
    "office deadline schedule",
    "free offer claim now",
    "team lunch tomorrow office",
]
labels = [1, 1, 0, 0, 1, 0]  # 1=spam, 0=ham

# MultinomialNB alpha tuning
pipe = Pipeline([
    ('vec', CountVectorizer()),
    ('clf', MultinomialNB()),
])

param_grid = {'clf__alpha': np.logspace(-3, 1, 30)}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1')
grid.fit(texts, labels)

print(f"সেরা alpha:    {grid.best_params_['clf__alpha']:.4f}")
print(f"সেরা CV score: {grid.best_score_:.4f}")

# alpha=0 হলে কী হয়?
pipe_no_smooth = Pipeline([
    ('vec', CountVectorizer()),
    ('clf', MultinomialNB(alpha=0)),
])
try:
    pipe_no_smooth.fit(texts, labels)
    print("alpha=0 দিয়ে trained (সাবধান!)")
except Exception as e:
    print(f"Error with alpha=0: {e}")

# GaussianNB-এ var_smoothing
from sklearn.naive_bayes import GaussianNB
import numpy as np

# var_smoothing: variance-এ ছোট মান যোগ করে numerical stability নিশ্চিত করে
gnb = GaussianNB(var_smoothing=1e-9)  # default
print(f"\\nGaussianNB var_smoothing: {gnb.var_smoothing}")
</code></pre>

    <h3>৬. Log Probability — Underflow সমস্যা</h3>
    <pre><code>import numpy as np

# ছোট probability গুণ করতে করতে computer-এ 0 হয়ে যায় (underflow)
p1 = 0.001
p2 = 0.002
p3 = 0.0005
# 1000টি শব্দ গুণ করলে:
product = 1.0
for _ in range(1000):
    product *= p1
print(f"Direct product (1000 terms): {product}")  # 0.0 — underflow!

# সমাধান: log নাও, গুণের বদলে যোগ করো
log_sum = 0
for _ in range(1000):
    log_sum += np.log(p1)
print(f"Log sum (1000 terms): {log_sum:.2f}")  # finite!

# তুলনা করার সময় শুধু কোনটা বড় সেটাই দরকার:
log_p_spam = -50.3
log_p_ham  = -48.7
print(f"Prediction: {'spam' if log_p_spam > log_p_ham else 'ham'}")
# log comparison-এই যথেষ্ট — actual probability বের করতে হয় না
</code></pre>

    <h3>৭. Smoothing Method তুলনা</h3>
    <table>
      <thead><tr><th>Method</th><th>Formula</th><th>alpha</th><th>ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td>No Smoothing</td><td>count / total</td><td>0</td><td>avoid — zero probability</td></tr>
        <tr><td>Laplace</td><td>(count+1) / (total+|V|)</td><td>1</td><td>সাধারণ default</td></tr>
        <tr><td>Lidstone</td><td>(count+α) / (total+α|V|)</td><td>0–∞</td><td>tune করলে ভালো</td></tr>
        <tr><td>Jeffreys-Perks</td><td>(count+0.5) / (total+0.5|V|)</td><td>0.5</td><td>sparse data-তে</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Zero Probability</td><td>training-এ না দেখা শব্দ → P=0 → posterior=0</td></tr>
        <tr><td>Laplace Smoothing</td><td>প্রতিটি count-এ α যোগ করো</td></tr>
        <tr><td>alpha=1</td><td>Laplace; alpha&lt;1 = Lidstone</td></tr>
        <tr><td>Log Probability</td><td>গুণের বদলে log-sum → underflow এড়ায়</td></tr>
        <tr><td>GridSearch</td><td>alpha-কে hyperparameter হিসেবে tune করো</td></tr>
      </tbody>
    </table>
  `,
};
