export const nb_1_bayes_theorem = {
  title: "Bayes Theorem ও Naive Bayes-এর গণিত",
  description: "Prior, Likelihood, Posterior — Bayes Theorem বাংলায় বোঝো এবং Naive Bayes কীভাবে classification করে তার সম্পূর্ণ গণিত ও Python কোড।",
  date: "২৩ মে, ২০২৬",
  category: "নেইভ বেজ",
  readTime: 12,
  slug: "nb-bayes-theorem-math",
  content: `
    <h3>১. Probability-র মূল ধারণা</h3>
    <p>দুটি ঘটনা A ও B-এর ক্ষেত্রে:</p>
    <pre><code>P(A)        = A ঘটার সম্ভাবনা (prior probability)
P(B|A)      = A ঘটলে B ঘটার সম্ভাবনা (conditional / likelihood)
P(A ∩ B)    = A এবং B দুটোই ঘটার সম্ভাবনা (joint probability)

সম্পর্ক:
P(A ∩ B) = P(B|A) × P(A)
         = P(A|B) × P(B)</code></pre>

    <h3>২. Bayes Theorem</h3>
    <p>উপরের সম্পর্ক থেকে:</p>
    <pre><code>P(A|B) = [P(B|A) × P(A)] / P(B)

চিকিৎসার উদাহরণ:
  A = রোগী আসলে অসুস্থ
  B = টেস্ট positive এসেছে

P(অসুস্থ | positive) = P(positive | অসুস্থ) × P(অসুস্থ)
                       ─────────────────────────────────────
                                    P(positive)

= Sensitivity × Prevalence / P(positive)

import numpy as np

# উদাহরণ:
sensitivity = 0.99   # P(positive | অসুস্থ)
prevalence  = 0.01   # P(অসুস্থ)
specificity = 0.95   # P(negative | সুস্থ)

# P(positive):
p_positive = sensitivity * prevalence + (1-specificity) * (1-prevalence)

# Bayes Theorem:
p_sick_given_positive = (sensitivity * prevalence) / p_positive
print(f"টেস্ট positive হলে আসলে অসুস্থ হওয়ার সম্ভাবনা: {p_sick_given_positive:.2%}")
# মাত্র ~16.7%! কারণ রোগটি বিরল (prevalence = 1%)</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 480 115" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:480px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="240" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Bayes Theorem-এর উপাদান</text>
        <rect x="15"  y="28" width="95" height="70" rx="5" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="62"  y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#1e40af">Prior</text>
        <text x="62"  y="66" text-anchor="middle" font-size="9" fill="#3b82f6">P(A)</text>
        <text x="62"  y="82" text-anchor="middle" font-size="8" fill="#6b7280">আগের জ্ঞান</text>
        <text x="117" y="66" text-anchor="middle" font-size="16" fill="#94a3b8">×</text>
        <rect x="130" y="28" width="100" height="70" rx="5" fill="#fef3c7" stroke="#fcd34d"/>
        <text x="180" y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#92400e">Likelihood</text>
        <text x="180" y="66" text-anchor="middle" font-size="9" fill="#d97706">P(B|A)</text>
        <text x="180" y="82" text-anchor="middle" font-size="8" fill="#6b7280">ডেটা দেখার সম্ভাবনা</text>
        <text x="237" y="66" text-anchor="middle" font-size="16" fill="#94a3b8">÷</text>
        <rect x="250" y="28" width="100" height="70" rx="5" fill="#fce7f3" stroke="#f9a8d4"/>
        <text x="300" y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#9d174d">Evidence</text>
        <text x="300" y="66" text-anchor="middle" font-size="9" fill="#be185d">P(B)</text>
        <text x="300" y="82" text-anchor="middle" font-size="8" fill="#6b7280">normalizing constant</text>
        <text x="357" y="66" text-anchor="middle" font-size="16" fill="#94a3b8">=</text>
        <rect x="370" y="28" width="95" height="70" rx="5" fill="#dcfce7" stroke="#86efac"/>
        <text x="417" y="52" text-anchor="middle" font-size="10" font-weight="600" fill="#166534">Posterior</text>
        <text x="417" y="66" text-anchor="middle" font-size="9" fill="#16a34a">P(A|B)</text>
        <text x="417" y="82" text-anchor="middle" font-size="8" fill="#6b7280">আপডেট করা জ্ঞান</text>
      </svg>
    </div>

    <h3>৩. Naive Bayes Classification</h3>
    <p>ইমেইল spam কিনা ঠিক করতে হবে। ইমেইলে X = (x₁, x₂, ..., xₙ) শব্দ আছে। Bayes Theorem অনুযায়ী:</p>
    <pre><code">P(spam | X) = P(X | spam) × P(spam) / P(X)

P(X | spam) = P(x₁,x₂,...,xₙ | spam)

# এখানেই "Naive" assumption:
# ধরা হয় features পরস্পর independent!

P(X | spam) ≈ P(x₁|spam) × P(x₂|spam) × ... × P(xₙ|spam)

# তাহলে:
P(spam | X) ∝ P(spam) × Π P(xᵢ | spam)
P(ham  | X) ∝ P(ham)  × Π P(xᵢ | ham)

# যে class-এর score বেশি, সে class-ই prediction</code></pre>

    <h3>৪. হাতে-কলমে হিসাব</h3>
    <pre><code">import numpy as np
from collections import defaultdict

# Training data
emails = [
    ("free money win prize", "spam"),
    ("win free cash prize",  "spam"),
    ("free lottery winner",  "spam"),
    ("meeting tomorrow work","ham"),
    ("project deadline work","ham"),
    ("lunch meeting office", "ham"),
]

# ধাপ ১: Prior probability
spam_count = sum(1 for _, l in emails if l == "spam")
ham_count  = sum(1 for _, l in emails if l == "ham")
total = len(emails)

p_spam = spam_count / total
p_ham  = ham_count  / total
print(f"P(spam) = {p_spam:.3f}")
print(f"P(ham)  = {p_ham:.3f}")

# ধাপ ২: Likelihood — প্রতিটি শব্দের ক্ষেত্রে P(word|class)
spam_words = defaultdict(int)
ham_words  = defaultdict(int)

for text, label in emails:
    for word in text.split():
        if label == "spam": spam_words[word] += 1
        else:               ham_words[word]  += 1

total_spam_words = sum(spam_words.values())
total_ham_words  = sum(ham_words.values())

# ধাপ ৩: নতুন ইমেইল classify করো
test_email = "free win money"
test_words = test_email.split()

# Log probability ব্যবহার করো (underflow এড়াতে)
log_p_spam = np.log(p_spam)
log_p_ham  = np.log(p_ham)

vocab_size = len(set(spam_words) | set(ham_words))  # Laplace smoothing

for word in test_words:
    p_w_spam = (spam_words[word] + 1) / (total_spam_words + vocab_size)
    p_w_ham  = (ham_words[word]  + 1) / (total_ham_words  + vocab_size)
    log_p_spam += np.log(p_w_spam)
    log_p_ham  += np.log(p_w_ham)

print(f"\\nlog P(spam|email) ∝ {log_p_spam:.3f}")
print(f"log P(ham|email)  ∝ {log_p_ham:.3f}")
print(f"Prediction: {'spam' if log_p_spam > log_p_ham else 'ham'}")</code></pre>

    <h3>৫. কেন "Naive"?</h3>
    <p>বাস্তবে শব্দগুলো স্বাধীন নয় — "বাংলাদেশ" এবং "ক্রিকেট" একসাথে আসার সম্ভাবনা বেশি। কিন্তু independence assumption ধরলেও Naive Bayes অনেক ক্ষেত্রে ভালো কাজ করে। কারণ:</p>
    <ul>
      <li>আমরা exact probability নই, শুধু কোন class বেশি সম্ভাব্য সেটা দেখছি</li>
      <li>অনেক সময় rank order ঠিক থাকে even if probabilities wrong</li>
      <li>Training data কম হলেও ভালো কাজ করে</li>
    </ul>

    <h3>৬. তিন ধরনের Naive Bayes</h3>
    <table>
      <thead><tr><th>ধরন</th><th>Feature-এর ধরন</th><th>Likelihood Model</th><th>ব্যবহার</th></tr></thead>
      <tbody>
        <tr><td><strong>Gaussian NB</strong></td><td>Continuous (real numbers)</td><td>Normal Distribution</td><td>Iris, medical data</td></tr>
        <tr><td><strong>Multinomial NB</strong></td><td>Count (non-negative integers)</td><td>Multinomial Distribution</td><td>Text classification, word count</td></tr>
        <tr><td><strong>Bernoulli NB</strong></td><td>Binary (0 বা 1)</td><td>Bernoulli Distribution</td><td>Spam (শব্দ আছে/নেই)</td></tr>
      </tbody>
    </table>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>ধারণা</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>Bayes Theorem</td><td>Posterior ∝ Likelihood × Prior</td></tr>
        <tr><td>Naive assumption</td><td>features পরস্পর independent — বাস্তবে ভুল কিন্তু কার্যকর</td></tr>
        <tr><td>Classification</td><td>সব class-এর posterior হিসাব করো, সবচেয়ে বেশিটা বেছে নাও</td></tr>
        <tr><td>Log probability</td><td>underflow এড়াতে product → sum of logs</td></tr>
        <tr><td>Laplace smoothing</td><td>zero probability এড়াতে count-এ 1 যোগ করো</td></tr>
      </tbody>
    </table>
  `,
};
