export const nlp_1_tokenization = {
  slug: 'nlp-1-tokenization',
  title: 'টোকেনাইজেশন: টেক্সট প্রি-প্রসেসিং-এর প্রথম ধাপ',
  description: 'ওয়ার্ড টোকেনাইজেশন, সেন্টেন্স টোকেনাইজেশন, BPE ও WordPiece সাবওয়ার্ড টোকেনাইজেশন, স্টপ ওয়ার্ড রিমুভাল, স্টেমিং বনাম লেমাটাইজেশন — NLTK ও spaCy সহ সম্পূর্ণ গাইড।',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 12,
  content: `
    <h3>১. টোকেনাইজেশন কী এবং কেন দরকার?</h3>
    <p>
      ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং (NLP)-এ কাজ করতে হলে সবার আগে টেক্সটকে একটি মেশিন-পঠনযোগ্য রূপে রূপান্তর করতে হয়। <strong>টোকেনাইজেশন</strong> হলো সেই প্রথম ধাপ — যেখানে একটি বড় টেক্সটকে ছোট ছোট অর্থবহ একককে ভেঙে ফেলা হয়। এই ছোট একককে <em>টোকেন</em> বলা হয়।
    </p>
    <p>
      উদাহরণস্বরূপ, "আমি বাংলা ভালোবাসি।" বাক্যটিকে টোকেনাইজ করলে পাওয়া যায়: <code>["আমি", "বাংলা", "ভালোবাসি", "."]</code>। কম্পিউটার সরাসরি শব্দ বুঝতে পারে না — তাকে সংখ্যা দিতে হয়। টোকেনাইজেশন সেই সংখ্যায় রূপান্তরের ভিত্তি তৈরি করে।
    </p>

    <h3>২. ওয়ার্ড টোকেনাইজেশন</h3>
    <p>
      সবচেয়ে সরল পদ্ধতি হলো ওয়ার্ড টোকেনাইজেশন — স্পেস বা বিরাম চিহ্নের ভিত্তিতে টেক্সট ভাঙা। তবে এটি সবসময় আদর্শ নয়। যেমন "New York" দুটি শব্দ হলেও একটি entity। বা "don't" কি "do" + "n't" নাকি "do" + "not"?
    </p>
    <pre><code>import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

text = "Natural language processing is fascinating. It helps computers understand human language."

# ওয়ার্ড টোকেনাইজেশন
word_tokens = word_tokenize(text)
print("ওয়ার্ড টোকেন:", word_tokens)
# ['Natural', 'language', 'processing', 'is', 'fascinating', '.', ...]

# সেন্টেন্স টোকেনাইজেশন
sent_tokens = sent_tokenize(text)
print("সেন্টেন্স টোকেন:", sent_tokens)
# ['Natural language processing is fascinating.',
#  'It helps computers understand human language.']</code></pre>

    <h3>৩. সেন্টেন্স টোকেনাইজেশন</h3>
    <p>
      কখনো কখনো আমাদের সম্পূর্ণ বাক্যকে আলাদা করতে হয়। এটি document summarization বা machine translation-এ বিশেষভাবে কার্যকর। NLTK-এর <code>sent_tokenize</code> Punkt অ্যালগরিদম ব্যবহার করে — যা ভাষার প্যাটার্ন শিখে বাক্যের সীমা নির্ধারণ করে।
    </p>
    <p>
      একটি চ্যালেঞ্জ: "Dr. Smith works at St. Mary's Hospital." — এখানে "Dr." এবং "St." বাক্যের শেষ নয়, কিন্তু পিরিয়ড আছে। Punkt এই ধরনের abbreviation চিনতে পারে।
    </p>

    <h3>৪. স্টপ ওয়ার্ড রিমুভাল</h3>
    <p>
      <strong>স্টপ ওয়ার্ড</strong> হলো এমন শব্দ যেগুলো বাক্যে খুব বেশি ঘটে কিন্তু অর্থগত তথ্য কম বহন করে — যেমন "the", "is", "at", "which", "on"। এগুলো সরিয়ে দিলে মডেলের noise কমে এবং গুরুত্বপূর্ণ শব্দগুলো বেশি প্রভাব ফেলতে পারে।
    </p>
    <pre><code>from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

text = "This is a simple example showing stop word removal in NLP"
tokens = word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]

print("আগে:", tokens)
print("পরে:", filtered)
# পরে: ['simple', 'example', 'showing', 'stop', 'word', 'removal', 'nlp']</code></pre>
    <p>
      সতর্কতা: sentiment analysis-এ স্টপ ওয়ার্ড সরানো উচিত নয়, কারণ "not good" থেকে "not" সরালে অর্থ উল্টে যায়।
    </p>

    <h3>৫. স্টেমিং বনাম লেমাটাইজেশন</h3>
    <p>
      "running", "runs", "ran" — এগুলো সবই "run" ক্রিয়ার রূপ। <strong>স্টেমিং</strong> এবং <strong>লেমাটাইজেশন</strong> দুটি পদ্ধতি এই শব্দগুলোকে তাদের মূল রূপে নামিয়ে আনতে চায়।
    </p>
    <p>
      <strong>স্টেমিং</strong> একটি rule-based পদ্ধতি — শব্দের শেষ অংশ কেটে ফেলে। দ্রুত কিন্তু অনেক সময় ভুল ফলাফল দেয়। যেমন "studies" → "studi" (সঠিক শব্দ নয়)।
    </p>
    <p>
      <strong>লেমাটাইজেশন</strong> ভাষার অভিধান ও ব্যাকরণ ব্যবহার করে। "studies" → "study", "better" → "good"। ধীর কিন্তু নির্ভুল।
    </p>
    <pre><code>from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better", "caring", "wolves"]

print("স্টেমিং:")
for w in words:
    print(f"  {w} -> {stemmer.stem(w)}")

print("\nলেমাটাইজেশন:")
for w in words:
    print(f"  {w} -> {lemmatizer.lemmatize(w, pos='v')}")

# স্টেমিং:
#   running -> run
#   studies -> studi   (ভুল!)
#   better  -> better
#   wolves  -> wolv    (ভুল!)

# লেমাটাইজেশন:
#   running -> run
#   studies -> study
#   better  -> better  (pos='a' দিলে 'good' পাওয়া যায়)
#   wolves  -> wolf</code></pre>

    <h3>৬. সাবওয়ার্ড টোকেনাইজেশন: BPE ও WordPiece</h3>
    <p>
      আধুনিক ভাষা মডেলগুলো (BERT, GPT) ওয়ার্ড টোকেনাইজেশন ব্যবহার করে না — তারা ব্যবহার করে <strong>সাবওয়ার্ড টোকেনাইজেশন</strong>। এর কারণ হলো vocabulary size নিয়ন্ত্রণে রাখা এবং OOV (out-of-vocabulary) সমস্যা এড়ানো।
    </p>

    <h4>৬.১ Byte Pair Encoding (BPE)</h4>
    <p>
      BPE শুরু করে individual characters দিয়ে এবং পুনরাবৃত্তিমূলকভাবে সবচেয়ে বেশি ঘটা character জোড়াকে একটি নতুন token হিসেবে merge করে। উদাহরণ:
    </p>
    <p>
      শুরু: <code>["l", "o", "w", "e", "r"]</code><br/>
      সবচেয়ে ঘন জোড়া "lo" → merge → <code>["lo", "w", "e", "r"]</code><br/>
      পরের জোড়া "low" → merge → <code>["low", "e", "r"]</code><br/>
      ইত্যাদি।
    </p>
    <p>GPT-2 BPE ব্যবহার করে এবং ৫০,২৫৭ টোকেনের vocabulary রাখে।</p>

    <h4>৬.২ WordPiece</h4>
    <p>
      BERT-এ ব্যবহৃত WordPiece BPE-র মতোই কিন্তু merge criteria আলাদা — maximum likelihood ব্যবহার করে। WordPiece-এ সাবওয়ার্ড টোকেনের আগে <code>##</code> যোগ হয়:
    </p>
    <p>
      "unaffable" → <code>["un", "##aff", "##able"]</code>
    </p>
    <pre><code>from transformers import BertTokenizer, GPT2Tokenizer

# BERT tokenizer (WordPiece)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = bert_tokenizer.tokenize("unaffable tokenization")
print("BERT:", tokens)
# ['un', '##aff', '##able', 'token', '##ization']

# GPT-2 tokenizer (BPE)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = gpt2_tokenizer.encode("unaffable tokenization", return_tensors=None)
print("GPT-2 token IDs:", tokens)</code></pre>

    <h3>৭. spaCy দিয়ে টোকেনাইজেশন</h3>
    <p>
      spaCy একটি শক্তিশালী NLP লাইব্রেরি যা NLTK-এর তুলনায় দ্রুত এবং আরও বেশি feature সমৃদ্ধ। spaCy-তে টোকেনাইজেশন rule-based এবং language-specific।
    </p>
    <pre><code>import spacy

# ইংরেজি মডেল লোড করুন
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

print("টোকেন | লেমা | POS | স্টপওয়ার্ড?")
print("-" * 50)
for token in doc:
    print(f"{token.text:15} | {token.lemma_:12} | {token.pos_:6} | {token.is_stop}")

# টোকেন          | লেমা         | POS    | স্টপওয়ার্ড?
# Apple           | Apple        | PROPN  | False
# is              | be           | AUX    | True
# looking         | look         | VERB   | False
# at              | at           | ADP    | True
# buying          | buy          | VERB   | False</code></pre>

    <h3>৮. পূর্ণ প্রি-প্রসেসিং পাইপলাইন</h3>
    <p>
      বাস্তব প্রজেক্টে একটি সম্পূর্ণ প্রি-প্রসেসিং পাইপলাইন এরকম দেখায়:
    </p>
    <pre><code>import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # ১. লোয়ারকেস
    text = text.lower()

    # ২. URL ও স্পেশাল ক্যারেক্টার সরানো
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # ৩. spaCy দিয়ে টোকেনাইজেশন ও লেমাটাইজেশন
    doc = nlp(text)

    # ৪. স্টপওয়ার্ড সরানো ও লেমাটাইজেশন
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]

    return tokens

sample = "The researchers are studying NLP techniques for better text understanding."
result = preprocess(sample)
print(result)
# ['researcher', 'study', 'nlp', 'technique', 'better', 'text', 'understanding']</code></pre>

    <h3>৯. কোন পদ্ধতি কখন ব্যবহার করবেন?</h3>
    <p>
      <strong>ওয়ার্ড টোকেনাইজেশন:</strong> সহজ text classification, keyword extraction।<br/>
      <strong>সেন্টেন্স টোকেনাইজেশন:</strong> summarization, machine translation, question answering।<br/>
      <strong>BPE/WordPiece:</strong> transformer মডেল (BERT, GPT), neural machine translation।<br/>
      <strong>স্টেমিং:</strong> information retrieval, search engine (গতি প্রয়োজন হলে)।<br/>
      <strong>লেমাটাইজেশন:</strong> যেখানে নির্ভুলতা গুরুত্বপূর্ণ — sentiment analysis, chatbot।
    </p>

    <h3>১০. সারসংক্ষেপ</h3>
    <p>
      টোকেনাইজেশন NLP-এর ভিত্তিপ্রস্তর। সঠিক টোকেনাইজেশন ছাড়া পরবর্তী কোনো ধাপই কার্যকর হয় না। আধুনিক NLP সিস্টেমে BPE ও WordPiece সাবওয়ার্ড টোকেনাইজেশন রাজত্ব করছে কারণ এটি অজানা শব্দ (OOV) সমস্যা দূর করে এবং ভোকাবুলারি সাইজ নিয়ন্ত্রণে রাখে। পরবর্তী পোস্টে আমরা TF-IDF দিয়ে শব্দের গুরুত্ব পরিমাপ করা শিখব।
    </p>
  `
};
