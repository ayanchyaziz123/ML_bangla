export const nlp_6_ner = {
  slug: 'nlp-6-ner',
  title: 'NER: নেমড এন্টিটি রিকগনিশন',
  description: 'ব্যক্তি, সংস্থা ও স্থান শনাক্ত করতে BIO ট্যাগিং, CRF এবং spaCy ব্যবহার শিখুন',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 13,
  content: `
<h3>NER কী?</h3>
<p><strong>Named Entity Recognition (NER)</strong> হলো একটি NLP কাজ যেখানে টেক্সট থেকে বাস্তব-বিশ্বের সত্তা (entity) শনাক্ত করা হয়। যেমন:</p>
<ul>
<li><strong>ব্যক্তি (PER):</strong> "শেখ মুজিবুর রহমান", "বিল গেটস"</li>
<li><strong>সংস্থা (ORG):</strong> "গুগল", "বাংলাদেশ ব্যাংক"</li>
<li><strong>স্থান (LOC):</strong> "ঢাকা", "প্যারিস"</li>
<li><strong>তারিখ (DATE):</strong> "২৬ মার্চ ১৯৭১"</li>
<li><strong>অর্থ (MONEY):</strong> "১০ লক্ষ টাকা"</li>
</ul>

<h3>BIO ট্যাগিং স্কিম</h3>
<p>প্রতিটি টোকেনকে তিনটি ট্যাগের একটি দেওয়া হয়:</p>
<ul>
<li><strong>B (Beginning):</strong> একটি entity-র শুরু</li>
<li><strong>I (Inside):</strong> entity-র মধ্যবর্তী টোকেন</li>
<li><strong>O (Outside):</strong> কোনো entity নয়</li>
</ul>

<pre><code class="language-python"># উদাহরণ
sentence = "বারাক ওবামা আমেরিকায় জন্মেছিলেন"
bio_tags = ["B-PER", "I-PER", "B-LOC", "O", "O"]

# IOB2 স্কিম
# B-PER: ব্যক্তির নাম শুরু
# I-PER: ব্যক্তির নাম অব্যাহত
# B-LOC: স্থানের নাম শুরু
# O: entity নয়
</code></pre>

<h3>spaCy দিয়ে NER</h3>

<pre><code class="language-python">import spacy

# ইংরেজি মডেল লোড
nlp = spacy.load("en_core_web_sm")

text = "Apple was founded by Steve Jobs in Cupertino, California in 1976."
doc = nlp(text)

# entity শনাক্তকরণ
for ent in doc.ents:
    print(f"{ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")

# আউটপুট:
# Apple                | ORG        | Companies, agencies, institutions
# Steve Jobs           | PERSON     | People, including fictional
# Cupertino            | GPE        | Geopolitical entity
# California           | GPE        | Geopolitical entity
# 1976                 | DATE       | Absolute or relative dates or periods
</code></pre>

<h4>NER ভিজুয়ালাইজেশন</h4>

<pre><code class="language-python">from spacy import displacy

text = "Elon Musk founded SpaceX in 2002 in Hawthorne, California."
doc = nlp(text)

# Jupyter Notebook বা HTML-এ দেখান
displacy.render(doc, style="ent", jupyter=True)

# HTML হিসেবে সেভ করুন
html = displacy.render(doc, style="ent", page=True)
with open("ner_visualization.html", "w") as f:
    f.write(html)
</code></pre>

<h3>কাস্টম NER ট্রেনিং</h3>

<pre><code class="language-python">import spacy
from spacy.training import Example
import random

# ট্রেনিং ডেটা
TRAIN_DATA = [
    ("ঢাকায় গুগলের নতুন অফিস খোলা হয়েছে।", {
        "entities": [(0, 4, "LOC"), (7, 12, "ORG")]
    }),
    ("রাফি সাহেব বাংলাদেশ ব্যাংকের গভর্নর।", {
        "entities": [(0, 11, "PER"), (12, 28, "ORG")]
    }),
    ("আইবিএম চট্টগ্রামে বিনিয়োগ করবে।", {
        "entities": [(0, 5, "ORG"), (6, 16, "LOC")]
    }),
]

# ফাঁকা মডেল তৈরি
nlp = spacy.blank("bn")  # বাংলার জন্য
ner = nlp.add_pipe("ner")

# লেবেল যোগ করুন
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# ট্রেনিং
nlp.begin_training()
losses = {}

for epoch in range(30):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses, drop=0.3)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {losses['ner']:.4f}")

# মডেল সেভ
nlp.to_disk("custom_ner_model")
print("মডেল সেভ করা হয়েছে!")
</code></pre>

<h3>BERT-ভিত্তিক NER</h3>

<pre><code class="language-python">from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# পূর্ব-প্রশিক্ষিত NER মডেল
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# NER পাইপলাইন
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer,
                         aggregation_strategy="simple")

text = "My name is Clara and I live in Berkeley, California."
results = ner_pipeline(text)

for entity in results:
    print(f"শব্দ: {entity['word']:15} | "
          f"ট্যাগ: {entity['entity_group']:10} | "
          f"স্কোর: {entity['score']:.3f}")
</code></pre>

<h3>NER মূল্যায়ন</h3>

<pre><code class="language-python">from seqeval.metrics import classification_report

# সত্যিকারের ও ভবিষ্যদ্বাণী করা ট্যাগ
y_true = [["O", "B-PER", "I-PER", "O", "B-LOC"],
           ["B-ORG", "O", "O", "B-DATE"]]

y_pred = [["O", "B-PER", "I-PER", "O", "O"],
           ["B-ORG", "O", "O", "B-DATE"]]

print(classification_report(y_true, y_pred))
# precision, recall, F1 প্রতিটি entity টাইপের জন্য
</code></pre>

<h4>NER অ্যাপ্লিকেশন</h4>
<ul>
<li>সংবাদ নিবন্ধ থেকে ব্যক্তি ও সংস্থার নাম বের করা</li>
<li>চিকিৎসা রেকর্ড থেকে রোগের নাম ও ওষুধ শনাক্ত করা</li>
<li>আইনি দলিল থেকে পক্ষের নাম ও তারিখ বের করা</li>
<li>প্রশ্ন-উত্তর সিস্টেমে entity-র উপর ভিত্তি করে অনুসন্ধান</li>
</ul>
`
};
