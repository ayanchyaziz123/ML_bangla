export const nlpEn = [
  {
    slug: 'nlp-1-tokenization',
    title: 'Tokenization: First Step in Text Preprocessing',
    description: 'Word/sentence tokenization, subword BPE, stemming, lemmatization with NLTK and spaCy',
    category: 'NLP Fundamentals',
    content: `
<h3>Why Tokenization?</h3>
<p>Before any NLP model can process text, it must be broken into discrete units called <strong>tokens</strong>. The choice of tokenization strategy significantly impacts model performance.</p>

<pre><code class="language-python">import nltk
import spacy
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Word tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
text = "Machine learning is fascinating. It's changing the world!"
words = word_tokenize(text)
sentences = sent_tokenize(text)
print(f"Words: {words}")
print(f"Sentences: {sentences}")

# spaCy tokenization with linguistic annotations
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
    print(f"{token.text:<15} lemma={token.lemma_:<15} pos={token.pos_}")

# Byte-Pair Encoding (BPE) — subword tokenization
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(["Machine learning is fun", "Deep learning rocks"],
                               vocab_size=100, min_frequency=1)
encoded = tokenizer.encode("Machine learning")
print(f"BPE tokens: {encoded.tokens}")
</code></pre>

<h3>Stemming vs Lemmatization</h3>
<pre><code class="language-python">from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet', quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'studies']
for w in words:
    print(f"{w:<12} stem={stemmer.stem(w):<12} lemma={lemmatizer.lemmatize(w)}")
</code></pre>
`
  },
  {
    slug: 'nlp-2-tfidf',
    title: 'TF-IDF: Measuring Word Importance',
    description: 'Bag of words, TF-IDF formula, document similarity, and sklearn TfidfVectorizer',
    category: 'NLP Fundamentals',
    content: `
<h3>TF-IDF Formula</h3>
<p><strong>TF(t,d)</strong> = count(t,d) / total_words(d)</p>
<p><strong>IDF(t)</strong> = log(N / df(t)) where N=total docs, df=docs containing term</p>
<p><strong>TF-IDF(t,d)</strong> = TF(t,d) × IDF(t)</p>

<pre><code class="language-python">from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

corpus = [
    "The cat sat on the mat",
    "The dog sat on the mat",
    "The cat chased the dog",
    "Python is great for machine learning",
]

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(corpus)
print(f"Shape: {X.shape}")

# Top words per document
feature_names = vectorizer.get_feature_names_out()
for i, doc in enumerate(corpus):
    top_features = X[i].toarray()[0].argsort()[-3:][::-1]
    top_words = [(feature_names[j], X[i, j]) for j in top_features]
    print(f"Doc {i+1}: {top_words}")

# Cosine similarity
similarity = cosine_similarity(X)
print(f"\\nDoc1 vs Doc2 similarity: {similarity[0,1]:.4f}")
print(f"Doc1 vs Doc4 similarity: {similarity[0,3]:.4f}")
</code></pre>
`
  },
  {
    slug: 'nlp-3-word2vec',
    title: 'Word2Vec: Dense Word Representations',
    description: 'CBOW, Skip-gram, negative sampling, word analogies, and gensim implementation',
    category: 'NLP Fundamentals',
    content: `
<h3>Distributional Hypothesis</h3>
<p>"You shall know a word by the company it keeps." Words appearing in similar contexts have similar meanings — this is the foundation of Word2Vec.</p>

<pre><code class="language-python">from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# Prepare sentences
sentences = [
    ["king", "man", "woman", "queen", "royal"],
    ["paris", "france", "london", "england", "capital"],
    ["doctor", "hospital", "nurse", "medicine", "patient"],
    ["python", "programming", "code", "developer", "software"],
]

# Train Word2Vec (Skip-gram with negative sampling)
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1,
                 sg=1,       # 1=Skip-gram, 0=CBOW
                 negative=5, # Negative sampling
                 epochs=100, workers=4)

# Word similarity
print(model.wv.most_similar('king', topn=3))

# Word analogy: king - man + woman = ?
result = model.wv.most_similar(positive=['king', 'woman'],
                                negative=['man'], topn=3)
print(f"king - man + woman = {result}")

# Cosine similarity
sim = model.wv.similarity('paris', 'london')
print(f"paris ↔ london similarity: {sim:.4f}")
</code></pre>
`
  },
  {
    slug: 'nlp-4-glove',
    title: 'GloVe and FastText: Advanced Word Embeddings',
    description: 'Global co-occurrence with GloVe, subword embeddings with FastText, OOV handling',
    category: 'NLP Fundamentals',
    content: `
<h3>GloVe vs Word2Vec</h3>
<table>
<tr><th>Feature</th><th>Word2Vec</th><th>GloVe</th><th>FastText</th></tr>
<tr><td>Method</td><td>Local window</td><td>Global co-occurrence</td><td>Subword n-grams</td></tr>
<tr><td>OOV words</td><td>No</td><td>No</td><td>Yes</td></tr>
<tr><td>Morphology</td><td>No</td><td>No</td><td>Yes</td></tr>
<tr><td>Training</td><td>Online</td><td>Batch</td><td>Online</td></tr>
</table>

<pre><code class="language-python">import numpy as np

def load_glove(path):
    """Load pre-trained GloVe embeddings"""
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

# embeddings = load_glove('glove.6B.100d.txt')

# FastText - handles OOV words via subword n-grams
from gensim.models import FastText

sentences = [["learning", "machine", "learning", "deep"],
             ["natural", "language", "processing"]]
ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=50)

# OOV example - FastText can handle unseen words
print(ft_model.wv['machinelearning'])  # Not in training data but FastText handles it
print(ft_model.wv.most_similar('learning', topn=3))
</code></pre>
`
  },
  {
    slug: 'nlp-5-sentiment',
    title: 'Sentiment Analysis Project',
    description: 'IMDB sentiment classification comparing Naive Bayes, Logistic Regression, and LSTM',
    category: 'NLP Fundamentals',
    content: `
<h3>Complete Sentiment Pipeline</h3>
<pre><code class="language-python">from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from datasets import load_dataset

dataset = load_dataset('imdb')
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# Model comparison
models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ]),
}

results = {}
for name, model in models.items():
    model.fit(train_texts, train_labels)
    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# BERT approach (much better, needs GPU)
# from transformers import pipeline
# sentiment = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
# print(sentiment("This movie was absolutely fantastic!"))
</code></pre>
`
  },
  {
    slug: 'nlp-6-ner',
    title: 'Named Entity Recognition (NER)',
    description: 'BIO tagging, CRF models, spaCy NER, custom training, and BERT-based NER',
    category: 'NLP Fundamentals',
    content: `
<h3>NER with spaCy</h3>
<pre><code class="language-python">import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

texts = [
    "Apple was founded by Steve Jobs in Cupertino, California in 1976.",
    "Elon Musk's Tesla manufactures electric vehicles in Fremont.",
    "The World Health Organization reported 5 million cases in 2023.",
]

for text in texts:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Text: {text[:50]}...")
    print(f"Entities: {entities}")
    print()
</code></pre>

<h3>BERT-based NER</h3>
<pre><code class="language-python">from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

result = ner("Jeff Bezos founded Amazon in Bellevue, Washington.")
for entity in result:
    print(f"{entity['word']:<20} {entity['entity_group']:<10} {entity['score']:.3f}")

# Evaluation with seqeval
from seqeval.metrics import classification_report
y_true = [["O", "B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "I-LOC"]]
y_pred = [["O", "B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]]
print(classification_report(y_true, y_pred))
</code></pre>
`
  },
];
