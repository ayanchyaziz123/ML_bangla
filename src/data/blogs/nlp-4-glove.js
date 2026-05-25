export const nlp_4_glove = {
  slug: 'nlp-4-glove',
  title: 'GloVe ও FastText: উন্নত ওয়ার্ড এমবেডিং',
  description: 'GloVe-এর global co-occurrence matrix, FastText-এর subword embedding এবং OOV সমাধান — Word2Vec, GloVe ও FastText তুলনামূলক বিশ্লেষণ।',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 12,
  content: `
    <h3>১. Word2Vec-এর সীমাবদ্ধতা</h3>
    <p>
      Word2Vec অসাধারণ হলেও একটি মৌলিক সমস্যা আছে: এটি শুধু <strong>local context</strong> দেখে — প্রতিটি শব্দের জন্য একটি নির্দিষ্ট window-এর মধ্যে। কিন্তু ভাষার অনেক তথ্য global — পুরো corpus জুড়ে শব্দ একসাথে কতবার আসে।
    </p>
    <p>
      উদাহরণ: "ice" এবং "steam" উভয়ই "water" শব্দের সাথে সমান ঘন ঘন আসে, কিন্তু "solid" শব্দটি "ice"-এর সাথে বেশি আসে। এই global co-occurrence তথ্য Word2Vec উপেক্ষা করে।
    </p>
    <p>
      GloVe এই সমস্যার সমাধান করে।
    </p>

    <h3>২. GloVe: Global Vectors for Word Representation</h3>
    <p>
      <strong>GloVe</strong> (Global Vectors) ২০১৪ সালে Stanford-এ Jeffrey Pennington, Richard Socher ও Christopher Manning তৈরি করেন। এটি Word2Vec-এর local context এবং LSA (Latent Semantic Analysis)-এর global statistics উভয়ের সুবিধা একত্রিত করে।
    </p>

    <h4>২.১ Co-occurrence Matrix</h4>
    <p>
      GloVe প্রথমে একটি <strong>global word-word co-occurrence matrix</strong> তৈরি করে। Matrix-এর প্রতিটি cell X(i,j) বলে: শব্দ i এবং শব্দ j পুরো corpus-এ একসাথে কতবার এসেছে (নির্দিষ্ট window-এর মধ্যে)।
    </p>
    <pre><code">import numpy as np
from collections import defaultdict

def build_cooccurrence_matrix(corpus, window_size=2):
    """
    সরল co-occurrence matrix তৈরি
    """
    # Vocabulary তৈরি
    vocab = set()
    for sentence in corpus:
        for word in sentence:
            vocab.add(word)

    word2idx = {w: i for i, w in enumerate(sorted(vocab))}
    n = len(vocab)
    matrix = np.zeros((n, n))

    for sentence in corpus:
        for i, word in enumerate(sentence):
            # Window-এর মধ্যে প্রতিটি context শব্দের সাথে count বাড়ানো
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    matrix[word2idx[word]][word2idx[sentence[j]]] += 1

    return matrix, word2idx

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["cats", "and", "dogs", "are", "pets"],
]

matrix, word2idx = build_cooccurrence_matrix(corpus)
print("Vocabulary:", list(word2idx.keys()))
print("Matrix shape:", matrix.shape)</code></pre>

    <h4>২.২ GloVe-এর Objective Function</h4>
    <p>
      GloVe-এর মূল ধারণা: যদি দুটি শব্দ i, j-এর co-occurrence ratio অন্য শব্দ k-এর সাথে তুলনা করা হয়, তাহলে সেই ratio তাদের vectors-এর dot product দিয়ে প্রকাশ করা যায়।
    </p>
    <p>
      <strong>Objective:</strong> Σ f(X_ij) (w_i · w_j + b_i + b_j - log X_ij)²
    </p>
    <p>
      যেখানে f(X_ij) একটি weighting function যা অতি ঘন co-occurrence-এর প্রভাব কমায়।
    </p>

    <h3>৩. GloVe Pre-trained Embeddings লোড করা</h3>
    <pre><code">import numpy as np
import gensim.downloader as api

# GloVe embeddings লোড (gensim এগুলো নিজেই ডাউনলোড করে)
print("GloVe 50d লোড হচ্ছে...")
glove_50 = api.load("glove-wiki-gigaword-50")

print("GloVe 100d লোড হচ্ছে...")
glove_100 = api.load("glove-wiki-gigaword-100")

# Vocabulary size
print(f"Vocabulary size: {len(glove_50)}")

# Word similarity
print("GloVe similarity examples:")
print(f"  king-queen: {glove_50.similarity('king', 'queen'):.4f}")
print(f"  cat-dog: {glove_50.similarity('cat', 'dog'):.4f}")
print(f"  cat-car: {glove_50.similarity('cat', 'car'):.4f}")

# Word analogy
result = glove_50.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=5
)
print("\nking - man + woman:")
for word, score in result:
    print(f"  {word}: {score:.4f}")</code></pre>

    <h3>৪. FastText: Subword Embeddings</h3>
    <p>
      Word2Vec এবং GloVe উভয়েরই একটি মারাত্মক সমস্যা আছে: <strong>OOV (Out-of-Vocabulary)</strong> শব্দ। Training-এ না দেখা শব্দের কোনো embedding নেই।
    </p>
    <p>
      <strong>FastText</strong> (Facebook AI Research, ২০১৬) এই সমস্যা সমাধান করে <strong>subword (character n-gram) embeddings</strong> দিয়ে।
    </p>

    <h4>৪.১ FastText কীভাবে কাজ করে</h4>
    <p>
      FastText প্রতিটি শব্দকে character n-gram-এর সেট হিসেবে দেখে।
    </p>
    <p>
      উদাহরণ: "where" (n=3,4,5,6) → [&lt;wh, whe, her, ere, re&gt;, &lt;whe, wher, here, ere&gt;, ...]
    </p>
    <p>
      শব্দের final embedding = সব n-gram vectors-এর গড়।
    </p>
    <p>
      এর ফলে:
    </p>
    <p>
      <strong>OOV সমাধান:</strong> "unhappiness" training-এ না থাকলেও "un", "happy", "ness" n-gram থেকে embedding তৈরি হয়।<br/>
      <strong>Morphology ধরা যায়:</strong> "run", "running", "runner" একে অপরের সাথে n-gram শেয়ার করে।<br/>
      <strong>Rare words-এ ভালো:</strong> বিরল শব্দও n-gram থেকে ভালো representation পায়।
    </p>

    <h3>৫. gensim দিয়ে FastText</h3>
    <pre><code">from gensim.models import FastText
from gensim.utils import simple_preprocess

# Training data
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "i love machine learning and deep learning",
    "natural language processing is fascinating",
    "cats and dogs are common household pets",
    "running jogging and swimming are healthy activities",
    "the runner ran quickly through the running track",
    "unhappy people are often unproductive at work",
    "happiness comes from within yourself",
]

tokenized = [simple_preprocess(s) for s in sentences]

# FastText মডেল তৈরি
model = FastText(
    sentences=tokenized,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,          # Skip-gram
    min_n=3,       # minimum n-gram size
    max_n=6,       # maximum n-gram size
    epochs=50
)

# OOV শব্দের embedding তৈরি করা (Word2Vec পারে না!)
oov_word = "unhappiness"
print(f"OOV শব্দ '{oov_word}'-এর embedding তৈরি হলো!")
print(f"Vector shape: {model.wv[oov_word].shape}")

# N-gram থেকে embedding তৈরি
similar_to_oov = model.wv.most_similar(oov_word, topn=5)
print(f"\n'{oov_word}'-এর মতো শব্দ:")
for word, score in similar_to_oov:
    print(f"  {word}: {score:.4f}")</code></pre>

    <h3>৬. Pre-trained FastText লোড করা</h3>
    <pre><code">import fasttext.util
import fasttext

# Facebook-এর pre-trained FastText ডাউনলোড (English)
# fasttext.util.download_model('en', if_exists='ignore')
# ft_model = fasttext.load_model('cc.en.300.bin')

# গিজম দিয়ে
import gensim.downloader as api
# ft_wiki = api.load("fasttext-wiki-news-subwords-300")

# বা সরাসরি .vec ফাইল লোড করা
from gensim.models import KeyedVectors

# fasttext_model = KeyedVectors.load_word2vec_format(
#     'wiki-news-300d-1M-subword.vec', binary=False
# )

# Morphological similarity দেখা
# words_morph = ['run', 'running', 'runner', 'ran', 'runs']
# for w in words_morph:
#     sim = ft_model.get_nearest_neighbors(w, k=3)
#     print(f"{w}: {sim}")</code></pre>

    <h3>৭. Word2Vec বনাম GloVe বনাম FastText</h3>
    <p>
      <strong>Word2Vec:</strong><br/>
      Approach: Local context window (CBOW/Skip-gram)<br/>
      OOV: সমর্থন নেই<br/>
      Speed: দ্রুত training<br/>
      Best for: General purpose, large corpus<br/>
      Limitation: Global statistics উপেক্ষা করে
    </p>
    <p>
      <strong>GloVe:</strong><br/>
      Approach: Global co-occurrence matrix factorization<br/>
      OOV: সমর্থন নেই<br/>
      Speed: মাঝারি<br/>
      Best for: Word analogy tasks, large corpus<br/>
      Limitation: Subword তথ্য নেই
    </p>
    <p>
      <strong>FastText:</strong><br/>
      Approach: Subword (character n-gram) embeddings<br/>
      OOV: সমর্থন আছে<br/>
      Speed: ধীর training কিন্তু inference দ্রুত<br/>
      Best for: Morphologically rich languages, rare words<br/>
      Limitation: Large model size
    </p>
    <pre><code">from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import numpy as np

# Pre-trained models লোড
w2v = api.load("word2vec-google-news-300")  # Google News
glove = api.load("glove-wiki-gigaword-100")  # Wikipedia + Gigaword

# Comparison: Analogy task
analogy_pairs = [
    (['king', 'woman'], ['man']),  # → queen
    (['france', 'berlin'], ['paris']),  # → germany
    (['good', 'worse'], ['better']),  # → bad
]

print("Analogy Comparison:")
print(f"{'Analogy':<30} {'Word2Vec':<15} {'GloVe':<15}")
print("-" * 60)

for pos, neg in analogy_pairs:
    try:
        w2v_result = w2v.most_similar(positive=pos, negative=neg, topn=1)[0][0]
    except:
        w2v_result = "N/A"
    try:
        glove_result = glove.most_similar(positive=pos, negative=neg, topn=1)[0][0]
    except:
        glove_result = "N/A"

    label = f"{pos} - {neg}"
    print(f"{str(label):<30} {w2v_result:<15} {glove_result:<15}")</code></pre>

    <h3>৮. Multilingual FastText</h3>
    <p>
      Facebook Research ১৫৭টি ভাষার জন্য pre-trained FastText embeddings প্রকাশ করেছে। বাংলা ভাষার জন্যও FastText available!
    </p>
    <pre><code">import fasttext

# বাংলা FastText ডাউনলোড
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.bn.vec

# বা directly:
# fasttext.util.download_model('bn', if_exists='ignore')
# bn_model = fasttext.load_model('cc.bn.300.bin')

# বাংলা শব্দের embedding
# vector = bn_model.get_word_vector('বাংলাদেশ')
# print("বাংলাদেশ vector shape:", vector.shape)

# Similar Bengali words
# neighbors = bn_model.get_nearest_neighbors('বাংলাদেশ')
# print("বাংলাদেশ-এর কাছের শব্দ:", neighbors)</code></pre>

    <h3>৯. কোন Embedding কখন ব্যবহার করবেন?</h3>
    <p>
      <strong>General English NLP:</strong> GloVe (Wikipedia+Gigaword) বা Word2Vec (Google News) — উভয়ই ভালো।<br/>
      <strong>Morphologically rich language:</strong> FastText সর্বোত্তম (বাংলা, তুর্কি, আরবি, জার্মান)।<br/>
      <strong>Domain-specific text:</strong> সেই domain-এর corpus দিয়ে নিজস্ব embedding train করুন।<br/>
      <strong>State-of-the-art NLP:</strong> BERT/RoBERTa-এর contextual embeddings (পরবর্তী সিরিজে)।
    </p>

    <h3>১০. সারসংক্ষেপ</h3>
    <p>
      GloVe global co-occurrence statistics ব্যবহার করে Word2Vec-এর চেয়ে ভালো word analogies capture করতে পারে। FastText subword information ব্যবহার করে OOV সমস্যা সমাধান করে এবং morphologically rich language-এ বিশেষ কার্যকর। পরবর্তী পোস্টে আমরা এই embeddings ব্যবহার করে একটি সম্পূর্ণ Sentiment Analysis প্রজেক্ট তৈরি করব।
    </p>
  `
};
