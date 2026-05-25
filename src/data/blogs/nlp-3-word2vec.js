export const nlp_3_word2vec = {
  slug: 'nlp-3-word2vec',
  title: 'Word2Vec: শব্দের ভেক্টর উপস্থাপনা',
  description: 'Distributional hypothesis, CBOW ও Skip-gram আর্কিটেকচার, নেগেটিভ স্যাম্পলিং, এবং gensim দিয়ে word analogies ও similarity — সম্পূর্ণ গাইড।',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 13,
  content: `
    <h3>১. Word2Vec কেন দরকার?</h3>
    <p>
      TF-IDF-এ "king" এবং "queen" সম্পূর্ণ আলাদা শব্দ — তাদের মধ্যে কোনো সম্পর্ক নেই। কিন্তু আমরা জানি এই দুটি শব্দ অর্থগতভাবে কাছাকাছি। <strong>Word2Vec</strong> এই সমস্যা সমাধান করে — এটি প্রতিটি শব্দকে একটি dense, low-dimensional vector-এ রূপান্তর করে, যেখানে অর্থগতভাবে কাছাকাছি শব্দগুলোর vector-ও কাছাকাছি থাকে।
    </p>
    <p>
      ২০১৩ সালে Google-এর Tomas Mikolov এই অ্যালগরিদম প্রবর্তন করেন। এটি NLP-এর ইতিহাসে একটি বিপ্লব ঘটায়।
    </p>

    <h3>২. Distributional Hypothesis</h3>
    <p>
      Word2Vec-এর মূল ভিত্তি হলো <strong>Distributional Hypothesis</strong>: "একটি শব্দের অর্থ তার পরিবেশ (context) দ্বারা নির্ধারিত হয়।" — John Rupert Firth (১৯৫৭)।
    </p>
    <p>
      অর্থাৎ, যদি "dog" এবং "cat" একই ধরনের বাক্যে একই ধরনের প্রেক্ষাপটে আসে ("I have a ___ at home", "___ is hungry"), তাহলে তাদের অর্থ কাছাকাছি।
    </p>
    <p>
      Word2Vec এই নীতি ব্যবহার করে একটি শব্দের vector শেখে তার আশেপাশের শব্দগুলো দেখে।
    </p>

    <h3>৩. CBOW (Continuous Bag of Words)</h3>
    <p>
      <strong>CBOW</strong> মডেল context শব্দগুলো দিয়ে target শব্দ predict করে।
    </p>
    <p>
      উদাহরণ: "The ___ sat on the mat" — context: [The, sat, on, the, mat] → target: cat
    </p>
    <p>
      CBOW আর্কিটেকচার:
    </p>
    <p>
      Input layer: context শব্দের one-hot vectors<br/>
      Hidden layer: context vectors-এর গড় (projection layer)<br/>
      Output layer: softmax দিয়ে target শব্দ predict
    </p>
    <p>
      CBOW দ্রুত train হয় এবং ঘন ঘন ব্যবহৃত শব্দে ভালো কাজ করে।
    </p>

    <h3>৪. Skip-gram</h3>
    <p>
      <strong>Skip-gram</strong> উল্টো কাজ করে — একটি target শব্দ দিয়ে context শব্দগুলো predict করে।
    </p>
    <p>
      উদাহরণ: "cat" → predict: [The, sat, on, the, mat] (window size = 2)
    </p>
    <p>
      Skip-gram আর্কিটেকচার:
    </p>
    <p>
      Input: target শব্দের one-hot vector<br/>
      Hidden: embedding layer (শেখার লক্ষ্য এটিই)<br/>
      Output: প্রতিটি context শব্দের softmax probability
    </p>
    <p>
      Skip-gram ধীর কিন্তু বিরল শব্দে CBOW-এর চেয়ে ভালো কাজ করে।
    </p>

    <h3>৫. Negative Sampling</h3>
    <p>
      Word2Vec-এর একটি বড় চ্যালেঞ্জ: প্রতিটি training step-এ পুরো vocabulary-র উপর softmax হিসাব করা অনেক ব্যয়সাপেক্ষ (vocabulary ৫০,০০০+ শব্দ হলে)।
    </p>
    <p>
      <strong>Negative Sampling</strong> এই সমস্যা সমাধান করে। প্রতিটি positive pair-এর জন্য random কিছু "negative" শব্দ নেওয়া হয় এবং শুধুমাত্র এই ছোট সেটের উপর binary classification করা হয়।
    </p>
    <p>
      Training objective: positive pair-এর score maximize করো, negative pair-এর score minimize করো।
    </p>
    <p>
      Negative sampling সংখ্যা সাধারণত ৫-২০ রাখা হয় ছোট dataset-এ এবং ২-৫ বড় dataset-এ।
    </p>

    <h3>৬. gensim দিয়ে Word2Vec</h3>
    <pre><code>from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import requests

# নমুনা corpus (বাস্তবে আরও বড় corpus ব্যবহার করুন)
sentences = [
    "the king ruled the kingdom with wisdom",
    "the queen governed the palace with grace",
    "the man went to work every morning",
    "the woman went to school every day",
    "paris is the capital of france",
    "berlin is the capital of germany",
    "london is the capital of england",
    "cats and dogs are common pets",
    "cats like to eat fish",
    "dogs like to eat meat",
    "machine learning is a branch of artificial intelligence",
    "deep learning uses neural networks",
    "natural language processing studies human language",
]

# টোকেনাইজেশন
tokenized = [simple_preprocess(sent) for sent in sentences]

# Word2Vec মডেল ট্রেন করা
model = Word2Vec(
    sentences=tokenized,
    vector_size=100,    # embedding dimension
    window=5,           # context window size
    min_count=1,        # minimum word frequency
    workers=4,          # parallel threads
    sg=1,               # 1=Skip-gram, 0=CBOW
    negative=5,         # negative sampling count
    epochs=100
)

# মডেল সংরক্ষণ
model.save("word2vec_model.bin")

print("Vocabulary size:", len(model.wv))
print("'king' vector shape:", model.wv['king'].shape)</code></pre>

    <h3>৭. Word Similarity</h3>
    <pre><code># সবচেয়ে মিল শব্দ খোঁজা
similar_to_king = model.wv.most_similar('king', topn=5)
print("'king'-এর মতো শব্দ:")
for word, score in similar_to_king:
    print(f"  {word}: {score:.4f}")

# দুটি শব্দের মধ্যে সিমিলারিটি
sim = model.wv.similarity('cat', 'dog')
print(f"\n'cat' ও 'dog'-এর similarity: {sim:.4f}")

sim2 = model.wv.similarity('cat', 'machine')
print(f"'cat' ও 'machine'-এর similarity: {sim2:.4f}")

# মেলে না এমন শব্দ খোঁজা
odd_one_out = model.wv.doesnt_match(['cat', 'dog', 'fish', 'learning'])
print(f"\nমেলে না: {odd_one_out}")</code></pre>

    <h3>৮. Word Analogies: king - man + woman = queen</h3>
    <p>
      Word2Vec-এর সবচেয়ে বিখ্যাত বৈশিষ্ট্য হলো word analogies — vector arithmetic দিয়ে অর্থগত সম্পর্ক প্রকাশ করা।
    </p>
    <p>
      <strong>vec(king) - vec(man) + vec(woman) ≈ vec(queen)</strong>
    </p>
    <p>
      এই ধারণাটি দেখায় যে Word2Vec শুধু শব্দ নয়, তাদের মধ্যকার সম্পর্কও শিখতে পারে।
    </p>
    <pre><code># Word analogies
# king - man + woman = ?
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=3
)
print("king - man + woman =", result)

# france - paris + berlin = germany?
result2 = model.wv.most_similar(
    positive=['france', 'berlin'],
    negative=['paris'],
    topn=3
)
print("france - paris + berlin =", result2)

# ম্যানুয়াল vector arithmetic
import numpy as np

king_vec = model.wv['king']
man_vec = model.wv['man']
woman_vec = model.wv['woman']

# Arithmetic
analogy_vec = king_vec - man_vec + woman_vec

# Cosine similarity দিয়ে সবচেয়ে কাছের শব্দ খোঁজা
from sklearn.metrics.pairwise import cosine_similarity

vocab = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[w] for w in vocab])
analogy_vec = analogy_vec.reshape(1, -1)

scores = cosine_similarity(analogy_vec, vectors)[0]
top_idx = np.argsort(scores)[::-1][:5]

print("\nManual analogy result:")
for idx in top_idx:
    if vocab[idx] not in ['king', 'man', 'woman']:
        print(f"  {vocab[idx]}: {scores[idx]:.4f}")</code></pre>

    <h3>৯. Pre-trained Word2Vec ব্যবহার</h3>
    <p>
      ছোট corpus-এ train করা Word2Vec ভালো ফলাফল দেয় না। বাস্তব প্রজেক্টে Google-এর pre-trained Word2Vec (Google News, ৩ বিলিয়ন শব্দ, ৩ মিলিয়ন vocabulary) ব্যবহার করা উচিত।
    </p>
    <pre><code">import gensim.downloader as api

# Google News pre-trained Word2Vec লোড করা (১.৬ GB)
# model = api.load("word2vec-google-news-300")

# আরও ছোট মডেল দিয়ে শুরু করুন
model_small = api.load("glove-wiki-gigaword-50")  # GloVe 50d

# এখন সত্যিকারের analogies কাজ করবে
result = model_small.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=3
)
print("king - man + woman:", result)
# [('queen', 0.85), ('princess', 0.78), ('throne', 0.72)]</code></pre>

    <h3>১০. Word2Vec Visualization</h3>
    <pre><code>import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# কিছু শব্দ নির্বাচন করুন
words = ['king', 'queen', 'man', 'woman', 'cat', 'dog',
         'france', 'germany', 'paris', 'berlin']

# Vectors নিন
vectors = np.array([model.wv[w] for w in words if w in model.wv])
words_filtered = [w for w in words if w in model.wv]

# t-SNE দিয়ে 2D-তে নামিয়ে আনুন
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(vectors)

# Plot করুন
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c='steelblue', s=100)

for i, word in enumerate(words_filtered):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]),
                fontsize=12, ha='right')

plt.title("Word2Vec Embeddings (t-SNE visualization)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150)
plt.show()</code></pre>

    <h3>১১. CBOW বনাম Skip-gram: কোনটি কখন?</h3>
    <p>
      <strong>CBOW ব্যবহার করুন যখন:</strong> Dataset বড়, দ্রুত training দরকার, ঘন ঘন ব্যবহৃত শব্দ নিয়ে কাজ।<br/>
      <strong>Skip-gram ব্যবহার করুন যখন:</strong> Dataset ছোট, বিরল শব্দের ভালো representation দরকার, সাধারণত ভালো ফলাফল দেয়।
    </p>

    <h3>১২. সারসংক্ষেপ</h3>
    <p>
      Word2Vec একটি যুগান্তকারী অ্যালগরিদম যা শব্দের অর্থগত সম্পর্ক dense vector-এ ধারণ করতে পারে। Distributional hypothesis-এর উপর ভিত্তি করে CBOW এবং Skip-gram দুটি আর্কিটেকচার এই কাজ করে। Negative sampling training-কে দ্রুত ও কার্যকর করে। পরবর্তী পোস্টে আমরা GloVe ও FastText দেখব — Word2Vec-এর উন্নত বিকল্প।
    </p>
  `
};
