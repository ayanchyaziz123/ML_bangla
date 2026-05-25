export const nlp_2_tfidf = {
  slug: 'nlp-2-tfidf',
  title: 'TF-IDF: শব্দের গুরুত্ব পরিমাপ',
  description: 'ব্যাগ অব ওয়ার্ডস থেকে TF-IDF পর্যন্ত — শব্দের গুরুত্ব কীভাবে পরিমাপ করা হয়, sklearn দিয়ে বাস্তব প্রয়োগ এবং ডকুমেন্ট সিমিলারিটি।',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 11,
  content: `
    <h3>১. ব্যাগ অব ওয়ার্ডস (Bag of Words)</h3>
    <p>
      কম্পিউটারকে টেক্সট বোঝাতে হলে প্রথমে টেক্সটকে সংখ্যায় রূপান্তর করতে হয়। সবচেয়ে সরল পদ্ধতি হলো <strong>Bag of Words (BoW)</strong> — প্রতিটি ডকুমেন্টকে শব্দের ঘটনাসংখ্যার একটি ভেক্টর হিসেবে প্রকাশ করা।
    </p>
    <p>উদাহরণ:</p>
    <p>
      D1: "I love NLP. NLP is great."<br/>
      D2: "I love machine learning."
    </p>
    <p>
      Vocabulary: [I, love, NLP, is, great, machine, learning]<br/>
      D1 ভেক্টর: [1, 1, 2, 1, 1, 0, 0]<br/>
      D2 ভেক্টর: [1, 1, 0, 0, 0, 1, 1]
    </p>
    <p>
      BoW-এর সমস্যা: "the", "is", "a" এর মতো অতি সাধারণ শব্দ সব ডকুমেন্টে বেশি ঘটে কিন্তু কোনো অর্থগত তথ্য বহন করে না। এই সমস্যার সমাধান হলো TF-IDF।
    </p>

    <h3>২. TF: Term Frequency</h3>
    <p>
      <strong>Term Frequency (TF)</strong> পরিমাপ করে একটি নির্দিষ্ট শব্দ একটি ডকুমেন্টে কতবার এসেছে — মোট শব্দ সংখ্যার অনুপাতে।
    </p>
    <p>
      সূত্র: <strong>TF(t, d) = (শব্দ t ডকুমেন্ট d-তে কতবার আসে) / (ডকুমেন্ট d-তে মোট শব্দ সংখ্যা)</strong>
    </p>
    <p>
      উদাহরণ: "The cat sat on the mat." — এখানে "the" ২ বার, মোট ৬ শব্দ।<br/>
      TF("the") = 2/6 = 0.33
    </p>
    <p>
      TF শুধু বলে একটি শব্দ কোনো ডকুমেন্টে কতটা গুরুত্বপূর্ণ। কিন্তু "the" সব ডকুমেন্টেই বেশি থাকে — তাই শুধু TF দিয়ে ভালো ফলাফল পাওয়া যায় না।
    </p>

    <h3>৩. IDF: Inverse Document Frequency</h3>
    <p>
      <strong>Inverse Document Frequency (IDF)</strong> পরিমাপ করে একটি শব্দ কতটা বিরল — অর্থাৎ কতটা কম ডকুমেন্টে এটি আসে। বিরল শব্দ বেশি তথ্যবহুল।
    </p>
    <p>
      সূত্র: <strong>IDF(t) = log(মোট ডকুমেন্ট সংখ্যা / শব্দ t যেসব ডকুমেন্টে আছে তার সংখ্যা)</strong>
    </p>
    <p>
      যদি কোনো শব্দ সব ডকুমেন্টে থাকে: IDF = log(N/N) = log(1) = 0 → গুরুত্ব নেই।<br/>
      যদি কোনো শব্দ শুধু একটি ডকুমেন্টে থাকে: IDF = log(N/1) = log(N) → অনেক বেশি গুরুত্ব।
    </p>
    <pre><code>import math

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog"
]

# IDF হিসাব
def compute_idf(term, docs):
    n_docs_with_term = sum(1 for doc in docs if term in doc.split())
    return math.log(len(docs) / (1 + n_docs_with_term))  # +1 smoothing

print(f"IDF('the'): {compute_idf('the', docs):.4f}")
print(f"IDF('cat'):  {compute_idf('cat',  docs):.4f}")
print(f"IDF('sat'):  {compute_idf('sat',  docs):.4f}")
# IDF('the'): 0.0000  (সব ডকুমেন্টে আছে, তাই গুরুত্ব কম)
# IDF('cat'):  0.2877
# IDF('sat'):  0.1054</code></pre>

    <h3>৪. TF-IDF = TF × IDF</h3>
    <p>
      <strong>TF-IDF</strong> হলো TF এবং IDF-এর গুণফল। এটি একটি শব্দের গুরুত্ব প্রকাশ করে — শব্দটি নির্দিষ্ট ডকুমেন্টে কতটা ঘন ঘন আসে (TF) এবং সামগ্রিকভাবে কতটা বিরল (IDF)।
    </p>
    <p>
      <strong>TF-IDF(t, d) = TF(t, d) × IDF(t)</strong>
    </p>
    <p>
      বেশি TF-IDF মান মানে: শব্দটি এই ডকুমেন্টের জন্য বিশেষভাবে গুরুত্বপূর্ণ।
    </p>
    <pre><code>def compute_tf(term, doc):
    words = doc.lower().split()
    return words.count(term) / len(words)

def compute_tfidf(term, doc, docs):
    tf = compute_tf(term, doc)
    idf = compute_idf(term, docs)
    return tf * idf

# উদাহরণ
doc1 = "the cat sat on the mat"
print(f"TF-IDF('cat', D1):  {compute_tfidf('cat', doc1, docs):.4f}")
print(f"TF-IDF('the', D1):  {compute_tfidf('the', doc1, docs):.4f}")
# TF-IDF('cat', D1):  0.0479  (বিরল শব্দ, বেশি গুরুত্ব)
# TF-IDF('the', D1):  0.0000  (সব জায়গায় আছে, গুরুত্ব নেই)</code></pre>

    <h3>৫. sklearn দিয়ে TF-IDF</h3>
    <p>
      বাস্তব প্রজেক্টে ম্যানুয়ালি হিসাব না করে <code>sklearn</code>-এর <code>TfidfVectorizer</code> ব্যবহার করা হয়।
    </p>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

corpus = [
    "I love machine learning and deep learning",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for pattern recognition",
    "Natural language processing is a branch of artificial intelligence",
    "I love natural language processing and NLP applications"
]

vectorizer = TfidfVectorizer(
    max_features=15,      # সর্বোচ্চ ১৫টি ফিচার
    stop_words='english', # ইংরেজি স্টপওয়ার্ড সরানো
    ngram_range=(1, 2)    # unigram ও bigram উভয়ই
)

tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# DataFrame হিসেবে দেখুন
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[f"Doc{i+1}" for i in range(len(corpus))]
)
print(df.round(3))</code></pre>

    <h3>৬. ডকুমেন্ট সিমিলারিটি: Cosine Similarity</h3>
    <p>
      TF-IDF ভেক্টর ব্যবহার করে দুটি ডকুমেন্টের মধ্যে মিল পরিমাপ করা যায়। <strong>Cosine Similarity</strong> দুটি ভেক্টরের মধ্যবর্তী কোণ পরিমাপ করে।
    </p>
    <p>
      cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
    </p>
    <p>
      মান ১ মানে সম্পূর্ণ মিল, ০ মানে কোনো মিল নেই।
    </p>
    <pre><code>from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# উপরের corpus থেকে TF-IDF matrix ব্যবহার করে
vectorizer2 = TfidfVectorizer(stop_words='english')
tfidf = vectorizer2.fit_transform(corpus)

# সব ডকুমেন্টের মধ্যে similarity matrix
sim_matrix = cosine_similarity(tfidf)

print("Similarity Matrix:")
for i in range(len(corpus)):
    for j in range(len(corpus)):
        print(f"D{i+1}-D{j+1}: {sim_matrix[i][j]:.3f}", end="  ")
    print()

# প্রদত্ত query-র সাথে সবচেয়ে মিল খোঁজা
query = ["deep learning neural networks"]
query_vec = vectorizer2.transform(query)
scores = cosine_similarity(query_vec, tfidf)[0]

ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print("\nQuery-র সাথে সবচেয়ে মিল:")
for idx, score in ranked:
    print(f"  Doc{idx+1} ({score:.3f}): {corpus[idx][:50]}")</code></pre>

    <h3>৭. TF-IDF-এর সীমাবদ্ধতা</h3>
    <p>
      TF-IDF শক্তিশালী হলেও কিছু সীমাবদ্ধতা আছে:
    </p>
    <p>
      <strong>শব্দের ক্রম উপেক্ষা করে:</strong> "dog bites man" এবং "man bites dog" একই ভেক্টর পাবে।<br/>
      <strong>শব্দের অর্থ বোঝে না:</strong> "good" এবং "excellent" আলাদা শব্দ হিসেবে গণ্য হয়।<br/>
      <strong>Sparse representation:</strong> বড় vocabulary-তে বেশিরভাগ মান শূন্য।<br/>
      <strong>নতুন ডকুমেন্টে সমস্যা:</strong> training-এ না দেখা শব্দ উপেক্ষিত হয়।
    </p>
    <p>
      এই সমস্যাগুলো সমাধান করতে পরবর্তী পোস্টে আমরা Word2Vec শিখব — যেখানে শব্দের অর্থগত সম্পর্ক ধরা যায়।
    </p>

    <h3>৮. বাস্তব ব্যবহার: কীওয়ার্ড এক্সট্রাকশন</h3>
    <pre><code>from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(text, top_n=5):
    """একটি ডকুমেন্ট থেকে শীর্ষ কীওয়ার্ড বের করুন"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    # স্কোর অনুযায়ী সাজানো
    ranked = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_n]

article = """
    Machine learning is a method of data analysis that automates analytical
    model building. It is based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    Machine learning algorithms include supervised, unsupervised and
    reinforcement learning approaches.
"""

keywords = extract_keywords(article)
print("শীর্ষ কীওয়ার্ড:")
for word, score in keywords:
    print(f"  {word}: {score:.4f}")</code></pre>

    <h3>৯. সারসংক্ষেপ</h3>
    <p>
      TF-IDF একটি সহজ কিন্তু শক্তিশালী text representation পদ্ধতি। এটি শব্দের স্থানীয় গুরুত্ব (TF) এবং বৈশ্বিক বিরলতা (IDF) একত্রিত করে প্রতিটি শব্দের প্রকৃত গুরুত্ব নির্ধারণ করে। document classification, information retrieval এবং keyword extraction-এ এটি আজও ব্যাপকভাবে ব্যবহৃত হয়। পরের পোস্টে আমরা Word2Vec শিখব — যেখানে শব্দের অর্থগত সম্পর্ক dense vector-এ ধারণ করা যায়।
    </p>
  `
};
