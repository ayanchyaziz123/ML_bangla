export const llm_3_rag = {
  slug: 'llm-3-rag',
  title: 'RAG: LLM-কে নিজের Data দিয়ে জ্ঞানী করা',
  description: 'Retrieval Augmented Generation — Vector Embeddings, FAISS Vector Database, Semantic Search এবং সম্পূর্ণ RAG Pipeline — LLM hallucination কমানোর সবচেয়ে effective পদ্ধতি।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 15,
  content: `
    <h3>১. LLM-এর সমস্যা: Hallucination ও Knowledge Cutoff</h3>
    <p>LLM-এর দুটি বড় সীমাবদ্ধতা:</p>
    <ul>
      <li><strong>Knowledge cutoff:</strong> Training data-এর পরের ঘটনা LLM জানে না</li>
      <li><strong>Hallucination:</strong> LLM confidently ভুল তথ্য দেয়</li>
      <li><strong>Private data:</strong> কোম্পানির internal documents LLM-এর কাছে নেই</li>
    </ul>
    <p><strong>RAG (Retrieval Augmented Generation)</strong> এই সমস্যাগুলো সমাধান করে। প্রশ্নের সাথে relevant documents retrieve করে LLM-কে context হিসেবে দেওয়া হয়।</p>

    <h3>২. Vector Embeddings</h3>
    <p>Semantic similarity বোঝার জন্য text-কে dense vector-এ রূপান্তর করা হয়। Similar meaning = similar vectors (high cosine similarity)।</p>

    <pre><code class="language-python">from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "I enjoy eating pizza",
    "Artificial intelligence includes machine learning",
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (4, 384)

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ML-related sentences are similar
print(f"ML vs DL: {cosine_sim(embeddings[0], embeddings[1]):.3f}")    # ~0.7
print(f"ML vs AI: {cosine_sim(embeddings[0], embeddings[3]):.3f}")    # ~0.8
print(f"ML vs pizza: {cosine_sim(embeddings[0], embeddings[2]):.3f}") # ~0.1
</code></pre>

    <h3>৩. FAISS Vector Store</h3>
    <p>Millions of vectors-এর মধ্যে nearest neighbors খোঁজার জন্য <strong>FAISS</strong> (Facebook AI Similarity Search) ব্যবহার করা হয়।</p>

    <pre><code class="language-python">import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base
documents = [
    "Bangladesh gained independence in 1971 after the Liberation War.",
    "Dhaka is the capital and largest city of Bangladesh.",
    "The Padma Bridge opened in 2022 connecting southern Bangladesh.",
    "Bangladesh is known for its garment industry which employs millions.",
    "The Sundarbans mangrove forest is shared between Bangladesh and India.",
    "Bengali (Bangla) is the official language of Bangladesh.",
    "Bangladesh won independence from Pakistan on December 16, 1971.",
    "The population of Bangladesh is approximately 170 million people.",
]

# Create embeddings
doc_embeddings = model.encode(documents, show_progress_bar=True)
dimension = doc_embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine with normalized vectors)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings.astype('float32'))
print(f"Index contains {index.ntotal} vectors")

def retrieve(query, k=3):
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb.astype('float32'), k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({'score': score, 'text': documents[idx]})
    return results

results = retrieve("When did Bangladesh become independent?")
for r in results:
    print(f"Score: {r['score']:.3f} | {r['text']}")
</code></pre>

    <h3>৪. Complete RAG Pipeline</h3>

    <pre><code class="language-python">import faiss, numpy as np, openai
from sentence_transformers import SentenceTransformer

client = openai.OpenAI(api_key="your-key")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGSystem:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = embed_model.encode(documents)
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))

    def retrieve(self, query, k=3):
        q_emb = embed_model.encode([query])
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype('float32'), k)
        return [self.documents[i] for i in idxs[0] if scores[0][list(idxs[0]).index(i)] > 0.3]

    def answer(self, question):
        context_docs = self.retrieve(question)
        context = "\\n".join([f"- {d}" for d in context_docs])

        prompt = f"""Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return {
            'answer': resp.choices[0].message.content,
            'sources': context_docs
        }

# Example usage
docs = [
    "Our return policy allows returns within 30 days with receipt.",
    "Shipping takes 3-5 business days for standard delivery.",
    "Express shipping (1-2 days) costs \$15 extra.",
    "We accept Visa, Mastercard, and PayPal.",
    "Customer service is available Mon-Fri 9am-6pm EST.",
]

rag = RAGSystem(docs)
result = rag.answer("How long does shipping take?")
print("Answer:", result['answer'])
print("Sources:", result['sources'])
</code></pre>

    <h3>৫. RAG vs Fine-tuning: কোনটি কখন?</h3>
    <table>
      <thead><tr><th>Aspect</th><th>RAG</th><th>Fine-tuning</th></tr></thead>
      <tbody>
        <tr><td>Data update</td><td>Real-time (add docs)</td><td>Retrain needed</td></tr>
        <tr><td>Cost</td><td>Low setup</td><td>High compute</td></tr>
        <tr><td>Hallucination</td><td>Reduced (grounded)</td><td>Model-dependent</td></tr>
        <tr><td>Style/tone</td><td>Limited</td><td>Highly customizable</td></tr>
        <tr><td>Best for</td><td>Dynamic knowledge base</td><td>Specific behavior/style</td></tr>
      </tbody>
    </table>
  `
};
