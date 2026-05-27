export const llm_6_project = {
  slug: 'llm-6-project',
  title: 'LLM Project: PDF থেকে AI Chatbot তৈরি',
  description: 'সম্পূর্ণ RAG Chatbot তৈরি — PDF ingestion, chunking, FAISS vector store, conversation history, hallucination detection এবং evaluation।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 18,
  content: `
    <h3>Project Overview</h3>
    <p>এই project-এ একটি PDF document থেকে questions answer করার chatbot তৈরি করব। Real-world RAG pipeline-এর সমস্ত components থাকবে।</p>
    <pre><code>PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer</code></pre>

    <h3>১. Document Ingestion ও Chunking</h3>
    <pre><code class="language-python">from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\\n[Page {page_num+1}]\\n{page_text}"
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Clean chunks
    chunks = [re.sub(r'\\s+', ' ', c).strip() for c in chunks if len(c) > 50]
    return chunks

# Demo with sample text
sample_text = """
Machine learning is a method of data analysis that automates analytical model building.
It is based on the idea that systems can learn from data, identify patterns and make decisions
with minimal human intervention.

Deep learning is part of a broader family of machine learning methods based on artificial
neural networks. Learning can be supervised, semi-supervised or unsupervised.
""" * 10

chunks = chunk_text(sample_text)
print(f"Total chunks: {len(chunks)}")
print(f"Sample chunk: {chunks[0][:100]}...")
</code></pre>

    <h3>২. Vector Store তৈরি</h3>
    <pre><code class="language-python">import faiss, numpy as np, pickle
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []

    def build(self, chunks, metadata=None):
        self.chunks = chunks
        self.metadata = metadata or [{}] * len(chunks)
        embeddings = self.model.encode(chunks, show_progress_bar=True, batch_size=32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        print(f"Built index with {self.index.ntotal} vectors")

    def search(self, query, k=5, min_score=0.3):
        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype('float32'), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if score >= min_score:
                results.append({
                    'text': self.chunks[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        return results

    def save(self, path):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'metadata': self.metadata}, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.metadata = data['metadata']
</code></pre>

    <h3>৩. Conversational RAG Chatbot</h3>
    <pre><code class="language-python">import openai
from collections import deque

client = openai.OpenAI(api_key="your-key")

class RAGChatbot:
    def __init__(self, vector_store, model="gpt-4o-mini", max_history=5):
        self.vs = vector_store
        self.model = model
        self.history = deque(maxlen=max_history * 2)  # user+assistant pairs

    def chat(self, user_query, verbose=False):
        # Retrieve relevant context
        results = self.vs.search(user_query, k=4)
        context = "\\n\\n".join([f"[Score: {r['score']:.2f}] {r['text']}" for r in results])

        # Build messages with history
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions based on provided context.
Rules:
- Answer ONLY from the provided context
- If unsure, say "I don't have enough information in the documents"
- Be concise and accurate
- Cite relevant parts when possible"""
            }
        ]
        # Add conversation history
        messages.extend(list(self.history))
        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"Context from documents:\\n{context}\\n\\nQuestion: {user_query}"
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content

        # Update history (without context to save tokens)
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        if verbose:
            print(f"Retrieved {len(results)} chunks (scores: {[r['score']:.2f] for r in results})")
        return answer

# Demo
vs = VectorStore()
vs.build(chunks)
chatbot = RAGChatbot(vs)

questions = [
    "What is machine learning?",
    "How does deep learning relate to it?",
    "Can you summarize what you just explained?",  # Tests conversation memory
]
for q in questions:
    print(f"Q: {q}")
    print(f"A: {chatbot.chat(q)}\\n")
</code></pre>

    <h3>৪. Evaluation</h3>
    <pre><code class="language-python">from rouge_score import rouge_scorer
import numpy as np

def evaluate_rag(chatbot, qa_pairs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = []
    for qa in qa_pairs:
        pred = chatbot.chat(qa['question'])
        scores = scorer.score(qa['answer'], pred)
        results.append({
            'question': qa['question'],
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'answer_len': len(pred.split())
        })
    avg_r1 = np.mean([r['rouge1'] for r in results])
    avg_rL = np.mean([r['rougeL'] for r in results])
    print(f"Avg ROUGE-1: {avg_r1:.3f} | Avg ROUGE-L: {avg_rL:.3f}")
    return results

test_pairs = [
    {"question": "What is ML?", "answer": "Machine learning automates analytical model building using data."},
    {"question": "What is deep learning?", "answer": "Deep learning uses neural networks and is part of ML."},
]
evaluate_rag(chatbot, test_pairs)
</code></pre>
  `
};
