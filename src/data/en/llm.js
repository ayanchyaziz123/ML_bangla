export const llmEn = [
  {
    slug: 'llm-1-intro',
    title: 'LLMs: How Large Language Models Work',
    description: 'Tokenization, next-token prediction, context windows, temperature, and scaling laws — the foundations of GPT, Claude, and Gemini',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>What is a Language Model?</h3>
<p>A <strong>Large Language Model</strong> predicts the probability of the next token given all preceding tokens. At inference time, sampling from this distribution generates coherent text.</p>
<pre><code class="language-python">from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch, torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("The capital of Bangladesh is", return_tensors='pt')
with torch.no_grad():
    logits = model(input_ids).logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

top5 = torch.topk(probs[0], 5)
for prob, idx in zip(top5.values, top5.indices):
    print(f"'{tokenizer.decode([idx])}': {prob.item():.4f}")
</code></pre>

<h3>Tokenization (BPE)</h3>
<p>Text is split into subword tokens. GPT-4 vocabulary ≈ 100,000 tokens. "unbelievable" → ["un", "believe", "able"].</p>

<h3>Context Window & Sampling</h3>
<ul>
<li><strong>Context window:</strong> max tokens the model can attend to (GPT-4: 128K)</li>
<li><strong>Temperature = 0:</strong> deterministic (argmax); higher = more creative</li>
<li><strong>Top-p (nucleus sampling):</strong> sample from tokens covering cumulative probability p</li>
</ul>
<pre><code class="language-python">from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
result = generator("The future of AI", max_new_tokens=50, temperature=0.7, do_sample=True)
print(result[0]['generated_text'])
</code></pre>

<h3>Scaling Laws</h3>
<p>Performance ∝ log(Parameters × Data × Compute). Chinchilla law: optimal training uses ~20 tokens per parameter. GPT-3 (175B params) should train on ~3.5T tokens.</p>
`
  },
  {
    slug: 'llm-2-prompting',
    title: 'Prompt Engineering: Getting the Best from LLMs',
    description: 'Zero-shot, few-shot, chain-of-thought prompting, system prompts, role prompting, and structured JSON output',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>Zero-shot vs Few-shot</h3>
<pre><code class="language-python">import anthropic
client = anthropic.Anthropic(api_key="your-key")

few_shot = """Classify sentiment as Positive, Negative, or Neutral.

Review: "Excellent product!" → Positive
Review: "Waste of money." → Negative
Review: "Okay, nothing special." → Neutral
Review: "Arrived on time but quality was disappointing." →"""

resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=20,
    messages=[{"role": "user", "content": few_shot}])
print(resp.content[0].text)
</code></pre>

<h3>Chain-of-Thought</h3>
<p>Adding "Let's think step by step" improves accuracy on reasoning tasks by 40-80% (Wei et al. 2022).</p>

<h3>Structured Output</h3>
<pre><code class="language-python">import openai, json
client = openai.OpenAI(api_key="your-key")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: John Smith, 35, software engineer in NYC. Return JSON with name, age, role, city."}],
    temperature=0, response_format={"type": "json_object"}
)
print(json.loads(resp.choices[0].message.content))
</code></pre>

<h3>Best Practices</h3>
<ul>
<li>Be specific about format, length, tone, and audience</li>
<li>Use system prompts to define persona and constraints</li>
<li>Temperature 0 for factual tasks, 0.7+ for creative</li>
<li>Iterate and refine — treat prompts as code</li>
</ul>
`
  },
  {
    slug: 'llm-3-rag',
    title: 'RAG: Retrieval Augmented Generation',
    description: 'Vector embeddings, FAISS semantic search, complete RAG pipeline — grounding LLMs in your own documents',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>Why RAG?</h3>
<p>LLMs have a knowledge cutoff and hallucinate. RAG retrieves relevant documents at query time and injects them as context, grounding the answer in real information.</p>
<pre><code class="language-python">import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import openai

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = openai.OpenAI(api_key="your-key")

class RAG:
    def __init__(self, docs):
        self.docs = docs
        emb = embed_model.encode(docs)
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype('float32'))

    def ask(self, question, k=3):
        q = embed_model.encode([question])
        faiss.normalize_L2(q)
        _, idxs = self.index.search(q.astype('float32'), k)
        context = "\n".join(self.docs[i] for i in idxs[0])
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role": "user", "content":
                f"Answer using only this context:\n{context}\n\nQ: {question}"}])
        return resp.choices[0].message.content

rag = RAG(["Bangladesh gained independence in 1971.",
           "Dhaka is the capital of Bangladesh."])
print(rag.ask("When did Bangladesh become independent?"))
</code></pre>

<h3>RAG vs Fine-tuning</h3>
<table>
<tr><th>Aspect</th><th>RAG</th><th>Fine-tuning</th></tr>
<tr><td>Knowledge update</td><td>Real-time</td><td>Retrain needed</td></tr>
<tr><td>Hallucination</td><td>Reduced</td><td>Model-dependent</td></tr>
<tr><td>Cost</td><td>Low</td><td>High</td></tr>
<tr><td>Best for</td><td>Dynamic knowledge</td><td>Style/behavior</td></tr>
</table>
`
  },
  {
    slug: 'llm-4-finetune',
    title: 'Fine-tuning LLMs: Teaching Domain Knowledge',
    description: 'Instruction tuning datasets, HuggingFace Trainer pipeline, RLHF and DPO fundamentals',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>When to Fine-tune</h3>
<p>Fine-tune when: (1) consistent output format required, (2) domain-specific knowledge not in base model, (3) specific tone/style, (4) latency requirements (smaller fine-tuned model beats prompting a large model).</p>

<h3>Dataset Format (Alpaca/ChatML)</h3>
<pre><code class="language-python">import json

training_examples = [
    {"instruction": "Summarize this medical report",
     "input": "Patient presents with...",
     "output": "Summary: ..."},
]

def to_chatml(ex):
    return {"messages": [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": ex['instruction'] + "\n" + ex.get('input','')},
        {"role": "assistant", "content": ex['output']}
    ]}

with open('train.jsonl', 'w') as f:
    for ex in training_examples:
        f.write(json.dumps(to_chatml(ex)) + '\n')
</code></pre>

<h3>Training with TRL/SFTTrainer</h3>
<pre><code class="language-python">from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
dataset = load_dataset("json", data_files="train.jsonl", split="train")

trainer = SFTTrainer(model=model, train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"]),
    args=SFTConfig(output_dir="./output", num_train_epochs=3, per_device_train_batch_size=4))
trainer.train()
</code></pre>

<h3>RLHF vs DPO</h3>
<p><strong>RLHF:</strong> SFT → Reward Model → PPO optimization (3 stages, complex).<br/>
<strong>DPO:</strong> Direct Preference Optimization — learns from (chosen, rejected) pairs directly without a reward model. Simpler, often better.</p>
`
  },
  {
    slug: 'llm-5-peft',
    title: 'LoRA & QLoRA: Fine-tuning on Consumer Hardware',
    description: 'Low-rank adaptation math, PEFT library, QLoRA 4-bit quantization — fine-tune 7B models on a single GPU',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>LoRA Math</h3>
<p>Instead of updating W (d×d), learn ΔW = BA where B is (d×r) and A is (r×d), with r &lt;&lt; d. For a 4096×4096 matrix at rank 8: 33M → 65K trainable parameters (0.2%).</p>
<pre><code class="language-python">from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16)
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,359,296 || all params: 1,237,825,536 || trainable%: 0.1906
</code></pre>

<h3>QLoRA (4-bit + LoRA)</h3>
<pre><code class="language-python">from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", quantization_config=bnb_config, device_map="auto")
# 7B model: ~28GB FP32 → ~4GB with QLoRA
</code></pre>

<h3>Merge and Deploy</h3>
<pre><code class="language-python">from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base, "./lora-adapter")
merged = model.merge_and_unload()  # Single model, no PEFT dependency
merged.save_pretrained("./merged-model")
</code></pre>
`
  },
  {
    slug: 'llm-6-project',
    title: 'LLM Project: PDF Chatbot with RAG',
    description: 'End-to-end RAG chatbot — PDF ingestion, chunking, FAISS vector store, conversation memory, and ROUGE evaluation',
    category: 'LLMs & Fine-tuning',
    content: `
<h3>Architecture</h3>
<pre><code>PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer
                                                                    ↑ conversation history</code></pre>

<h3>Complete Pipeline</h3>
<pre><code class="language-python">from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, openai
from collections import deque

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = openai.OpenAI(api_key="your-key")

def load_and_chunk(pdf_path, chunk_size=500, overlap=50):
    text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf_path).pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [c for c in splitter.split_text(text) if len(c) > 50]

class PDFChatbot:
    def __init__(self, chunks):
        emb = embed_model.encode(chunks)
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype('float32'))
        self.chunks = chunks
        self.history = deque(maxlen=10)

    def chat(self, query):
        q = embed_model.encode([query]); faiss.normalize_L2(q)
        _, idxs = self.index.search(q.astype('float32'), 4)
        context = "\n".join(f"- {self.chunks[i]}" for i in idxs[0])
        messages = [{"role":"system","content":"Answer from context only. Say 'I don't know' if not found."}]
        messages.extend(list(self.history))
        messages.append({"role":"user","content":f"Context:\n{context}\n\nQ: {query}"})
        answer = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0).choices[0].message.content
        self.history.extend([{"role":"user","content":query},{"role":"assistant","content":answer}])
        return answer

chunks = load_and_chunk("document.pdf")
bot = PDFChatbot(chunks)
print(bot.chat("What is the main topic?"))
print(bot.chat("Can you elaborate on that?"))  # Uses conversation history
</code></pre>

<h3>Evaluation</h3>
<pre><code class="language-python">from rouge_score import rouge_scorer
import numpy as np

def evaluate(bot, qa_pairs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
    scores = [scorer.score(qa['answer'], bot.chat(qa['question'])) for qa in qa_pairs]
    print(f"ROUGE-1: {np.mean([s['rouge1'].fmeasure for s in scores]):.3f}")
    print(f"ROUGE-L: {np.mean([s['rougeL'].fmeasure for s in scores]):.3f}")
</code></pre>
`
  },
];
