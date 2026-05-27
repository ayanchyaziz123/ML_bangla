export const llm_1_intro = {
  slug: 'llm-1-intro',
  title: 'LLM: বড় ভাষা মডেলের পরিচয়',
  description: 'Language Model কী, GPT কীভাবে কাজ করে, Tokenization, Next-Token Prediction, Context Window এবং Temperature — আধুনিক AI-এর মূল ভিত্তি বাংলায়।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 14,
  content: `
    <h3>১. Language Model কী?</h3>
    <p>একটি <strong>Language Model (LM)</strong> হলো এমন একটি system যা text-এর probability মডেল করে — অর্থাৎ, একটি sequence of words কতটা "likely" তা নির্ধারণ করে।</p>
    <p><strong>Large Language Model (LLM)</strong> হলো বিশাল পরিমাণ text data-তে train করা language model — GPT-4, Claude, Gemini, LLaMA এই পরিবারের অন্তর্গত।</p>
    <p>LLM-এর মূল কাজ: <strong>Next Token Prediction</strong> — আগের token গুলো দেখে পরবর্তী token-এর probability বের করা।</p>
    <p>উদাহরণ: "বাংলাদেশের রাজধানীর নাম হলো ___" — এখানে LLM "ঢাকা" token-কে সবচেয়ে বেশি probability দেবে।</p>

    <h3>২. Tokenization: Text থেকে Numbers</h3>
    <p>Neural network text সরাসরি বুঝতে পারে না — text-কে numbers-এ রূপান্তরিত করতে হয়। এই প্রক্রিয়াকে <strong>Tokenization</strong> বলে।</p>
    <p>আধুনিক LLM গুলো <strong>Byte Pair Encoding (BPE)</strong> ব্যবহার করে। BPE subword units তৈরি করে:</p>
    <ul>
      <li>"running" → ["run", "##ning"] বা ["running"]</li>
      <li>"unbelievable" → ["un", "believe", "##able"]</li>
    </ul>
    <p>প্রতিটি token একটি integer ID-তে map হয়। GPT-4 এর vocabulary size ~100,000।</p>

    <pre><code class="language-python">from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Machine learning is fascinating"
tokens = tokenizer.encode(text)
print(f"Tokens (IDs): {tokens}")
print(f"Token count: {len(tokens)}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# See actual token strings
token_strings = [tokenizer.decode([t]) for t in tokens]
print(f"Token strings: {token_strings}")
# ['Machine', ' learning', ' is', ' fascinating']
</code></pre>

    <h3>৩. Transformer Architecture: LLM-এর মূল কাঠামো</h3>
    <p>সমস্ত আধুনিক LLM Transformer architecture-এর উপর নির্মিত। মূল components:</p>
    <ul>
      <li><strong>Embedding Layer:</strong> Token ID → Dense vector (e.g., 768 dimension)</li>
      <li><strong>Positional Encoding:</strong> Token-এর position সম্পর্কে তথ্য যোগ করে</li>
      <li><strong>Multi-Head Self-Attention:</strong> tokens একে অপরের সাথে interact করে</li>
      <li><strong>Feed-Forward Network:</strong> প্রতিটি position-এ independently apply হয়</li>
      <li><strong>Layer Normalization:</strong> training stabilize করে</li>
    </ul>
    <p>GPT-3 (175B parameters) = 96 Transformer layers, 96 attention heads, 12288 hidden dimension।</p>

    <h3>৪. Context Window</h3>
    <p><strong>Context window</strong> হলো LLM একসাথে কতটুকু text "দেখতে" পারে তার সীমা।</p>
    <ul>
      <li>GPT-3.5: 4,096 tokens (~3,000 words)</li>
      <li>GPT-4: 128,000 tokens (~96,000 words)</li>
      <li>Claude 3: 200,000 tokens (~150,000 words)</li>
    </ul>
    <p>Context window-এর বাইরের তথ্য LLM "ভুলে যায়"। এই সীমা অতিক্রম করতে RAG (Retrieval Augmented Generation) ব্যবহার করা হয়।</p>

    <h3>৫. Temperature ও Sampling</h3>
    <p>LLM সবসময় সবচেয়ে probable next token বেছে নেয় না। Sampling parameters দিয়ে output-এর randomness নিয়ন্ত্রণ করা হয়:</p>
    <ul>
      <li><strong>Temperature = 0:</strong> সবসময় সবচেয়ে probable token (deterministic, repetitive)</li>
      <li><strong>Temperature = 1:</strong> Model-এর original distribution অনুযায়ী sample</li>
      <li><strong>Temperature &gt; 1:</strong> আরও random/creative output</li>
      <li><strong>Top-p (nucleus sampling):</strong> Cumulative probability p পর্যন্ত tokens থেকে sample</li>
    </ul>

    <pre><code class="language-python">from transformers import pipeline
import torch

# Text generation pipeline
generator = pipeline(
    'text-generation',
    model='gpt2',
    device=0 if torch.cuda.is_available() else -1
)

prompt = "The future of artificial intelligence"

# Low temperature: deterministic, focused
result_low = generator(
    prompt, max_new_tokens=50,
    temperature=0.3, do_sample=True,
    pad_token_id=50256
)

# High temperature: creative, diverse
result_high = generator(
    prompt, max_new_tokens=50,
    temperature=1.2, do_sample=True,
    top_p=0.9, pad_token_id=50256
)

print("Low temp:", result_low[0]['generated_text'])
print("High temp:", result_high[0]['generated_text'])

# Token probabilities
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

model = GPT2LMHeadModel.from_pretrained('gpt2')
tok = GPT2Tokenizer.from_pretrained('gpt2')

input_ids = tok.encode("The capital of Bangladesh is", return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

top5 = torch.topk(probs[0], 5)
for prob, idx in zip(top5.values, top5.indices):
    print(f"  '{tok.decode([idx])}': {prob.item():.4f}")
</code></pre>

    <h3>৬. LLM Scale এবং Emergent Abilities</h3>
    <p>LLM-এর একটি চমকপ্রদ বৈশিষ্ট্য হলো <strong>emergent abilities</strong> — নির্দিষ্ট parameter count অতিক্রম করলে হঠাৎ নতুন capabilities appear করে:</p>
    <table>
      <thead><tr><th>Model</th><th>Parameters</th><th>Notable Capability</th></tr></thead>
      <tbody>
        <tr><td>GPT-2</td><td>1.5B</td><td>Coherent paragraph generation</td></tr>
        <tr><td>GPT-3</td><td>175B</td><td>Few-shot learning, code generation</td></tr>
        <tr><td>PaLM</td><td>540B</td><td>Chain-of-thought reasoning</td></tr>
        <tr><td>GPT-4</td><td>~1.8T (est.)</td><td>Multi-modal, advanced reasoning</td></tr>
      </tbody>
    </table>
    <p>Scaling Law: Performance ∝ log(Parameters × Data × Compute) — Chinchilla (DeepMind) paper এটি quantify করেছে।</p>
  `
};
