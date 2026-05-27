export const llm_4_finetune = {
  slug: 'llm-4-finetune',
  title: 'LLM Fine-tuning: নিজের কাজের জন্য LLM তৈরি',
  description: 'কেন Fine-tune করবেন, Instruction Tuning, Dataset তৈরি, HuggingFace Trainer দিয়ে সম্পূর্ণ Fine-tuning Pipeline এবং RLHF-এর মূল ধারণা।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 16,
  content: `
    <h3>১. কেন Fine-tune করবেন?</h3>
    <p>Base LLM (যেমন LLaMA, Mistral) অনেক কিছু জানে কিন্তু:</p>
    <ul>
      <li>নির্দিষ্ট domain-এর specialized knowledge নেই (medical, legal, Bangla)</li>
      <li>নির্দিষ্ট tone বা format follow করে না</li>
      <li>Company-specific workflows জানে না</li>
    </ul>
    <p><strong>Fine-tuning</strong> = Base model থেকে শুরু করে নিজের specific data-তে আরও train করা। Full training-এর চেয়ে অনেক কম compute লাগে।</p>
    <table>
      <thead><tr><th>Method</th><th>Cost</th><th>Performance</th><th>Use Case</th></tr></thead>
      <tbody>
        <tr><td>Prompt Engineering</td><td>Free</td><td>Good</td><td>Quick experiments</td></tr>
        <tr><td>RAG</td><td>Low</td><td>Good for knowledge</td><td>Dynamic data</td></tr>
        <tr><td>Fine-tuning (LoRA)</td><td>Medium</td><td>Better style/behavior</td><td>Specific tasks</td></tr>
        <tr><td>Full Fine-tuning</td><td>High</td><td>Best</td><td>Domain specialization</td></tr>
      </tbody>
    </table>

    <h3>২. Instruction Tuning Dataset তৈরি</h3>
    <p>Fine-tuning-এর জন্য <strong>instruction-response pairs</strong> দরকার। প্রতিটি example: instruction (কী করতে হবে) + input (context, optional) + output (desired response)।</p>

    <pre><code class="language-python">import json

# Alpaca-style format
training_data = [
    {
        "instruction": "বাংলায় একটি কবিতার summary লেখো",
        "input": "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি...",
        "output": "এই কবিতায় কবি বাংলাদেশের প্রকৃতির সৌন্দর্য ও মাতৃভূমির প্রতি গভীর ভালোবাসা প্রকাশ করেছেন।"
    },
    {
        "instruction": "Python code-এর bug fix করো",
        "input": "def factorial(n):\\n    if n == 0: return 1\\n    return n * factorial(n)",
        "output": "Bug: base case missing for n=1. Fix:\\ndef factorial(n):\\n    if n <= 1: return 1\\n    return n * factorial(n-1)"
    },
]

# Convert to ChatML format (for instruction tuning)
def to_chatml(example):
    system = "You are a helpful assistant."
    user = example['instruction']
    if example.get('input'):
        user += f"\\n\\nInput:\\n{example['input']}"
    assistant = example['output']
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

chatml_data = [to_chatml(ex) for ex in training_data]
with open('train.jsonl', 'w') as f:
    for item in chatml_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\\n')
print(f"Saved {len(chatml_data)} training examples")
</code></pre>

    <h3>৩. HuggingFace Trainer দিয়ে Fine-tuning</h3>

    <pre><code class="language-python">from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import load_dataset
import torch

model_name = "microsoft/phi-2"  # Small model for demo (2.7B params)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def format_prompt(example):
    text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return {"text": text}

dataset = load_dataset("json", data_files="train.jsonl", split="train")
dataset = dataset.map(format_prompt)

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./finetuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

trainer.train()
trainer.save_model("./finetuned-model")
print("Fine-tuning complete!")
</code></pre>

    <h3>৪. RLHF এর মূল ধারণা</h3>
    <p><strong>Reinforcement Learning from Human Feedback (RLHF)</strong> হলো ChatGPT-কে helpful ও harmless করার পদ্ধতি। তিনটি ধাপ:</p>
    <ol>
      <li><strong>Supervised Fine-tuning (SFT):</strong> Human-written demonstrations দিয়ে base model fine-tune করা</li>
      <li><strong>Reward Model Training:</strong> Human preference data দিয়ে একটি reward model train করা (কোন response better?)</li>
      <li><strong>PPO Optimization:</strong> Reward model-এর signal দিয়ে policy gradient-এ LLM optimize করা</li>
    </ol>
    <p>সহজ বিকল্প: <strong>DPO (Direct Preference Optimization)</strong> — reward model ছাড়াই preference data থেকে directly fine-tune করা যায়।</p>
  `
};
