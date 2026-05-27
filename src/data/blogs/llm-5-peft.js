export const llm_5_peft = {
  slug: 'llm-5-peft',
  title: 'LoRA ও PEFT: কম খরচে LLM Fine-tuning',
  description: 'Parameter-Efficient Fine-Tuning, LoRA-র গণিত, QLoRA দিয়ে 4-bit quantization, এবং PEFT library দিয়ে consumer GPU-তে LLM fine-tuning।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 15,
  content: `
    <h3>১. Full Fine-tuning-এর সমস্যা</h3>
    <p>LLaMA-3 7B model fine-tune করতে গেলে full fine-tuning-এ:</p>
    <ul>
      <li>~28 GB GPU VRAM লাগে (float32-এ)</li>
      <li>Gradient storage-এ আরও ~56 GB</li>
      <li>প্রতিটি training run-এ সম্পূর্ণ model copy save করতে হয়</li>
    </ul>
    <p><strong>PEFT (Parameter-Efficient Fine-Tuning)</strong> পদ্ধতিগুলো মূল model frozen রেখে শুধু ছোট trainable parameters যোগ করে।</p>

    <h3>২. LoRA: Low-Rank Adaptation</h3>
    <p>LoRA-র মূল idea: Neural network-এর weight matrix W (d×d) কে full rank-এ update না করে low-rank approximation ব্যবহার করা।</p>
    <p><strong>W' = W + ΔW = W + BA</strong></p>
    <p>যেখানে B (d×r) এবং A (r×d), r &lt;&lt; d।</p>
    <p>উদাহরণ: W = (4096 × 4096) → 16M parameters<br/>
    LoRA rank=8: A = (8×4096) + B = (4096×8) → মাত্র 65K parameters (~0.4%)</p>
    <p>LoRA advantages:</p>
    <ul>
      <li>VRAM ~3x কম লাগে</li>
      <li>Training ~2-3x দ্রুত</li>
      <li>LoRA adapters আলাদাভাবে save করা যায় (কয়েক MB)</li>
      <li>Multiple adapters swap করা যায়</li>
    </ul>

    <pre><code class="language-python">from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                     # Rank
    lora_alpha=32,            # Scaling factor (alpha/r = 2 is common)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply to attention
    bias="none",
)

model = get_peft_model(model, lora_config)

# Compare trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"Total: {total:,}")
# Trainable: ~2.4M (0.15%) of 1B+ parameters
</code></pre>

    <h3>৩. QLoRA: 4-bit Quantization + LoRA</h3>
    <p><strong>QLoRA</strong> = base model 4-bit quantize করে (NF4 format) + LoRA adapters train করা। এতে 7B model single consumer GPU (24 GB VRAM)-এ fine-tune করা যায়।</p>

    <pre><code class="language-python">from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:500]")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=SFTConfig(
        output_dir="./qlora-output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False, bf16=True,
        logging_steps=25,
        max_seq_length=512,
    ),
)
trainer.train()

# Save adapter only (few MB, not full model)
trainer.model.save_pretrained("./qlora-adapter")
print("QLoRA training complete! Adapter saved.")
</code></pre>

    <h3>৪. LoRA Adapter Merge ও Inference</h3>

    <pre><code class="language-python">from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")

# Option 1: Keep adapter (memory efficient, can swap adapters)
# Option 2: Merge into base model (faster inference, no PEFT dependency)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")

# Inference
def generate(prompt, model, tokenizer, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

response = generate("Explain gradient descent in simple terms:", model, tokenizer)
print(response)
</code></pre>

    <h3>৫. PEFT Methods তুলনা</h3>
    <table>
      <thead><tr><th>Method</th><th>Approach</th><th>Memory</th><th>Performance</th></tr></thead>
      <tbody>
        <tr><td>Full FT</td><td>All params</td><td>Very High</td><td>Best</td></tr>
        <tr><td>LoRA</td><td>Low-rank matrices</td><td>Low</td><td>Near Full FT</td></tr>
        <tr><td>QLoRA</td><td>4-bit + LoRA</td><td>Very Low</td><td>Near LoRA</td></tr>
        <tr><td>Prefix Tuning</td><td>Soft prompts</td><td>Very Low</td><td>Good</td></tr>
        <tr><td>Adapter</td><td>Small bottleneck layers</td><td>Low</td><td>Good</td></tr>
      </tbody>
    </table>
  `
};
