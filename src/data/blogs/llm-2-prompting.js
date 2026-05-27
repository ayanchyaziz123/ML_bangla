export const llm_2_prompting = {
  slug: 'llm-2-prompting',
  title: 'Prompt Engineering: LLM থেকে সেরা ফলাফল পাওয়ার কৌশল',
  description: 'Zero-shot, Few-shot, Chain-of-Thought prompting, System prompts, Role prompting এবং structured output — LLM-কে কার্যকরভাবে ব্যবহার করার সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'এলএলএম ও ফাইন-টিউনিং',
  readTime: 12,
  content: `
    <h3>১. Prompt Engineering কেন গুরুত্বপূর্ণ?</h3>
    <p>একই LLM-এ একই প্রশ্ন ভিন্নভাবে করলে সম্পূর্ণ ভিন্ন উত্তর পাওয়া যায়। <strong>Prompt Engineering</strong> হলো LLM-কে সঠিকভাবে instruct করার শিল্প যাতে desired output পাওয়া যায়।</p>
    <p>Stanford DAWNBench-এ দেখা গেছে: ভালো prompt fine-tuning-এর চেয়েও ভালো performance দিতে পারে।</p>

    <h3>২. Zero-shot, One-shot, Few-shot</h3>
    <p><strong>Zero-shot:</strong> কোনো example ছাড়াই সরাসরি task বর্ণনা করা</p>
    <p><strong>Few-shot:</strong> কয়েকটি input-output example দিয়ে pattern শেখানো</p>

    <pre><code class="language-python">import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# Zero-shot
zero_shot = """Classify the sentiment of this review as Positive, Negative, or Neutral.

Review: "The product arrived on time but quality was disappointing."
Sentiment:"""

# Few-shot
few_shot = """Classify sentiment as Positive, Negative, or Neutral.

Review: "Excellent product, exceeded expectations!" → Positive
Review: "Waste of money, broke after one day." → Negative
Review: "It's okay, nothing special." → Neutral
Review: "The product arrived on time but quality was disappointing." →"""

for prompt_type, prompt in [("Zero-shot", zero_shot), ("Few-shot", few_shot)]:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"{prompt_type}: {response.content[0].text.strip()}")
</code></pre>

    <h3>৩. Chain-of-Thought (CoT) Prompting</h3>
    <p>Complex reasoning problems-এ LLM-কে step-by-step চিন্তা করতে বলা হয়। "Let's think step by step" বা explicit reasoning steps যোগ করলে accuracy উল্লেখযোগ্যভাবে বাড়ে।</p>

    <pre><code class="language-python">import openai

client = openai.OpenAI(api_key="your-key")

# Without CoT
direct = """A store has 15 red balls and 20 blue balls.
If 8 red and 5 blue balls are sold, what fraction of remaining balls are red?"""

# With CoT
cot = """A store has 15 red balls and 20 blue balls.
If 8 red and 5 blue balls are sold, what fraction of remaining balls are red?

Let's think step by step:"""

# Zero-shot CoT: just add "Let's think step by step"
# Few-shot CoT: provide examples with reasoning steps

for style, prompt in [("Direct", direct), ("Chain-of-Thought", cot)]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200, temperature=0
    )
    print(f"\\n{style}:")
    print(resp.choices[0].message.content)
</code></pre>

    <h3>৪. System Prompts ও Role Prompting</h3>
    <p><strong>System prompt</strong> দিয়ে LLM-এর behavior, persona এবং constraints define করা হয়। এটি conversation-এর আগেই LLM-কে "যে কে" তা বলে দেয়।</p>

    <pre><code class="language-python">import openai
client = openai.OpenAI(api_key="your-key")

def create_expert(role, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are {role}.
Respond with technical depth appropriate to your expertise.
Keep answers concise (under 150 words).
Always acknowledge limitations and uncertainties."""
            },
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

question = "How should I invest my savings?"

roles = [
    "a conservative financial advisor focused on capital preservation",
    "an aggressive growth investor who believes in high-risk high-reward",
    "a behavioral economist who studies human financial decision-making"
]

for role in roles:
    print(f"\\nRole: {role[:50]}...")
    print(create_expert(role, question))
    print("-" * 60)
</code></pre>

    <h3>৫. Structured Output</h3>
    <p>LLM-কে JSON বা specific format-এ output দিতে বলা হয় যাতে programmatically parse করা যায়।</p>

    <pre><code class="language-python">import openai, json
client = openai.OpenAI(api_key="your-key")

def extract_structured(text):
    prompt = f"""Extract information from the following job posting and return ONLY valid JSON.

JSON Schema:
{{
  "title": "job title",
  "company": "company name",
  "location": "city, country",
  "salary_range": "salary if mentioned, else null",
  "required_skills": ["skill1", "skill2"],
  "experience_years": number or null,
  "remote": true/false
}}

Job Posting:
{text}

Return only the JSON, no explanation:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

job_post = """
Senior ML Engineer at TechCorp, San Francisco.
5+ years experience required. Skills: Python, PyTorch, MLOps, Kubernetes.
Salary: \$150k-\$200k. Hybrid work (3 days in office).
"""

result = extract_structured(job_post)
print(json.dumps(result, indent=2))
</code></pre>

    <h3>৬. Prompt Engineering Best Practices</h3>
    <ul>
      <li><strong>Specific এবং clear:</strong> "একটি email লেখো" নয়, "একজন software engineer-এর job application rejection email লেখো, professional tone-এ, 100 words-এর মধ্যে"</li>
      <li><strong>Constraints দাও:</strong> format, length, tone, audience specify করো</li>
      <li><strong>Context দাও:</strong> Background information যোগ করো</li>
      <li><strong>Output format specify করো:</strong> "JSON-এ return করো", "numbered list হিসেবে"</li>
      <li><strong>Iterative refinement:</strong> প্রথম prompt থেকে শুরু করে refine করতে থাকো</li>
      <li><strong>Temperature adjust করো:</strong> Factual tasks = low, Creative tasks = high</li>
    </ul>
  `
};
