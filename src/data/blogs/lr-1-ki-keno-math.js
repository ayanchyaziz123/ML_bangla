export const lr_1_ki_keno_math = {
  title: "লিনিয়ার রিগ্রেশন কী এবং এর পেছনের গণিত",
  description: "লিনিয়ার রিগ্রেশনের মূল ধারণা, সরল উদাহরণ, এবং OLS পদ্ধতির গণিত — একদম সহজ ভাষায় বাংলায়।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 10,
  slug: "lr-ki-keno-math",
  content: `
    <h3>১. লিনিয়ার রিগ্রেশন কী?</h3>
    <p>লিনিয়ার রিগ্রেশন হলো একটি supervised ML অ্যালগরিদম যেটি দুটি বা তার বেশি ভেরিয়েবলের মধ্যে <strong>সরল রৈখিক সম্পর্ক</strong> খোঁজে। এটি একটি সংখ্যা ভবিষ্যদ্বাণী করতে ব্যবহার হয়।</p>
    <p>উদাহরণ: বাড়ির আয়তন জানা থাকলে দাম কত হবে? পড়াশোনার ঘণ্টা জানা থাকলে পরীক্ষার নম্বর কত হবে?</p>

    <h3>২. সহজ উদাহরণ: ছোট ডেটাসেট</h3>
    <p>ধরো, ৫ জন শিক্ষার্থীর পড়ার ঘণ্টা এবং পরীক্ষার নম্বর:</p>
    <table>
      <thead><tr><th>শিক্ষার্থী</th><th>পড়ার ঘণ্টা (X)</th><th>নম্বর (Y)</th></tr></thead>
      <tbody>
        <tr><td>রাহেলা</td><td>2</td><td>35</td></tr>
        <tr><td>করিম</td><td>4</td><td>55</td></tr>
        <tr><td>সুমাইয়া</td><td>6</td><td>70</td></tr>
        <tr><td>তানভীর</td><td>8</td><td>85</td></tr>
        <tr><td>নাফিসা</td><td>10</td><td>95</td></tr>
      </tbody>
    </table>
    <p>লক্ষ্য করো — পড়ার ঘণ্টা বাড়লে নম্বর বাড়ছে। এই সম্পর্কটাই লিনিয়ার রিগ্রেশন ধরতে পারে।</p>

    <h3>৩. লিনিয়ার রিগ্রেশনের সমীকরণ</h3>
    <pre><code># সরল লিনিয়ার রিগ্রেশন (একটি ফিচার):
Y = mX + b

# যেখানে:
# Y  = ভবিষ্যদ্বাণী করা মান (target)
# X  = ইনপুট ফিচার
# m  = slope (ঢাল) — X বাড়লে Y কতটা বাড়ে
# b  = intercept — X=0 হলে Y-এর মান

# মাল্টিপল লিনিয়ার রিগ্রেশন (একাধিক ফিচার):
Y = m₁X₁ + m₂X₂ + ... + mₙXₙ + b</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 420 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:420px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="210" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">Best-fit রেখা (Regression Line)</text>
        <line x1="40" y1="155" x2="390" y2="155" stroke="#e5e7eb" stroke-width="1.5"/>
        <line x1="40" y1="20"  x2="40"  y2="155" stroke="#e5e7eb" stroke-width="1.5"/>
        <text x="395" y="158" font-size="9" fill="#9ca3af">X</text>
        <text x="35"  y="15"  font-size="9" fill="#9ca3af">Y</text>
        <!-- Data points -->
        <circle cx="90"  cy="135" r="5" fill="#1e40af"/>
        <circle cx="150" cy="110" r="5" fill="#1e40af"/>
        <circle cx="210" cy="85"  r="5" fill="#1e40af"/>
        <circle cx="280" cy="55"  r="5" fill="#1e40af"/>
        <circle cx="350" cy="35"  r="5" fill="#1e40af"/>
        <!-- Regression line -->
        <line x1="60" y1="150" x2="380" y2="28" stroke="#dc2626" stroke-width="2"/>
        <text x="345" y="22" font-size="9" fill="#dc2626" font-weight="600">Y = mX + b</text>
        <!-- Residual lines -->
        <line x1="90"  cy="135" x2="90"  y2="143" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,2"/>
        <line x1="150" cy="110" x2="150" y2="117" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,2"/>
        <line x1="210" cy="85"  x2="210" y2="88"  stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,2"/>
        <text x="210" y="170" text-anchor="middle" font-size="9" fill="#6b7280">হলুদ রেখা = Residual (প্রকৃত - ভবিষ্যদ্বাণী)</text>
      </svg>
    </div>

    <h3>৪. Cost Function — কীভাবে সেরা রেখা খোঁজা হয়?</h3>
    <p>মডেল চায় এমন একটি m এবং b খুঁজে পেতে যাতে ভুল সবচেয়ে কম হয়। এই ভুল পরিমাপ করা হয় <strong>Mean Squared Error (MSE)</strong> দিয়ে:</p>
    <pre><code>MSE = (1/n) × Σ(Yᵢ - Ŷᵢ)²

# যেখানে:
# Yᵢ  = প্রকৃত মান
# Ŷᵢ  = মডেলের ভবিষ্যদ্বাণী
# n   = ডেটা পয়েন্টের সংখ্যা

# আমাদের ডেটায় হিসাব:
# যদি m=9, b=17 হয়:
X = [2,  4,  6,  8,  10]
Y = [35, 55, 70, 85,  95]   # প্রকৃত
Ŷ = [35, 53, 71, 89, 107]   # ভবিষ্যদ্বাণী (9×X + 17)

# Residuals: [0, -2, +1, +4, -12]
# MSE = (0 + 4 + 1 + 16 + 144) / 5 = 33</code></pre>

    <h3>৫. OLS — সেরা m এবং b বের করার গণিত</h3>
    <p>OLS (Ordinary Least Squares) পদ্ধতিতে গণিত দিয়ে সরাসরি সেরা m এবং b বের করা যায়:</p>
    <pre><code># OLS সূত্র:
m = Σ((Xᵢ - X̄)(Yᵢ - Ȳ)) / Σ(Xᵢ - X̄)²
b = Ȳ - m × X̄

# আমাদের ডেটায়:
X = [2, 4, 6, 8, 10],  X̄ = 6
Y = [35, 55, 70, 85, 95],  Ȳ = 68

# (Xᵢ - X̄): [-4, -2, 0, 2, 4]
# (Yᵢ - Ȳ): [-33, -13, 2, 17, 27]

# Σ((Xᵢ-X̄)(Yᵢ-Ȳ)) = 132 + 26 + 0 + 34 + 108 = 300
# Σ(Xᵢ-X̄)² = 16 + 4 + 0 + 4 + 16 = 40

m = 300 / 40 = 7.5
b = 68 - 7.5 × 6 = 23

# সমীকরণ: Y = 7.5X + 23
# ৭ ঘণ্টা পড়লে: Y = 7.5×7 + 23 = 75.5</code></pre>

    <h3>৬. Gradient Descent — আরেকটি পদ্ধতি</h3>
    <p>বড় ডেটায় OLS ধীর হয়ে যায়। তখন <strong>Gradient Descent</strong> ব্যবহার করা হয় — এটি ধীরে ধীরে m এবং b আপডেট করে সেরা মান খোঁজে।</p>
    <pre><code># প্রতি ধাপে আপডেট:
m = m - α × (∂MSE/∂m)
b = b - α × (∂MSE/∂b)

# α = learning rate (কত দ্রুত শিখবে, সাধারণত 0.01–0.1)
# বড় α → দ্রুত কিন্তু অস্থির
# ছোট α → ধীর কিন্তু নির্ভরযোগ্য</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>বিষয়</th><th>মূল কথা</th></tr></thead>
      <tbody>
        <tr><td>লিনিয়ার রিগ্রেশন</td><td>সংখ্যা ভবিষ্যদ্বাণীর জন্য সরল রৈখিক মডেল</td></tr>
        <tr><td>সমীকরণ</td><td>Y = mX + b</td></tr>
        <tr><td>Cost Function</td><td>MSE — ভুলের বর্গের গড়</td></tr>
        <tr><td>OLS</td><td>গণিত দিয়ে সরাসরি সেরা m, b বের করে</td></tr>
        <tr><td>Gradient Descent</td><td>ধীরে ধীরে ভুল কমিয়ে সেরা মান খোঁজে</td></tr>
      </tbody>
    </table>
  `,
};
