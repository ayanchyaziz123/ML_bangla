export const lr_3_metrics = {
  title: "R², MSE, RMSE, MAE — লিনিয়ার রিগ্রেশনের সব এরর মেট্রিক্স",
  description: "লিনিয়ার রিগ্রেশন মূল্যায়নের সব মেট্রিক্স — R², Adjusted R², MSE, RMSE, MAE — কী, কেন, এবং কখন কোনটা ব্যবহার করবে।",
  date: "২৩ মে, ২০২৬",
  category: "লিনিয়ার রিগ্রেশন",
  readTime: 8,
  slug: "lr-metrics-r2-mse",
  content: `
    <h3>১. কেন মেট্রিক্স দরকার?</h3>
    <p>মডেল ট্রেন করার পর জানতে হবে সে কতটা ভালো কাজ করছে। রিগ্রেশনের জন্য বিভিন্ন মেট্রিক্স আছে — প্রতিটি ভিন্ন দৃষ্টিকোণ থেকে মডেলের ভুল পরিমাপ করে।</p>
    <p>আমাদের উদাহরণ ডেটা:</p>
    <pre><code>y_true (Y):    [35, 55, 70, 85, 95]
y_pred (Ŷ):    [38, 52, 73, 82, 98]
residuals:     [-3, +3, -3, +3, -3]   ← Residuals</code></pre>

    <h3>২. MAE — Mean Absolute Error</h3>
    <p>প্রতিটি ভুলের নিরঙ্কুশ মান নিয়ে গড় করা হয়।</p>
    <pre><code>MAE = (1/n) × Σ|Yᵢ - Ŷᵢ|

# আমাদের ডেটায়:
MAE = (3 + 3 + 3 + 3 + 3) / 5 = 3.0

# মানে: গড়ে ৩ নম্বর ভুল হচ্ছে
# সুবিধা: বোঝা সহজ, outlier-এ কম প্রভাবিত
# অসুবিধা: বড় ভুলকে extra punishment দেয় না</code></pre>

    <h3>৩. MSE — Mean Squared Error</h3>
    <p>প্রতিটি ভুলকে বর্গ করে গড় নেওয়া হয় — বড় ভুলকে বেশি শাস্তি দেয়।</p>
    <pre><code>MSE = (1/n) × Σ(Yᵢ - Ŷᵢ)²

# আমাদের ডেটায়:
MSE = (9 + 9 + 9 + 9 + 9) / 5 = 9.0

# সুবিধা: গণিত সহজ, গ্রেডিয়েন্ট ডিসেন্টে ব্যবহার হয়
# অসুবিধা: একক (unit) বর্গ হয়ে যায় (নম্বর² → বোঝা কঠিন)
# outlier থাকলে MSE অনেক বেড়ে যায়</code></pre>

    <h3>৪. RMSE — Root Mean Squared Error</h3>
    <p>MSE-র বর্গমূল নেওয়া হয় — একক (unit) আবার আসল ডেটার মতো হয়।</p>
    <pre><code>RMSE = √MSE = √9 = 3.0

# সুবিধা: MAE-এর মতো বোঝা সহজ, কিন্তু বড় ভুলে বেশি sensitive
# সাধারণত RMSE > MAE হয়
# RMSE ≈ MAE মানে outlier কম</code></pre>

    <div style="margin:1.5rem 0;">
      <svg viewBox="0 0 500 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;display:block;margin:0 auto;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb;">
        <text x="250" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#111827">MAE vs MSE vs RMSE তুলনা</text>
        <!-- MAE -->
        <rect x="40"  y="35" width="90" height="70" rx="4" fill="#dbeafe" stroke="#93c5fd"/>
        <text x="85"  y="58" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">MAE</text>
        <text x="85"  y="75" text-anchor="middle" font-size="9" fill="#1e3a8a">গড় নিরঙ্কুশ ভুল</text>
        <text x="85"  y="90" text-anchor="middle" font-size="10" fill="#1e40af" font-weight="600">= 3.0</text>
        <text x="85"  y="100" text-anchor="middle" font-size="8" fill="#6b7280">Outlier-কম</text>
        <!-- MSE -->
        <rect x="200" y="35" width="90" height="70" rx="4" fill="#fef3c7" stroke="#f59e0b"/>
        <text x="245" y="58" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">MSE</text>
        <text x="245" y="75" text-anchor="middle" font-size="9" fill="#92400e">গড় বর্গ ভুল</text>
        <text x="245" y="90" text-anchor="middle" font-size="10" fill="#d97706" font-weight="600">= 9.0</text>
        <text x="245" y="100" text-anchor="middle" font-size="8" fill="#6b7280">Outlier-sensitive</text>
        <!-- RMSE -->
        <rect x="360" y="35" width="90" height="70" rx="4" fill="#d1fae5" stroke="#6ee7b7"/>
        <text x="405" y="58" text-anchor="middle" font-size="11" font-weight="700" fill="#065f46">RMSE</text>
        <text x="405" y="75" text-anchor="middle" font-size="9" fill="#065f46">√MSE</text>
        <text x="405" y="90" text-anchor="middle" font-size="10" fill="#059669" font-weight="600">= 3.0</text>
        <text x="405" y="100" text-anchor="middle" font-size="8" fill="#6b7280">সবচেয়ে জনপ্রিয়</text>
      </svg>
    </div>

    <h3>৫. R² — Coefficient of Determination</h3>
    <p>R² বলে মডেল ডেটার কতটুকু variation ব্যাখ্যা করতে পারছে।</p>
    <pre><code>R² = 1 - (SS_res / SS_tot)

# SS_res = Σ(Yᵢ - Ŷᵢ)²   ← মডেলের ভুল
# SS_tot = Σ(Yᵢ - Ȳ)²    ← মোট variation

# উদাহরণ:
Y  = [35, 55, 70, 85, 95],  Ȳ = 68
Ŷ  = [38, 52, 73, 82, 98]

SS_res = 9+9+9+9+9 = 45
SS_tot = (35-68)²+(55-68)²+(70-68)²+(85-68)²+(95-68)²
       = 1089+169+4+289+729 = 2280

R² = 1 - (45/2280) = 1 - 0.0197 = 0.980

# মানে: মডেল ৯৮% variation ব্যাখ্যা করতে পারছে ✓</code></pre>

    <h3>৬. R² ব্যাখ্যা</h3>
    <table>
      <thead><tr><th>R² মান</th><th>মানে</th><th>মূল্যায়ন</th></tr></thead>
      <tbody>
        <tr><td>1.0</td><td>সম্পূর্ণ নিখুঁত ফিট</td><td>সন্দেহজনক (হয়তো overfitting)</td></tr>
        <tr><td>0.9 – 0.99</td><td>খুব ভালো</td><td>✅ চমৎকার</td></tr>
        <tr><td>0.7 – 0.9</td><td>ভালো</td><td>✅ গ্রহণযোগ্য</td></tr>
        <tr><td>0.5 – 0.7</td><td>মাঝামাঝি</td><td>⚠️ আরও কাজ করা দরকার</td></tr>
        <tr><td>0 – 0.5</td><td>দুর্বল</td><td>❌ মডেল ভালো না</td></tr>
        <tr><td>&lt; 0</td><td>গড়ের চেয়েও খারাপ</td><td>❌ মডেল কাজ করছে না</td></tr>
      </tbody>
    </table>

    <h3>৭. Adjusted R²</h3>
    <p>সাধারণ R² সমস্যা: ফিচার যোগ করলে R² সবসময় বাড়ে (এমনকি অর্থহীন ফিচারেও)। Adjusted R² এটা ঠিক করে।</p>
    <pre><code>Adjusted R² = 1 - [(1-R²)(n-1) / (n-k-1)]

# n = ডেটা পয়েন্ট
# k = ফিচার সংখ্যা

# উদাহরণ: n=100, k=5, R²=0.85
Adj_R² = 1 - [(1-0.85)(99)/(94)] = 1 - [0.1579] = 0.842

# নিয়ম:
# - নতুন ফিচার useful হলে Adjusted R² বাড়বে
# - নতুন ফিচার useless হলে Adjusted R² কমবে বা একই থাকবে</code></pre>

    <h3>৮. Python দিয়ে সব মেট্রিক্স</h3>
    <pre><code>from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [35, 55, 70, 85, 95]
y_pred = [38, 52, 73, 82, 98]

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)

n, k = len(y_true), 1
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(f"MAE:         {mae:.3f}")
print(f"MSE:         {mse:.3f}")
print(f"RMSE:        {rmse:.3f}")
print(f"R²:          {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <table>
      <thead><tr><th>মেট্রিক্স</th><th>সূত্র</th><th>কখন ব্যবহার করবে</th></tr></thead>
      <tbody>
        <tr><td>MAE</td><td>গড় |ভুল|</td><td>outlier থাকলে, বোধগম্য রিপোর্টে</td></tr>
        <tr><td>MSE</td><td>গড় ভুল²</td><td>optimization-এ, gradient descent-এ</td></tr>
        <tr><td>RMSE</td><td>√MSE</td><td>সবচেয়ে জনপ্রিয়, বড় ভুলে sensitive</td></tr>
        <tr><td>R²</td><td>1 - SS_res/SS_tot</td><td>মডেল কতটা ভালো — সাধারণ মূল্যায়নে</td></tr>
        <tr><td>Adjusted R²</td><td>R² + penalty</td><td>একাধিক ফিচার তুলনায়</td></tr>
      </tbody>
    </table>
  `,
};
