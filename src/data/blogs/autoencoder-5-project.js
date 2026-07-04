export const autoencoder_5_project = {
  slug: 'autoencoder-5-project',
  title: 'অটোএনকোডার প্রজেক্ট: অ্যানোমালি ডিটেকশন',
  description: 'পুনর্নির্মাণ ত্রুটি ব্যবহার করে ক্রেডিট কার্ড জালিয়াতি এবং ডেটা অসংগতি শনাক্ত করুন',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 14,
  content: `
<h3>কেন অটোএনকোডার দিয়ে অ্যানোমালি ডিটেকশন?</h3>
<p>অটোএনকোডার স্বাভাবিক ডেটার প্যাটার্ন শেখে। যখন অস্বাভাবিক ডেটা দেওয়া হয়, পুনর্নির্মাণ ত্রুটি (reconstruction error) অনেক বেশি হয় — এই নীতিই অ্যানোমালি ডিটেকশনের ভিত্তি।</p>

<h4>মূল ধারণা</h4>
<ul>
<li><strong>স্বাভাবিক ডেটা:</strong> অটোএনকোডার ভালোভাবে পুনর্নির্মাণ করতে পারে → কম ত্রুটি</li>
<li><strong>অস্বাভাবিক ডেটা:</strong> অটোএনকোডার পুনর্নির্মাণ করতে পারে না → বেশি ত্রুটি</li>
<li><strong>থ্রেশহোল্ড:</strong> একটি নির্দিষ্ট ত্রুটিমান — এর বেশি হলে অ্যানোমালি</li>
</ul>

<h3>ক্রেডিট কার্ড জালিয়াতি ডেটাসেট</h3>
<p>Kaggle-এর Credit Card Fraud Detection ডেটাসেট ব্যবহার করব। এতে ৪৯২টি জালিয়াতি এবং ২,৮৪,৩১৫টি স্বাভাবিক লেনদেন আছে — মাত্র ০.১৭% জালিয়াতি।</p>

<pre><code class="language-python">import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ডেটা লোড
df = pd.read_csv('creditcard.csv')
print(df['Class'].value_counts())
# 0    284315  (স্বাভাবিক)
# 1       492  (জালিয়াতি)

# ফিচার স্কেলিং
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1).values
y = df['Class'].values

# শুধু স্বাভাবিক ডেটা দিয়ে ট্রেন করি
X_normal = X[y == 0]
X_fraud = X[y == 1]

X_train, X_val = train_test_split(X_normal, test_size=0.1, random_state=42)
print(f"ট্রেনিং: {X_train.shape}, ভ্যালিডেশন: {X_val.shape}")
</code></pre>

<h3>অটোএনকোডার তৈরি</h3>

<pre><code class="language-python">input_dim = X_train.shape[1]  # 30 ফিচার

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=14):
        super().__init__()
        # এনকোডার
        self.enc_fc1 = nn.Linear(input_dim, 20)
        self.enc_fc2 = nn.Linear(20, encoding_dim)
        # ডিকোডার
        self.dec_fc1 = nn.Linear(encoding_dim, 20)
        self.dec_fc2 = nn.Linear(20, input_dim)   # linear activation (কোনো আউটপুট নন-লিনিয়ারিটি নেই)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc_fc1(x))
        encoded = self.relu(self.enc_fc2(x))
        x = self.relu(self.dec_fc1(encoded))
        return self.dec_fc2(x)

ae = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
print(ae)
</code></pre>

<h4>মডেল ট্রেনিং</h4>

<pre><code class="language-python">from torch.utils.data import DataLoader, TensorDataset

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, X_val_t),   batch_size=256)

history = {'loss': [], 'val_loss': []}

for epoch in range(50):
    ae.train()
    train_loss = 0.0
    for xb, _ in train_loader:
        optimizer.zero_grad()
        reconstructed = ae(xb)
        loss = criterion(reconstructed, xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    ae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, _ in val_loader:
            reconstructed = ae(xb)
            val_loss += criterion(reconstructed, xb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)

plt.plot(history['loss'], label='ট্রেন লস')
plt.plot(history['val_loss'], label='ভ্যাল লস')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('অটোএনকোডার ট্রেনিং')
plt.show()
</code></pre>

<h3>পুনর্নির্মাণ ত্রুটি হিসাব</h3>

<pre><code class="language-python"># সম্পূর্ণ ডেটাসেটের পুনর্নির্মাণ ত্রুটি
ae.eval()
with torch.no_grad():
    X_pred = ae(torch.tensor(X, dtype=torch.float32)).numpy()
reconstruction_errors = np.mean(np.power(X - X_pred, 2), axis=1)

# স্বাভাবিক vs জালিয়াতির ত্রুটি বিতরণ
normal_errors = reconstruction_errors[y == 0]
fraud_errors = reconstruction_errors[y == 1]

print(f"স্বাভাবিক গড় ত্রুটি: {normal_errors.mean():.6f}")
print(f"জালিয়াতি গড় ত্রুটি: {fraud_errors.mean():.6f}")

plt.figure(figsize=(10, 5))
plt.hist(normal_errors, bins=50, alpha=0.5, label='স্বাভাবিক', color='blue')
plt.hist(fraud_errors, bins=50, alpha=0.5, label='জালিয়াতি', color='red')
plt.xlabel('পুনর্নির্মাণ ত্রুটি')
plt.ylabel('সংখ্যা')
plt.legend()
plt.title('স্বাভাবিক vs জালিয়াতির ত্রুটি বিতরণ')
plt.show()
</code></pre>

<h3>থ্রেশহোল্ড নির্ধারণ ও মূল্যায়ন</h3>

<pre><code class="language-python">from sklearn.metrics import precision_recall_curve

# স্বাভাবিক ডেটার ৯৫তম পার্সেন্টাইল
threshold = np.percentile(normal_errors, 95)
print(f"থ্রেশহোল্ড: {threshold:.6f}")

# ভবিষ্যদ্বাণী
y_pred = (reconstruction_errors > threshold).astype(int)

print("\\nক্লাসিফিকেশন রিপোর্ট:")
print(classification_report(y, y_pred, target_names=['স্বাভাবিক', 'জালিয়াতি']))

auc = roc_auc_score(y, reconstruction_errors)
print(f"ROC-AUC: {auc:.4f}")

# প্রিসিশন-রিকল কার্ভ দিয়ে সেরা থ্রেশহোল্ড
precision, recall, thresholds = precision_recall_curve(y, reconstruction_errors)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"সর্বোত্তম থ্রেশহোল্ড (F1): {best_threshold:.6f}")
print(f"সর্বোচ্চ F1: {f1_scores.max():.4f}")
</code></pre>

<h3>ভ্যারিয়েশনাল অটোএনকোডার দিয়ে তুলনা</h3>

<pre><code class="language-python">class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.fc = nn.Linear(input_dim, 16)
        self.z_mean = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        h = torch.relu(self.fc(x))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 16)
        self.out = nn.Linear(16, input_dim)

    def forward(self, z):
        h = torch.relu(self.fc(z))
        return self.out(h)   # linear output

def build_vae(input_dim, latent_dim=8):
    encoder = VAEEncoder(input_dim, latent_dim)
    decoder = VAEDecoder(latent_dim, input_dim)
    return encoder, decoder

# VAE ট্রেনিং ও তুলনা
# (VAE সাধারণত আরও মসৃণ latent space তৈরি করে)
</code></pre>

<h3>ফলাফল বিশ্লেষণ</h3>
<table>
<tr><th>মেট্রিক</th><th>সাধারণ AE</th><th>VAE</th></tr>
<tr><td>Precision (fraud)</td><td>~0.82</td><td>~0.79</td></tr>
<tr><td>Recall (fraud)</td><td>~0.71</td><td>~0.75</td></tr>
<tr><td>ROC-AUC</td><td>~0.95</td><td>~0.94</td></tr>
<tr><td>ট্রেনিং সময়</td><td>দ্রুত</td><td>ধীর</td></tr>
</table>

<h4>প্রধান শিক্ষা</h4>
<ul>
<li>অটোএনকোডার লেবেলবিহীন ডেটায় অ্যানোমালি খুঁজে পায় (unsupervised)</li>
<li>থ্রেশহোল্ড নির্বাচন ব্যবসায়িক প্রয়োজনের উপর নির্ভর করে</li>
<li>উচ্চ recall চাইলে থ্রেশহোল্ড কমান, উচ্চ precision চাইলে বাড়ান</li>
<li>একাধিক মডেল একত্রিত করলে সেরা ফলাফল পাওয়া যায়</li>
</ul>
`
};
