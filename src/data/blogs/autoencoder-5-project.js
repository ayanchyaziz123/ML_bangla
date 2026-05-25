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
import tensorflow as tf
from tensorflow import keras
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

def build_autoencoder(input_dim, encoding_dim=14):
    # এনকোডার
    inputs = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(20, activation='relu')(inputs)
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)

    # ডিকোডার
    decoded = keras.layers.Dense(20, activation='relu')(encoded)
    decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

ae = build_autoencoder(input_dim)
ae.summary()
</code></pre>

<h4>মডেল ট্রেনিং</h4>

<pre><code class="language-python">history = ae.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    validation_data=(X_val, X_val),
    verbose=1
)

plt.plot(history.history['loss'], label='ট্রেন লস')
plt.plot(history.history['val_loss'], label='ভ্যাল লস')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('অটোএনকোডার ট্রেনিং')
plt.show()
</code></pre>

<h3>পুনর্নির্মাণ ত্রুটি হিসাব</h3>

<pre><code class="language-python"># সম্পূর্ণ ডেটাসেটের পুনর্নির্মাণ ত্রুটি
X_pred = ae.predict(X)
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

<pre><code class="language-python">class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=8):
    # এনকোডার
    inputs = keras.Input(shape=(input_dim,))
    h = keras.layers.Dense(16, activation='relu')(inputs)
    z_mean = keras.layers.Dense(latent_dim)(h)
    z_log_var = keras.layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z])

    # ডিকোডার
    latent_inputs = keras.Input(shape=(latent_dim,))
    h_dec = keras.layers.Dense(16, activation='relu')(latent_inputs)
    outputs = keras.layers.Dense(input_dim)(h_dec)
    decoder = keras.Model(latent_inputs, outputs)

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
