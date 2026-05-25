export const autoencoder_3_sparse = {
  slug: 'autoencoder-3-sparse',
  title: 'স্পার্স অটোএনকোডার: কম অ্যাক্টিভেশন, বেশি শেখা',
  description: 'L1 রেগুলারাইজেশন এবং KL divergence দিয়ে স্পার্স অটোএনকোডার তৈরি, ফিচার লার্নিং এবং সাধারণ AE-এর সাথে তুলনা।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 12,
  content: `
    <h3>১. স্পার্সিটি কী এবং কেন দরকার?</h3>
    <p>
      মানুষের মস্তিষ্কে যেকোনো একটি উদ্দীপনার প্রতি মাত্র কিছু নিউরন সক্রিয় হয়, বাকিগুলো নিষ্ক্রিয় থাকে। এই ধারণাটি নিউরোসায়েন্স থেকে নেওয়া এবং মেশিন লার্নিংয়ে প্রয়োগ করা হয়েছে। এটাই হলো <strong>স্পার্সিটি (Sparsity)</strong>।
    </p>
    <p>
      একটি স্তরে যদি বেশিরভাগ নিউরন ০ অথবা ০ এর কাছাকাছি মান পায়, তাহলে সেই স্তরটি <em>স্পার্স</em>। <strong>স্পার্স অটোএনকোডার (Sparse Autoencoder — SAE)</strong> লেটেন্ট স্তরে স্পার্সিটি নিশ্চিত করতে একটি পেনাল্টি যোগ করে।
    </p>
    <p>
      কেন স্পার্সিটি দরকার? কারণ:
      <br/><strong>১)</strong> এটি নেটওয়ার্ককে ওভারকমপ্লিট (overcomplete) লেটেন্ট স্পেস ব্যবহার করতে দেয় — মানে লেটেন্ট ডিমেনশন ইনপুটের চেয়ে বড় হতে পারে।
      <br/><strong>২)</strong> এটি নেটওয়ার্ককে আরও ডিসেন্ট্যাংগেলড (disentangled) ফিচার শিখতে সাহায্য করে।
      <br/><strong>৩)</strong> প্রতিটি ফিচার নিউরন ডেটার একটি নির্দিষ্ট দিক উপস্থাপন করে।
    </p>

    <h3>২. L1 রেগুলারাইজেশন দিয়ে স্পার্সিটি</h3>
    <p>
      সবচেয়ে সরল পদ্ধতি হলো লেটেন্ট অ্যাক্টিভেশনে <strong>L1 রেগুলারাইজেশন</strong> যোগ করা।
    </p>
    <p>
      মোট লস: <strong>L_total = L_reconstruction + λ · ||z||₁</strong>
    </p>
    <p>
      এখানে:
      <br/> — L_reconstruction হলো রিকনস্ট্রাকশন লস (BCE বা MSE)
      <br/> — ||z||₁ = Σ|zᵢ| হলো লেটেন্ট ভেক্টরের L1 নর্ম
      <br/> — λ (lambda) হলো স্পার্সিটির শক্তি নিয়ন্ত্রণকারী হাইপারপ্যারামিটার
    </p>
    <p>
      L1 পেনাল্টি ছোট অ্যাক্টিভেশনগুলোকে একদম ০ এ নামিয়ে আনে, ফলে প্রকৃত স্পার্সিটি তৈরি হয়। (L2 পেনাল্টি ছোট করে কিন্তু ০ করে না।)
    </p>

    <h3>৩. KL Divergence দিয়ে স্পার্সিটি পেনাল্টি</h3>
    <p>
      আরও পরিশীলিত পদ্ধতি হলো <strong>KL Divergence</strong> ব্যবহার করে একটি নির্দিষ্ট গড় অ্যাক্টিভেশন (ρ) নিশ্চিত করা।
    </p>
    <p>
      ধরুন আমরা চাই প্রতিটি নিউরনের গড় অ্যাক্টিভেশন ρ = ০.০৫ (মাত্র ৫%)। প্রতিটি নিউরনের প্রকৃত গড় অ্যাক্টিভেশন হলো ρ̂ⱼ।
    </p>
    <p>
      KL স্পার্সিটি পেনাল্টি: <strong>KL(ρ || ρ̂ⱼ) = ρ·log(ρ/ρ̂ⱼ) + (1-ρ)·log((1-ρ)/(1-ρ̂ⱼ))</strong>
    </p>
    <p>
      মোট লস: <strong>L_total = L_reconstruction + β · Σⱼ KL(ρ || ρ̂ⱼ)</strong>
    </p>
    <p>
      যদি ρ̂ⱼ = ρ হয়, তাহলে KL = ০। যদি ρ̂ⱼ ρ থেকে বিচ্যুত হয়, পেনাল্টি বাড়ে। এটি প্রতিটি নিউরনকে লক্ষ্য অ্যাক্টিভেশনের দিকে ঠেলে দেয়।
    </p>

    <h3>৪. Keras দিয়ে L1 স্পার্স অটোএনকোডার</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

# ডেটা প্রস্তুত
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32').reshape(-1, 784) / 255.0
x_test  = x_test.astype('float32').reshape(-1, 784)  / 255.0

# L1 রেগুলারাইজেশন সহ স্পার্স অটোএনকোডার
# encoding_dim > input_dim হতে পারে (overcomplete)
encoding_dim = 1024  # ইনপুট ৭৮৪ এর চেয়ে বড়!

inp = keras.Input(shape=(784,))
# activity_regularizer সরাসরি অ্যাক্টিভেশনে পেনাল্টি দেয়
encoded = layers.Dense(
    encoding_dim,
    activation='relu',
    activity_regularizer=regularizers.l1(1e-5)  # λ = 0.00001
)(inp)

decoded = layers.Dense(784, activation='sigmoid')(encoded)

sparse_ae_l1 = keras.Model(inp, decoded, name='sparse_ae_l1')
sparse_ae_l1.compile(optimizer='adam', loss='binary_crossentropy')
sparse_ae_l1.summary()</code></pre>

    <pre><code># ট্রেনিং
history_l1 = sparse_ae_l1.fit(
    x_train, x_train,
    epochs=40,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)</code></pre>

    <h3>৫. KL Divergence স্পার্সিটি কাস্টম লেয়ার</h3>
    <pre><code">import tensorflow as tf

class KLSparsityRegularizer(keras.regularizers.Regularizer):
    """KL Divergence ভিত্তিক স্পার্সিটি রেগুলারাইজার"""

    def __init__(self, rho=0.05, beta=3.0):
        self.rho  = rho    # টার্গেট স্পার্সিটি (গড় অ্যাক্টিভেশন)
        self.beta = beta   # পেনাল্টির শক্তি

    def __call__(self, activations):
        # ব্যাচে গড় অ্যাক্টিভেশন
        rho_hat = tf.reduce_mean(activations, axis=0)
        rho_hat = tf.clip_by_value(rho_hat, 1e-7, 1.0 - 1e-7)

        rho = self.rho
        # KL(rho || rho_hat)
        kl = rho * tf.math.log(rho / rho_hat) + \
             (1 - rho) * tf.math.log((1 - rho) / (1 - rho_hat))
        return self.beta * tf.reduce_sum(kl)

    def get_config(self):
        return {'rho': self.rho, 'beta': self.beta}</code></pre>

    <pre><code># KL স্পার্স অটোএনকোডার তৈরি
inp = keras.Input(shape=(784,))
encoded = layers.Dense(
    512,
    activation='sigmoid',  # sigmoid যাতে অ্যাক্টিভেশন ০-১ এর মধ্যে থাকে
    activity_regularizer=KLSparsityRegularizer(rho=0.05, beta=3.0)
)(inp)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

sparse_ae_kl = keras.Model(inp, decoded, name='sparse_ae_kl')
sparse_ae_kl.compile(optimizer='adam', loss='binary_crossentropy')

history_kl = sparse_ae_kl.fit(
    x_train, x_train,
    epochs=40,
    batch_size=256,
    validation_data=(x_test, x_test),
    verbose=1
)</code></pre>

    <h3>৬. স্পার্সিটি পরিমাপ করা</h3>
    <pre><code">def measure_sparsity(model, data, threshold=0.1):
    """নেটওয়ার্কের স্পার্সিটি পরিমাপ"""
    # এনকোডার লেয়ার পর্যন্ত মডেল তৈরি
    encoder = keras.Model(model.input, model.layers[-2].output)
    activations = encoder.predict(data, verbose=0)

    # threshold এর নিচের মান = নিষ্ক্রিয় নিউরন
    inactive = np.mean(activations < threshold)
    mean_activation = np.mean(activations)

    return {
        'sparsity': inactive,
        'mean_activation': mean_activation,
        'active_neurons_per_sample': np.mean(np.sum(activations > threshold, axis=1))
    }

# সাধারণ AE তৈরি (তুলনার জন্য)
inp = keras.Input(shape=(784,))
enc = layers.Dense(512, activation='relu')(inp)
dec = layers.Dense(784, activation='sigmoid')(enc)
standard_ae = keras.Model(inp, dec, name='standard_ae')
standard_ae.compile(optimizer='adam', loss='binary_crossentropy')
standard_ae.fit(x_train, x_train, epochs=30, batch_size=256, verbose=0)

# তুলনা
stats_standard = measure_sparsity(standard_ae, x_test)
stats_l1      = measure_sparsity(sparse_ae_l1, x_test)
stats_kl      = measure_sparsity(sparse_ae_kl, x_test)

print("Standard AE:")
print(f"  Sparsity:           {stats_standard['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_standard['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_standard['active_neurons_per_sample']:.1f}")

print("\\nL1 Sparse AE:")
print(f"  Sparsity:           {stats_l1['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_l1['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_l1['active_neurons_per_sample']:.1f}")

print("\\nKL Sparse AE:")
print(f"  Sparsity:           {stats_kl['sparsity']:.2%}")
print(f"  Mean Activation:    {stats_kl['mean_activation']:.4f}")
print(f"  Active Neurons/Sample: {stats_kl['active_neurons_per_sample']:.1f}")</code></pre>

    <h3>৭. লার্নড ফিচার ভিজুয়ালাইজেশন</h3>
    <pre><code">def visualize_filters(model, layer_name, n_filters=64):
    """এনকোডার ওয়েট ভিজুয়ালাইজ করা"""
    # প্রথম এনকোডার লেয়ারের ওয়েট নিন
    weights = model.get_layer(layer_name).get_weights()[0]
    # weights আকার: (784, n_neurons)

    n = min(n_filters, weights.shape[1])
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < n:
            filter_img = weights[:, i].reshape(28, 28)
            ax.imshow(filter_img, cmap='RdBu_r',
                      vmin=-abs(filter_img).max(),
                      vmax=abs(filter_img).max())
        ax.axis('off')

    plt.suptitle(f'Learned Features — {model.name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'features_{model.name}.png', dpi=150)
    plt.show()

# প্রথম Dense লেয়ারের নাম বের করা
for layer in sparse_ae_l1.layers:
    if 'dense' in layer.name:
        first_dense = layer.name
        break

visualize_filters(sparse_ae_l1, first_dense)</code></pre>

    <h3>৮. Overcomplete স্পার্স অটোএনকোডার</h3>
    <p>
      স্পার্স অটোএনকোডারের সবচেয়ে আকর্ষণীয় বৈশিষ্ট্য: লেটেন্ট ডিমেনশন ইনপুটের চেয়ে <em>বড়</em> হতে পারে। এটি <strong>overcomplete representation</strong> তৈরি করে।
    </p>
    <p>
      যদি encoding_dim = 2000 এবং input_dim = 784 হয়, তাহলে স্পার্সিটি ছাড়া নেটওয়ার্ক trivially ইনপুট কপি করবে। কিন্তু স্পার্সিটি পেনাল্টি নিশ্চিত করে যে যেকোনো একটি স্যাম্পলের জন্য মাত্র কিছু নিউরন সক্রিয় থাকবে।
    </p>
    <p>
      এর সুবিধা: প্রতিটি ফিচার নিউরন ডেটার একটি ছোট, নির্দিষ্ট দিক (আংশিক বৈশিষ্ট্য) শেখে। MNIST-এ একটি নিউরন হয়তো "উপরে বাঁকানো রেখা" শেখে, আরেকটি "গোলাকার আকৃতি" শেখে।
    </p>

    <h3>৯. স্পার্স AE বনাম অন্যান্য পদ্ধতি</h3>
    <p>
      <strong>স্পার্স AE বনাম আন্ডারকমপ্লিট AE:</strong> আন্ডারকমপ্লিট AE কম্প্রেশনের মাধ্যমে স্পার্সিটি নিশ্চিত করে। স্পার্স AE স্পষ্ট পেনাল্টি দিয়ে বড় লেটেন্ট স্পেসেও স্পার্সিটি বজায় রাখে।
    </p>
    <p>
      <strong>স্পার্স AE বনাম PCA:</strong> PCA orthogonal basis vectors শেখে। স্পার্স AE আরও localized এবং interpretable ফিচার শিখতে পারে।
    </p>
    <p>
      <strong>Dictionary Learning-এর সাথে সম্পর্ক:</strong> Overcomplete স্পার্স AE মূলত neural dictionary learning করছে — এটি এমন একটি dictionary (weight matrix) শেখে যেখানে যেকোনো ইনপুট কিছু সংখ্যক basis vector-এর linear combination হিসেবে প্রকাশ পায়।
    </p>

    <h3>১০. Lambda পছন্দ করা</h3>
    <pre><code">lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
sparsity_results = {}

for lam in lambdas:
    inp = keras.Input(shape=(784,))
    enc = layers.Dense(256, activation='relu',
                       activity_regularizer=regularizers.l1(lam))(inp)
    dec = layers.Dense(784, activation='sigmoid')(enc)
    model = keras.Model(inp, dec)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, x_train, epochs=15, batch_size=256,
              validation_data=(x_test, x_test), verbose=0)

    # রিকনস্ট্রাকশন লস
    val_loss = model.evaluate(x_test, x_test, verbose=0)
    # স্পার্সিটি
    enc_model = keras.Model(model.input, model.layers[1].output)
    acts = enc_model.predict(x_test, verbose=0)
    sparsity = np.mean(acts < 0.1)
    sparsity_results[lam] = {'loss': val_loss, 'sparsity': sparsity}
    print(f"lambda={lam:.0e}: val_loss={val_loss:.4f}, sparsity={sparsity:.2%}")

# ট্রেডঅফ প্লট
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
lams = list(sparsity_results.keys())

ax1.semilogx(lams, [sparsity_results[l]['loss'] for l in lams], 'bo-', linewidth=2)
ax1.set_xlabel('Lambda (L1 coefficient)', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Reconstruction Quality vs Lambda', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.semilogx(lams, [sparsity_results[l]['sparsity'] for l in lams], 'rs-', linewidth=2)
ax2.set_xlabel('Lambda (L1 coefficient)', fontsize=12)
ax2.set_ylabel('Sparsity (%)', fontsize=12)
ax2.set_title('Sparsity vs Lambda', fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('lambda_tradeoff.png', dpi=150)
plt.show()</code></pre>

    <h3>সারসংক্ষেপ</h3>
    <p>
      স্পার্স অটোএনকোডার লেটেন্ট স্তরে L1 রেগুলারাইজেশন বা KL Divergence পেনাল্টি যোগ করে নিউরনের অ্যাক্টিভেশন কম রাখে। এটি নেটওয়ার্ককে আরও অর্থপূর্ণ, localized এবং ইন্টারপ্রিটেবল ফিচার শিখতে সাহায্য করে। Overcomplete লেটেন্ট স্পেস ব্যবহার করলেও স্পার্সিটি trivial solution রোধ করে।
    </p>
  `
};
