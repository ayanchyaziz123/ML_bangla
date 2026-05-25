export const autoencoder_2_denoising = {
  slug: 'autoencoder-2-denoising',
  title: 'ডিনয়েজিং অটোএনকোডার: নয়েজ থেকে মুক্তি',
  description: 'ডিনয়েজিং অটোএনকোডার কীভাবে নয়েজযুক্ত ইনপুট থেকে পরিষ্কার ডেটা শেখে, Gaussian noise, DAE ট্রেনিং কৌশল এবং MNIST-এ সম্পূর্ণ Keras ইমপ্লিমেন্টেশন।',
  date: 'মে ২০২৫',
  category: 'অটোএনকোডার',
  readTime: 13,
  content: `
    <h3>১. ডিনয়েজিং অটোএনকোডার কী?</h3>
    <p>
      সাধারণ অটোএনকোডার শেখে কীভাবে ডেটাকে কম্প্রেস করে পুনরুদ্ধার করতে হয়। কিন্তু <strong>ডিনয়েজিং অটোএনকোডার (Denoising Autoencoder — DAE)</strong> একটি কঠিন কাজ করে: এটি <em>নয়েজযুক্ত ইনপুট</em> নেয় এবং <em>পরিষ্কার আউটপুট</em> তৈরি করতে শেখে।
    </p>
    <p>
      ধারণাটি সহজ কিন্তু শক্তিশালী। যদি আমরা একটি ছবিতে কৃত্রিম নয়েজ যোগ করে নেটওয়ার্ককে বলি "এই নয়েজযুক্ত ছবি থেকে আসল ছবি বের করো", তাহলে নেটওয়ার্ককে শুধু পিক্সেল কপি করলে চলবে না — সত্যিকারের উপযোগী বৈশিষ্ট্য (features) শিখতে হবে।
    </p>
    <p>
      DAE-এর প্রস্তাব করেছিলেন <strong>Pascal Vincent এবং দল (২০০৮)</strong>। তাদের মূল অন্তর্দৃষ্টি ছিল: নয়েজ যোগ করে ট্রেনিং নেটওয়ার্ককে আরও শক্তিশালী এবং সাধারণীকরণযোগ্য (generalizable) ফিচার শিখতে সাহায্য করে।
    </p>

    <h3>২. DAE-এর ট্রেনিং উদ্দেশ্য</h3>
    <p>
      সাধারণ অটোএনকোডারের লস: <strong>L(x, g(f(x)))</strong> — মূল ইনপুট এবং রিকনস্ট্রাকশনের পার্থক্য।
    </p>
    <p>
      DAE-এর লস: <strong>L(x, g(f(x̃)))</strong> — এখানে x̃ হলো নয়েজযুক্ত ইনপুট, কিন্তু টার্গেট হলো মূল পরিষ্কার x।
    </p>
    <p>
      এই পরিবর্তনটি ছোট কিন্তু গভীর প্রভাব ফেলে। নেটওয়ার্ককে শিখতে হয় কীভাবে দুর্নীতিগ্রস্ত (corrupted) ডেটা থেকে আসল তথ্য পুনরুদ্ধার করতে হয়।
    </p>

    <h3>৩. নয়েজের ধরন</h3>
    <p>
      <strong>Gaussian Noise:</strong> প্রতিটি পিক্সেলে একটি সাধারণ বিতরণ থেকে র্যান্ডম মান যোগ করা হয়। সবচেয়ে সাধারণ নয়েজ।
      <br/><code>x̃ = x + ε, যেখানে ε ~ N(0, σ²)</code>
    </p>
    <p>
      <strong>Salt-and-Pepper Noise:</strong> কিছু পিক্সেল এলোমেলোভাবে ০ (কালো) বা ১ (সাদা) করা হয়।
    </p>
    <p>
      <strong>Masking Noise (Dropout Noise):</strong> কিছু পিক্সেলকে এলোমেলোভাবে ০ করে দেওয়া হয়। এটি BERT-এর masked language model-এর ধারণার অনুরূপ।
    </p>

    <h3>৪. DAE কেন ভালো ফিচার শেখে?</h3>
    <p>
      সাধারণ অটোএনকোডার যদি লেটেন্ট স্পেস যথেষ্ট বড় হয়, তাহলে ইনপুট হুবহু কপি করে একটি trivial solution খুঁজে পেতে পারে। DAE এই সমস্যা সমাধান করে।
    </p>
    <p>
      <strong>কারণ:</strong> নয়েজযুক্ত ইনপুট থেকে আসল ডেটা পুনরুদ্ধার করতে হলে নেটওয়ার্ককে ডেটার সত্যিকারের কাঠামো (underlying structure) বুঝতে হবে। সে শিখবে কোন পিক্সেলগুলো পরস্পর সম্পর্কিত, ডেটায় কোন প্যাটার্ন আছে।
    </p>
    <p>
      <strong>রোবাস্টনেস:</strong> নয়েজে ট্রেনিং করলে নেটওয়ার্ক বাস্তব জগতের অপূর্ণ ডেটার সাথে মোকাবিলা করতে পারে।
    </p>

    <h3>৫. Keras দিয়ে DAE ইমপ্লিমেন্টেশন</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ডেটা লোড
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# Gaussian নয়েজ যোগ করা
noise_factor = 0.5

x_train_noisy = x_train + noise_factor * np.random.randn(*x_train.shape)
x_test_noisy  = x_test  + noise_factor * np.random.randn(*x_test.shape)

# পিক্সেল মান ০ থেকে ১ এর মধ্যে রাখা
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy  = np.clip(x_test_noisy,  0.0, 1.0)

print(f"Clean train shape: {x_train.shape}")
print(f"Noisy train shape: {x_train_noisy.shape}")</code></pre>

    <pre><code># DAE মডেল তৈরি
def build_dae(input_dim=784, encoding_dim=128):
    # এনকোডার
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.Dense(256, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)

    # ডিকোডার
    x = layers.Dense(256, activation='relu')(encoded)
    x = layers.Dense(512, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)

    model = keras.Model(inp, decoded, name='denoising_autoencoder')
    return model

dae = build_dae(encoding_dim=128)
dae.compile(optimizer='adam', loss='binary_crossentropy')
dae.summary()</code></pre>

    <pre><code># ট্রেনিং: নয়েজযুক্ত ইনপুট -> পরিষ্কার আউটপুট
history = dae.fit(
    x_train_noisy, x_train,       # নয়েজযুক্ত ইনপুট, পরিষ্কার টার্গেট
    epochs=40,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)</code></pre>

    <pre><code># ডিনয়েজিং রেজাল্ট দেখা
x_test_denoised = dae.predict(x_test_noisy)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # মূল পরিষ্কার ছবি
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Clean', fontsize=8)
    plt.axis('off')

    # নয়েজযুক্ত ছবি
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title('Noisy', fontsize=8)
    plt.axis('off')

    # ডিনয়েজড ছবি
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_test_denoised[i].reshape(28, 28), cmap='gray')
    plt.title('Denoised', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('denoising_results.png', dpi=150)
plt.show()</code></pre>

    <h3>৬. বিভিন্ন নয়েজ লেভেলের তুলনা</h3>
    <pre><code># বিভিন্ন noise_factor পরীক্ষা
noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
results = {}

for nf in noise_levels:
    x_noisy = np.clip(x_test + nf * np.random.randn(*x_test.shape), 0, 1)
    x_denoised = dae.predict(x_noisy, verbose=0)

    # MSE হিসাব করা (মূল ছবি vs ডিনয়েজড ছবি)
    mse = np.mean((x_test - x_denoised) ** 2)
    results[nf] = mse
    print(f"Noise Factor {nf:.1f} -> MSE: {mse:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), 'bo-', linewidth=2)
plt.xlabel('Noise Factor (σ)', fontsize=13)
plt.ylabel('Reconstruction MSE', fontsize=13)
plt.title('DAE Performance vs Noise Level', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('noise_vs_mse.png', dpi=150)
plt.show()</code></pre>

    <h3>৭. Convolutional DAE (CDAE)</h3>
    <p>
      ছবির জন্য Convolutional স্তর ব্যবহার করলে আরও ভালো ফলাফল পাওয়া যায়, কারণ CNN স্থানিক (spatial) তথ্য সংরক্ষণ করে।
    </p>
    <pre><code>def build_convolutional_dae():
    inp = keras.Input(shape=(28, 28, 1))

    # এনকোডার
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same')(x)

    # ডিকোডার
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    model = keras.Model(inp, decoded, name='conv_dae')
    return model

# ডেটা রিশেপ (CNN-এর জন্য ৪D)
x_train_4d = x_train.reshape(-1, 28, 28, 1)
x_test_4d  = x_test.reshape(-1, 28, 28, 1)
x_train_noisy_4d = x_train_noisy.reshape(-1, 28, 28, 1)
x_test_noisy_4d  = x_test_noisy.reshape(-1, 28, 28, 1)

cdae = build_convolutional_dae()
cdae.compile(optimizer='adam', loss='binary_crossentropy')

history_conv = cdae.fit(
    x_train_noisy_4d, x_train_4d,
    epochs=30,
    batch_size=128,
    validation_data=(x_test_noisy_4d, x_test_4d),
    verbose=1
)</code></pre>

    <h3>৮. DAE এবং সাধারণ AE-এর তুলনা</h3>
    <pre><code># সাধারণ AE তৈরি (একই আর্কিটেকচার কিন্তু পরিষ্কার ইনপুটে ট্রেন)
def build_standard_ae(input_dim=784, encoding_dim=128):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.Dense(256, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(encoded)
    x = layers.Dense(512, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)
    return keras.Model(inp, decoded, name='standard_ae')

sae = build_standard_ae()
sae.compile(optimizer='adam', loss='binary_crossentropy')
sae.fit(x_train, x_train, epochs=30, batch_size=256,
        validation_data=(x_test, x_test), verbose=0)

# নয়েজযুক্ত টেস্ট ডেটায় উভয় মডেলের MSE
sae_denoised = sae.predict(x_test_noisy, verbose=0)
dae_denoised = dae.predict(x_test_noisy, verbose=0)

mse_sae = np.mean((x_test - sae_denoised) ** 2)
mse_dae = np.mean((x_test - dae_denoised) ** 2)

print(f"Standard AE MSE on noisy data: {mse_sae:.4f}")
print(f"Denoising AE MSE on noisy data: {mse_dae:.4f}")
print(f"DAE improvement: {(mse_sae - mse_dae) / mse_sae * 100:.1f}%")</code></pre>

    <h3>৯. বাস্তব জীবনে DAE-এর প্রয়োগ</h3>
    <p>
      <strong>মেডিকেল ইমেজিং:</strong> MRI বা CT স্ক্যানে নয়েজ কমাতে DAE ব্যবহার হয়। কম রেডিয়েশনে তোলা ছবিকে উচ্চমানের ছবিতে রূপান্তরিত করা যায়।
    </p>
    <p>
      <strong>অডিও ডিনয়েজিং:</strong> ব্যাকগ্রাউন্ড নয়েজ থেকে পরিষ্কার কণ্ঠস্বর আলাদা করা। স্পিচ রিকগনিশন সিস্টেমে ব্যবহৃত হয়।
    </p>
    <p>
      <strong>ওল্ড ফটো রেস্টোরেশন:</strong> পুরানো, ক্ষতিগ্রস্ত ছবিকে পুনরুদ্ধার করতে DAE প্রয়োগ করা হচ্ছে।
    </p>
    <p>
      <strong>টেক্সট ক্লিনিং:</strong> স্ক্যান করা ডকুমেন্টে OCR-এর আগে DAE দিয়ে নয়েজ কমানো হয়।
    </p>
    <p>
      <strong>প্রি-ট্রেনিং:</strong> DAE দিয়ে প্রি-ট্রেন করা এনকোডার পরবর্তী কাজে (classification, etc.) fine-tune করা যায় এবং এটি ভালো generalization দেয়।
    </p>

    <h3>সারসংক্ষেপ</h3>
    <p>
      ডিনয়েজিং অটোএনকোডার সাধারণ অটোএনকোডারের একটি শক্তিশালী সংস্করণ। নয়েজযুক্ত ইনপুট থেকে পরিষ্কার আউটপুট শেখার মাধ্যমে এটি আরও অর্থপূর্ণ ফিচার শেখে এবং বাস্তব জগতের অপূর্ণ ডেটায় ভালো পারফর্ম করে। Gaussian noise, masking noise বা অন্য যেকোনো corruption strategy ব্যবহার করে DAE ট্রেন করা যায়।
    </p>
  `
};
