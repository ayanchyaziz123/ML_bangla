export const nlp_5_sentiment = {
  slug: 'nlp-5-sentiment',
  title: 'সেন্টিমেন্ট অ্যানালাইসিস প্রজেক্ট',
  description: 'IMDB ডেটাসেট দিয়ে binary sentiment classification — Naive Bayes, Logistic Regression ও LSTM মডেল তুলনা, feature engineering এবং সম্পূর্ণ Python পাইপলাইন।',
  date: 'মে ২০২৫',
  category: 'এনএলপি ফান্ডামেন্টালস',
  readTime: 15,
  content: `
    <h3>১. সেন্টিমেন্ট অ্যানালাইসিস কী?</h3>
    <p>
      <strong>সেন্টিমেন্ট অ্যানালাইসিস</strong> (বা Opinion Mining) হলো একটি NLP task যেখানে টেক্সটের আবেগ বা মনোভাব নির্ণয় করা হয় — সাধারণত positive, negative বা neutral। এটি product review, social media monitoring, customer feedback analysis-এ ব্যাপকভাবে ব্যবহৃত হয়।
    </p>
    <p>
      এই প্রজেক্টে আমরা IMDB Movie Review dataset ব্যবহার করব — ৫০,০০০ চলচ্চিত্র সমালোচনা যেগুলো positive বা negative হিসেবে labeled।
    </p>

    <h3>২. ডেটাসেট লোড ও এক্সপ্লোরেশন</h3>
    <pre><code">import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

# IMDB dataset (kaggle বা sklearn দিয়ে)
# Option 1: datasets library দিয়ে
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']

# DataFrame এ রূপান্তর
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print("Training set size:", len(train_df))
print("Test set size:", len(test_df))
print("\nLabel distribution:")
print(train_df['label'].value_counts())
print("\nSample reviews:")
print(train_df.iloc[0]['text'][:300])

# Review length analysis
train_df['length'] = train_df['text'].apply(lambda x: len(x.split()))
print(f"\nAverage review length: {train_df['length'].mean():.0f} words")
print(f"Max review length: {train_df['length'].max()} words")
print(f"Min review length: {train_df['length'].min()} words")</code></pre>

    <h3>৩. টেক্সট প্রি-প্রসেসিং</h3>
    <pre><code">import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'stopwords', 'wordnet'])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# sentiment-এ এগুলো রাখা উচিত:
sentiment_words = {'not', 'no', 'never', 'nor', 'cannot', "can't", "won't",
                   "didn't", "doesn't", "isn't", "wasn't", "aren't", "weren't"}
stop_words -= sentiment_words

def preprocess_text(text):
    # HTML tags সরানো
    text = re.sub(r'<[^>]+>', '', text)
    # URL সরানো
    text = re.sub(r'http\S+|www\S+', '', text)
    # লোয়ারকেস
    text = text.lower()
    # শুধু অক্ষর ও স্পেস রাখা (numbers ও punctuation সরানো)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # টোকেনাইজেশন
    tokens = word_tokenize(text)
    # স্টপওয়ার্ড সরানো ও লেমাটাইজেশন
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)

# Apply preprocessing
print("Preprocessing করা হচ্ছে...")
train_df['processed'] = train_df['text'].apply(preprocess_text)
test_df['processed'] = test_df['text'].apply(preprocess_text)

print("আগে:", train_df.iloc[0]['text'][:200])
print("\nপরে:", train_df.iloc[0]['processed'][:200])</code></pre>

    <h3>৪. Feature Engineering: TF-IDF</h3>
    <pre><code">from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X_train = train_df['processed']
y_train = train_df['label']
X_test = test_df['processed']
y_test = test_df['label']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=50000,    # শীর্ষ ৫০,০০০ শব্দ
    ngram_range=(1, 2),    # unigram ও bigram
    min_df=5,              # কমপক্ষে ৫ ডকুমেন্টে থাকতে হবে
    max_df=0.95,           # ৯৫%-এর বেশি ডকুমেন্টে থাকলে বাদ
    sublinear_tf=True      # log(TF+1) ব্যবহার করা
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Test matrix shape: {X_test_tfidf.shape}")</code></pre>

    <h3>৫. মডেল ১: Naive Bayes</h3>
    <p>
      <strong>Naive Bayes</strong> classification-এর জন্য সবচেয়ে সরল এবং দ্রুত মডেল। এটি Bayes theorem ব্যবহার করে এবং শব্দগুলো একে অপর থেকে স্বাধীন (naive assumption) ধরে নেয়। Text classification-এ চমৎকার baseline দেয়।
    </p>
    <pre><code">from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Multinomial Naive Bayes
print("Naive Bayes training...")
start = time.time()
nb_model = MultinomialNB(alpha=0.1)  # Laplace smoothing
nb_model.fit(X_train_tfidf, y_train)
nb_time = time.time() - start

# Evaluation
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

print(f"Training time: {nb_time:.2f}s")
print(f"Accuracy: {nb_acc:.4f} ({nb_acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, nb_pred, target_names=['Negative', 'Positive']))</code></pre>

    <h3>৬. মডেল ২: Logistic Regression</h3>
    <p>
      <strong>Logistic Regression</strong> text classification-এ একটি শক্তিশালী এবং interpretable মডেল। এটি প্রতিটি feature (শব্দ)-এর coefficient শেখে যা positive বা negative classification-এ কতটা contribute করে।
    </p>
    <pre><code">from sklearn.linear_model import LogisticRegression

print("Logistic Regression training...")
start = time.time()
lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    n_jobs=-1
)
lr_model.fit(X_train_tfidf, y_train)
lr_time = time.time() - start

lr_pred = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"Training time: {lr_time:.2f}s")
print(f"Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['Negative', 'Positive']))

# সবচেয়ে গুরুত্বপূর্ণ শব্দ দেখা
feature_names = tfidf.get_feature_names_out()
coefs = lr_model.coef_[0]

top_positive = sorted(zip(coefs, feature_names), reverse=True)[:10]
top_negative = sorted(zip(coefs, feature_names))[:10]

print("\nPositive sentiment শব্দ:")
for coef, word in top_positive:
    print(f"  {word}: {coef:.4f}")

print("\nNegative sentiment শব্দ:")
for coef, word in top_negative:
    print(f"  {word}: {coef:.4f}")</code></pre>

    <h3>৭. মডেল ৩: LSTM Neural Network</h3>
    <p>
      <strong>LSTM (Long Short-Term Memory)</strong> একটি recurrent neural network যা sequential data ভালো handle করে। এটি শব্দের ক্রম মনে রাখতে পারে — "not good" এবং "good" আলাদাভাবে বুঝতে পারে।
    </p>
    <pre><code">import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                      Dropout, Bidirectional)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
VOCAB_SIZE = 20000
MAX_LEN = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

# Tokenizer তৈরি
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Sequences তৈরি
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Padded train shape: {X_train_pad.shape}")

# LSTM মডেল
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Training
print("\nLSTM training...")
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
lstm_loss, lstm_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")</code></pre>

    <h3>৮. মডেল তুলনা</h3>
    <pre><code">import matplotlib.pyplot as plt

models = ['Naive Bayes', 'Logistic Regression', 'LSTM (Bidirectional)']
accuracies = [nb_acc, lr_acc, lstm_acc]
times = [nb_time, lr_time, history.history.get('val_accuracy', [0])[-1]]

# Accuracy comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy bar plot
bars = axes[0].bar(models, [a*100 for a in accuracies],
                   color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison on IMDB')
axes[0].set_ylim([80, 96])
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold')

# LSTM training history
axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#3498db')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('LSTM Training History')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_comparison.png', dpi=150)
plt.show()

print("\nFinal Results:")
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 37)
for m, a in zip(models, accuracies):
    print(f"{m:<25} {a*100:>9.2f}%")</code></pre>

    <h3>৯. Evaluation Metrics</h3>
    <p>
      শুধু accuracy দিয়ে মডেল মূল্যায়ন যথেষ্ট নয়। বিশেষত imbalanced dataset-এ।
    </p>
    <p>
      <strong>Precision:</strong> যখন মডেল positive বলে, কতটা সত্যিই positive? TP / (TP + FP)<br/>
      <strong>Recall:</strong> সকল সত্যিকার positive-এর মধ্যে মডেল কতটা ধরতে পেরেছে? TP / (TP + FN)<br/>
      <strong>F1 Score:</strong> Precision ও Recall-এর harmonic mean = 2 × (P × R) / (P + R)<br/>
      <strong>ROC-AUC:</strong> Receiver Operating Characteristic curve-এর area under curve
    </p>
    <pre><code">from sklearn.metrics import roc_auc_score, roc_curve

# ROC-AUC score
lr_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
roc_auc = roc_auc_score(y_test, lr_proba)
print(f"Logistic Regression ROC-AUC: {roc_auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2ecc71', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()</code></pre>

    <h3>১০. নতুন রিভিউ Predict করা</h3>
    <pre><code">def predict_sentiment(text, model_type='lr'):
    """নতুন রিভিউ-এর sentiment predict করুন"""
    processed = preprocess_text(text)

    if model_type == 'nb':
        vec = tfidf.transform([processed])
        pred = nb_model.predict(vec)[0]
        prob = nb_model.predict_proba(vec)[0]
    elif model_type == 'lr':
        vec = tfidf.transform([processed])
        pred = lr_model.predict(vec)[0]
        prob = lr_model.predict_proba(vec)[0]
    else:  # lstm
        seq = tokenizer.texts_to_sequences([processed])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prob_val = model.predict(pad)[0][0]
        pred = 1 if prob_val > 0.5 else 0
        prob = [1 - prob_val, prob_val]

    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob)
    return sentiment, confidence

# উদাহরণ
reviews = [
    "This movie was absolutely brilliant! The acting was superb.",
    "Terrible film. Waste of time. The plot made no sense.",
    "It was okay, not great but not terrible either.",
    "One of the best movies I have ever seen in my life!",
]

print("Sentiment Predictions:")
print("-" * 60)
for review in reviews:
    sentiment, conf = predict_sentiment(review, 'lr')
    print(f"Review: {review[:50]}...")
    print(f"Sentiment: {sentiment} (confidence: {conf:.2%})\n")</code></pre>

    <h3>১১. সারসংক্ষেপ</h3>
    <p>
      এই প্রজেক্টে আমরা তিনটি ভিন্ন মডেল দিয়ে IMDB sentiment analysis করলাম। Naive Bayes দ্রুততার জন্য ভালো baseline, Logistic Regression ভালো accuracy ও interpretability দেয়, আর LSTM sequential context ধরতে পারে। বাস্তব প্রজেক্টে BERT-based model (transformer) সবচেয়ে ভালো ফলাফল দেয় — সেটি পরবর্তী সিরিজে দেখব। পরের পোস্টে আমরা NER (Named Entity Recognition) শিখব।
    </p>
  `
};
