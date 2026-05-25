export const mlops_1_versioning = {
  slug: 'mlops-1-versioning',
  title: 'MLOps: ডেটা ও মডেল ভার্সনিং',
  description: 'MLOps কী, ML লাইফসাইকেল, DVC দিয়ে ডেটাসেট ভার্সনিং, MLflow দিয়ে এক্সপেরিমেন্ট ট্র্যাকিং এবং মডেল রেজিস্ট্রি সম্পর্কে সম্পূর্ণ গাইড।',
  date: 'মে ২০২৬',
  category: 'এমএলঅপস',
  readTime: 14,
  content: `
    <h3>১. MLOps কী এবং কেন দরকার?</h3>
    <p>
      মেশিন লার্নিং প্রজেক্টে শুধু একটি ভালো মডেল তৈরি করাই যথেষ্ট নয়। সেই মডেলকে প্রোডাকশনে নেওয়া, নিয়মিত আপডেট করা, পারফরম্যান্স মনিটর করা — এই পুরো প্রক্রিয়াকে একসাথে ম্যানেজ করার পদ্ধতিকে বলা হয় <strong>MLOps (Machine Learning Operations)</strong>।
    </p>
    <p>
      MLOps হলো DevOps এবং ML-এর সমন্বয়। যেভাবে সফটওয়্যার ইঞ্জিনিয়ারিংয়ে CI/CD পাইপলাইন থাকে, MLOps-এ তেমনি থাকে ডেটা পাইপলাইন, মডেল ট্রেনিং পাইপলাইন, এবং ডিপ্লয়মেন্ট পাইপলাইন।
    </p>
    <p>MLOps ছাড়া ML প্রজেক্টে যেসব সমস্যা হয়:</p>
    <ul>
      <li>"আমার ল্যাপটপে কাজ করে কিন্তু সার্ভারে করে না" — পরিবেশের অসামঞ্জস্য</li>
      <li>কোন ডেটাসেট দিয়ে কোন মডেল ট্রেন করা হয়েছিল তা ভুলে যাওয়া</li>
      <li>কোন হাইপারপ্যারামিটার সবচেয়ে ভালো ফলাফল দিয়েছিল তা ট্র্যাক না করা</li>
      <li>প্রোডাকশন মডেলের পারফরম্যান্স কমে গেলে কারণ বুঝতে না পারা</li>
    </ul>

    <h3>২. ML লাইফসাইকেল: পাঁচটি মূল পর্যায়</h3>
    <p>একটি সম্পূর্ণ ML প্রজেক্ট পাঁচটি পর্যায়ে বিভক্ত:</p>
    <p>
      <strong>পর্যায় ১ — ডেটা সংগ্রহ ও প্রস্তুতি:</strong> Raw ডেটা সংগ্রহ, পরিষ্কার করা, ফিচার ইঞ্জিনিয়ারিং।<br/>
      <strong>পর্যায় ২ — এক্সপেরিমেন্টেশন:</strong> বিভিন্ন মডেল ও হাইপারপ্যারামিটার নিয়ে পরীক্ষা।<br/>
      <strong>পর্যায় ৩ — মডেল ট্রেনিং ও ভ্যালিডেশন:</strong> চূড়ান্ত মডেল ট্রেন করা এবং মূল্যায়ন।<br/>
      <strong>পর্যায় ৪ — ডিপ্লয়মেন্ট:</strong> মডেলকে API বা সার্ভিসে পরিণত করা।<br/>
      <strong>পর্যায় ৫ — মনিটরিং ও রিট্রেনিং:</strong> প্রোডাকশনে পারফরম্যান্স ট্র্যাক করা।
    </p>
    <p>
      MLOps এই প্রতিটি পর্যায়কে <strong>স্বয়ংক্রিয়</strong>, <strong>পুনরুৎপাদনযোগ্য</strong>, এবং <strong>ট্র্যাকযোগ্য</strong> করে তোলে।
    </p>

    <h3>৩. DVC: ডেটা ভার্সন কন্ট্রোল</h3>
    <p>
      Git যেভাবে কোড ভার্সন করে, <strong>DVC (Data Version Control)</strong> ঠিক সেভাবে ডেটাসেট ও মডেল ফাইল ভার্সন করে। বড় ফাইল গুলো (CSV, pickle, হাজার হাজার ইমেজ) সরাসরি Git-এ রাখা সম্ভব নয় — DVC এই সমস্যার সমাধান করে।
    </p>
    <p>DVC কীভাবে কাজ করে: ডেটা ফাইলের একটি ছোট মেটাডেটা ফাইল (.dvc) Git-এ রাখা হয়, আর আসল ডেটা S3/GCS/Azure বা লোকাল স্টোরেজে রাখা হয়।</p>
    <pre><code># DVC ইনস্টল করুন
pip install dvc dvc-s3

# Git রেপোতে DVC শুরু করুন
git init
dvc init

# ডেটাসেট ট্র্যাক করুন
dvc add data/train.csv
dvc add data/test.csv

# .dvc ফাইলগুলো Git-এ কমিট করুন
git add data/train.csv.dvc data/test.csv.dvc .gitignore
git commit -m "Add dataset tracking with DVC"

# Remote স্টোরেজ সেট করুন (লোকাল উদাহরণ)
dvc remote add -d myremote /tmp/dvc-storage
dvc push  # ডেটা রিমোটে পাঠান</code></pre>

    <p>ডেটা ভার্সন পরিবর্তন করলে পুরনো ভার্সনে ফিরে আসা:</p>
    <pre><code># নতুন ডেটাসেট যোগ করলে
dvc add data/train_v2.csv
git add data/train_v2.csv.dvc
git commit -m "Update training data to v2"

# পুরনো ভার্সনে ফিরে যেতে
git checkout HEAD~1 data/train.csv.dvc
dvc checkout  # পুরনো ডেটা রিস্টোর করবে

# ডেটা ডাউনলোড করুন (অন্য মেশিনে)
git clone https://github.com/yourrepo/project
dvc pull  # সব ডেটা ডাউনলোড হবে</code></pre>

    <h3>৪. DVC Pipeline: রিপ্রোডিউসিবল ওয়ার্কফ্লো</h3>
    <p>
      DVC শুধু ডেটা ভার্সনিং নয়, পুরো ML পাইপলাইনও ডিফাইন করতে পারে। <code>dvc.yaml</code> ফাইলে প্রতিটি ধাপ (stage) সংজ্ঞায়িত করা হয়:
    </p>
    <pre><code># dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw.csv
      - src/prepare.py
    outs:
      - data/processed.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed.csv
      - src/train.py
    params:
      - params.yaml:
          - model.n_estimators
          - model.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json</code></pre>

    <pre><code># params.yaml
model:
  n_estimators: 100
  max_depth: 5
  random_state: 42

data:
  test_size: 0.2</code></pre>

    <pre><code># পাইপলাইন চালানো
dvc repro  # শুধু পরিবর্তিত ধাপ রি-রান করবে

# মেট্রিক্স দেখা
dvc metrics show

# বিভিন্ন রান তুলনা করা
dvc metrics diff HEAD~1</code></pre>

    <h3>৫. MLflow: এক্সপেরিমেন্ট ট্র্যাকিং</h3>
    <p>
      <strong>MLflow</strong> হলো ML এক্সপেরিমেন্ট ম্যানেজমেন্টের সবচেয়ে জনপ্রিয় টুল। প্রতিটি ট্রেনিং রানের — প্যারামিটার, মেট্রিক্স, আর্টিফ্যাক্ট — সব কিছু স্বয়ংক্রিয়ভাবে লগ করে।
    </p>
    <pre><code>pip install mlflow scikit-learn pandas</code></pre>

    <pre><code>import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer

# ডেটা লোড করুন
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow Tracking URI সেট করুন
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer-classification")

# বিভিন্ন হাইপারপ্যারামিটার নিয়ে পরীক্ষা
param_grid = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

for params in param_grid:
    with mlflow.start_run():
        # প্যারামিটার লগ করুন
        mlflow.log_params(params)

        # মডেল ট্রেন করুন
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # প্রেডিকশন ও মেট্রিক্স
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # মেট্রিক্স লগ করুন
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # মডেল সেভ করুন
        mlflow.sklearn.log_model(model, "model")

        print(f"Params: {params} | Acc: {acc:.4f} | AUC: {auc:.4f}")</code></pre>

    <p>MLflow UI চালু করতে:</p>
    <pre><code># টার্মিনালে চালান
mlflow ui --backend-store-uri sqlite:///mlflow.db

# ব্রাউজারে যান: http://localhost:5000</code></pre>

    <h3>৬. MLflow মডেল রেজিস্ট্রি</h3>
    <p>
      MLflow-এর <strong>Model Registry</strong> হলো মডেল লাইফসাইকেল ম্যানেজমেন্টের কেন্দ্রবিন্দু। এখানে মডেলের তিনটি স্টেজ থাকে: <em>Staging</em>, <em>Production</em>, এবং <em>Archived</em>।
    </p>
    <pre><code>from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")

# সেরা রান খুঁজুন
experiment = client.get_experiment_by_name("breast-cancer-classification")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=1
)
best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"সেরা Run ID: {best_run_id}")
print(f"সেরা AUC: {best_run.data.metrics['roc_auc']:.4f}")

# মডেল রেজিস্টার করুন
model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(model_uri, "BreastCancerClassifier")

# Staging-এ প্রমোট করুন
client.transition_model_version_stage(
    name="BreastCancerClassifier",
    version=registered.version,
    stage="Staging"
)

# Production-এ প্রমোট করুন (টেস্ট পাস করার পর)
client.transition_model_version_stage(
    name="BreastCancerClassifier",
    version=registered.version,
    stage="Production"
)</code></pre>

    <h3>৭. এক্সপেরিমেন্ট রান তুলনা করা</h3>
    <p>MLflow দিয়ে সকল রান প্রোগ্রামাটিক্যালি তুলনা করা যায়:</p>
    <pre><code">import mlflow
import pandas as pd

# সব রানের ডেটা লোড করুন
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs_df = mlflow.search_runs(experiment_names=["breast-cancer-classification"])

# গুরুত্বপূর্ণ কলাম দেখুন
cols = [
    "run_id",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1_score",
    "metrics.roc_auc"
]
print(runs_df[cols].sort_values("metrics.roc_auc", ascending=False))

# সেরা মডেল লোড করুন Production থেকে
import mlflow.sklearn

model = mlflow.sklearn.load_model(
    "models:/BreastCancerClassifier/Production"
)
predictions = model.predict(X_test)
print(f"Production মডেলের Accuracy: {accuracy_score(y_test, predictions):.4f}")</code></pre>

    <h3>৮. DVC ও MLflow একসাথে ব্যবহার</h3>
    <p>
      দুটি টুল একসাথে ব্যবহার করলে সম্পূর্ণ reproducibility পাওয়া যায়: DVC ডেটা ও কোড ভার্সন করে, MLflow এক্সপেরিমেন্ট ট্র্যাক করে।
    </p>
    <pre><code># src/train.py
import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json

# DVC params লোড করুন
with open("params.yaml") as f:
    params = yaml.safe_load(f)

model_params = params["model"]

# ডেটা লোড করুন (DVC-tracked)
df = pd.read_csv("data/processed.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data"]["test_size"], random_state=42
)

# MLflow রান শুরু করুন
mlflow.set_experiment("dvc-mlflow-integration")

with mlflow.start_run():
    mlflow.log_params(model_params)

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")

# DVC metrics ফাইলে সেভ করুন
metrics = {"accuracy": acc, "roc_auc": auc}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Training complete. Accuracy: {acc:.4f}, AUC: {auc:.4f}")</code></pre>

    <h3>৯. সারসংক্ষেপ</h3>
    <p>এই অধ্যায়ে আমরা শিখলাম:</p>
    <ul>
      <li><strong>MLOps</strong> কেন প্রয়োজন এবং ML লাইফসাইকেলের পাঁচটি পর্যায়</li>
      <li><strong>DVC</strong> দিয়ে ডেটাসেট ও মডেল ফাইল ভার্সন কন্ট্রোল করা</li>
      <li><strong>DVC Pipeline</strong> দিয়ে reproducible ML ওয়ার্কফ্লো তৈরি করা</li>
      <li><strong>MLflow</strong> দিয়ে এক্সপেরিমেন্ট ট্র্যাকিং, মেট্রিক্স লগিং</li>
      <li><strong>MLflow Model Registry</strong> দিয়ে মডেল লাইফসাইকেল ম্যানেজমেন্ট</li>
    </ul>
    <p>পরবর্তী অধ্যায়ে আমরা শিখব কীভাবে স্বয়ংক্রিয় ML পাইপলাইন তৈরি করতে হয় Apache Airflow ও Prefect দিয়ে।</p>
  `
};
