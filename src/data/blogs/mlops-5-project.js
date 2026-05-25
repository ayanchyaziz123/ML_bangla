export const mlops_5_project = {
  slug: 'mlops-5-project',
  title: 'MLOps প্রজেক্ট: এন্ড-টু-এন্ড ML সিস্টেম',
  description: 'DVC, MLflow, FastAPI, Docker ও GitHub Actions দিয়ে একটি সম্পূর্ণ প্রোডাকশন ML পাইপলাইন তৈরি করুন',
  date: 'মে ২০২৫',
  category: 'এমএলঅপস',
  readTime: 18,
  content: `
<h3>আর্কিটেকচার পরিকল্পনা</h3>
<p>এই প্রজেক্টে একটি সম্পূর্ণ ML সিস্টেম তৈরি করব:</p>
<pre><code>ডেটা → DVC → প্রশিক্ষণ → MLflow → FastAPI → Docker → CI/CD</code></pre>

<h3>প্রজেক্ট কাঠামো</h3>

<pre><code class="language-bash">ml_project/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   └── main.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── dvc.yaml
└── params.yaml
</code></pre>

<h3>ধাপ ১: DVC দিয়ে ডেটা ভার্সনিং</h3>

<pre><code class="language-bash"># DVC সেটআপ
pip install dvc dvc-s3
git init && dvc init

# ডেটা ট্র্যাক করুন
dvc add data/raw/dataset.csv
git add data/raw/.gitignore data/raw/dataset.csv.dvc
git commit -m "ডেটা যোগ করা হয়েছে"

# দূরবর্তী স্টোরেজ (S3 বা Google Drive)
dvc remote add -d myremote s3://my-bucket/ml-data
dvc push
</code></pre>

<pre><code class="language-yaml"># dvc.yaml
stages:
  prepare:
    cmd: python src/data_prep.py
    deps:
      - src/data_prep.py
      - data/raw/dataset.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
    params:
      - params.yaml:
          - model.n_estimators
          - model.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json
</code></pre>

<h3>ধাপ ২: MLflow দিয়ে এক্সপেরিমেন্ট ট্র্যাকিং</h3>

<pre><code class="language-python">import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import yaml

# প্যারামিটার লোড
with open('params.yaml') as f:
    params = yaml.safe_load(f)

mlflow.set_experiment("ঋণ-অনুমোদন-মডেল")

with mlflow.start_run():
    # প্যারামিটার লগ
    mlflow.log_params(params['model'])

    # মডেল ট্রেনিং
    train_df = pd.read_csv('data/processed/train.csv')
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']

    model = RandomForestClassifier(**params['model'])
    model.fit(X_train, y_train)

    # মেট্রিক লগ
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    mlflow.log_metrics({'accuracy': accuracy, 'f1': f1})

    # মডেল রেজিস্ট্রি
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="loan-approval-model"
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
</code></pre>

<h3>ধাপ ৩: FastAPI সার্ভিং</h3>

<pre><code class="language-python"># api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np

app = FastAPI(title="ঋণ অনুমোদন API", version="1.0.0")

# মডেল লোড
model = mlflow.sklearn.load_model("models:/loan-approval-model/Production")

class LoanRequest(BaseModel):
    বয়স: int
    আয়: float
    ঋণ_পরিমাণ: float
    ক্রেডিট_স্কোর: int
    কর্মসংস্থান_বছর: int

class LoanResponse(BaseModel):
    সিদ্ধান্ত: str
    অনুমোদনের_সম্ভাবনা: float
    বিপদ_স্তর: str

@app.post("/predict", response_model=LoanResponse)
async def predict_loan(request: LoanRequest):
    try:
        features = pd.DataFrame([{
            'age': request.বয়স,
            'income': request.আয়,
            'loan_amount': request.ঋণ_পরিমাণ,
            'credit_score': request.ক্রেডিট_স্কোর,
            'employment_years': request.কর্মসংস্থান_বছর
        }])

        prob = model.predict_proba(features)[0][1]
        decision = "অনুমোদিত" if prob >= 0.5 else "প্রত্যাখ্যাত"
        risk = "কম" if prob >= 0.7 else "মাঝারি" if prob >= 0.4 else "বেশি"

        return LoanResponse(
            সিদ্ধান্ত=decision,
            অনুমোদনের_সম্ভাবনা=round(prob, 4),
            বিপদ_স্তর=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"স্থিতি": "সচল", "মডেল": "লোড হয়েছে"}
</code></pre>

<h3>ধাপ ৪: Docker কন্টেইনারাইজেশন</h3>

<pre><code class="language-dockerfile"># Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
</code></pre>

<pre><code class="language-yaml"># docker-compose.yml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
</code></pre>

<h3>ধাপ ৫: GitHub Actions CI/CD</h3>

<pre><code class="language-yaml"># .github/workflows/ci.yml
name: ML CI/CD পাইপলাইন

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Python সেটআপ
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: ডিপেন্ডেন্সি ইনস্টল
        run: pip install -r requirements.txt

      - name: ইউনিট টেস্ট
        run: pytest tests/ -v

      - name: DVC পাইপলাইন চালান
        run: dvc repro

      - name: Docker বিল্ড
        run: docker build -t ml-api:latest .

      - name: ইন্টিগ্রেশন টেস্ট
        run: |
          docker run -d -p 8000:8000 ml-api:latest
          sleep 5
          curl -f http://localhost:8000/health

      - name: Docker Hub-এ পুশ (main শাখায়)
        if: github.ref == 'refs/heads/main'
        run: |
          docker tag ml-api:latest myorg/ml-api:latest
          docker push myorg/ml-api:latest
</code></pre>

<h4>সম্পূর্ণ MLOps সিস্টেমের সুবিধা</h4>
<ul>
<li>পুনরুৎপাদনযোগ্য এক্সপেরিমেন্ট (DVC + MLflow)</li>
<li>স্বয়ংক্রিয় ডিপ্লয়মেন্ট (CI/CD)</li>
<li>স্কেলযোগ্য সার্ভিং (Docker + FastAPI)</li>
<li>মডেল ভার্সন কন্ট্রোল (MLflow Registry)</li>
<li>ড্রিফট মনিটরিং সহজ সংযোজনযোগ্য</li>
</ul>
`
};
