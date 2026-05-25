export const mlopsEn = [
  {
    slug: 'mlops-1-versioning',
    title: 'MLOps: Data and Model Versioning',
    description: 'DVC for dataset versioning and MLflow for experiment tracking and model registry',
    category: 'MLOps',
    content: `
<h3>The MLOps Problem</h3>
<p>Without proper tooling, ML experiments are unreproducible — different team members get different results, model versions get lost, and debugging is impossible. MLOps solves this with engineering rigor.</p>

<h3>DVC for Data Versioning</h3>
<pre><code class="language-bash">pip install dvc dvc-s3
git init && dvc init

# Track dataset
dvc add data/raw/dataset.csv
git add data/raw/.gitignore data/raw/dataset.csv.dvc .dvc/
git commit -m "Add dataset v1"

# Remote storage
dvc remote add -d myremote s3://mybucket/dvc-storage
dvc push

# Later: reproduce exact data version
git checkout v1.0
dvc pull  # Fetches exact data version
</code></pre>

<h3>MLflow Experiment Tracking</h3>
<pre><code class="language-python">import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

mlflow.set_experiment("breast-cancer-classification")

with mlflow.start_run(run_name="rf-baseline"):
    params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, proba)
    })

    # Log model to registry
    mlflow.sklearn.log_model(model, "model",
                              registered_model_name="breast-cancer-rf")

print(f"Run ID: {mlflow.active_run().info.run_id}")
</code></pre>
`
  },
  {
    slug: 'mlops-2-pipeline',
    title: 'ML Pipelines: Automated Workflows',
    description: 'sklearn Pipeline, DVC pipelines, and reproducible ML workflows',
    category: 'MLOps',
    content: `
<h3>Why Pipelines?</h3>
<p>ML pipelines prevent data leakage, make preprocessing reproducible, and enable clean deployment — the same transformation used at training time is automatically applied at inference time.</p>

<pre><code class="language-python">from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Sample loan data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.randint(22, 65, n),
    'income': np.random.uniform(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'employment': np.random.choice(['employed', 'self-employed', 'unemployed'], n),
    'approved': np.random.binomial(1, 0.7, n)
})

numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['employment']
X = df.drop('approved', axis=1)
y = df['approved']

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(clf_pipeline, X, y, cv=5, scoring='roc_auc')
print(f"ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

# Save pipeline (includes preprocessing)
import joblib
clf_pipeline.fit(X, y)
joblib.dump(clf_pipeline, 'loan_pipeline.pkl')
print("Pipeline saved — preprocessing included!")
</code></pre>
`
  },
  {
    slug: 'mlops-3-deployment',
    title: 'Model Deployment: FastAPI and Docker',
    description: 'Serving ML models as REST APIs with FastAPI, containerizing with Docker',
    category: 'MLOps',
    content: `
<h3>FastAPI Model Server</h3>
<pre><code class="language-python"># app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Loan Approval API", version="1.0.0")

# Load model at startup
model = joblib.load("loan_pipeline.pkl")

class LoanRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    employment: str = Field(pattern='^(employed|self-employed|unemployed)$')

class LoanResponse(BaseModel):
    approved: bool
    probability: float
    risk_level: str

@app.post("/predict", response_model=LoanResponse)
async def predict(request: LoanRequest):
    try:
        df = pd.DataFrame([request.dict()])
        prob = model.predict_proba(df)[0][1]
        approved = prob >= 0.5
        risk = "low" if prob >= 0.7 else "medium" if prob >= 0.4 else "high"
        return LoanResponse(approved=approved, probability=round(prob, 4), risk_level=risk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health(): return {"status": "healthy", "model": "loaded"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
</code></pre>

<pre><code class="language-dockerfile"># Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
</code></pre>

<pre><code class="language-bash">docker build -t loan-api:v1 .
docker run -p 8000:8000 loan-api:v1
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"age": 35, "income": 75000, "credit_score": 720, "employment": "employed"}'
</code></pre>
`
  },
  {
    slug: 'mlops-4-monitoring',
    title: 'Model Monitoring and Drift Detection',
    description: 'Data drift, concept drift, Evidently AI, and automated retraining triggers',
    category: 'MLOps',
    content: `
<h3>Types of Drift</h3>
<ul>
<li><strong>Data drift (covariate shift):</strong> Input feature distribution changes</li>
<li><strong>Concept drift:</strong> The relationship between inputs and outputs changes</li>
<li><strong>Label drift:</strong> Target distribution changes</li>
</ul>

<pre><code class="language-python">from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
import pandas as pd
import numpy as np
from scipy import stats

# Reference (training) data
np.random.seed(42)
ref = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.normal(60000, 15000, 1000),
    'credit_score': np.random.normal(700, 80, 1000),
    'prediction': np.random.binomial(1, 0.7, 1000),
    'target': np.random.binomial(1, 0.7, 1000)
})

# Current production data (drifted)
cur = pd.DataFrame({
    'age': np.random.normal(42, 12, 500),     # Distribution shifted
    'income': np.random.normal(52000, 20000, 500),
    'credit_score': np.random.normal(650, 100, 500),
    'prediction': np.random.binomial(1, 0.6, 500),
    'target': np.random.binomial(1, 0.52, 500)
})

# Evidently report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("drift_report.html")

# Manual KS test
for col in ['age', 'income', 'credit_score']:
    stat, p = stats.ks_2samp(ref[col], cur[col])
    status = "DRIFT DETECTED" if p < 0.05 else "OK"
    print(f"{col:<15}: KS={stat:.4f}, p={p:.4f} → {status}")

# Retraining trigger
def needs_retraining(current_acc, baseline_acc=0.92, threshold_drop=0.05):
    return (baseline_acc - current_acc) > threshold_drop

current_accuracy = 0.81
if needs_retraining(current_accuracy):
    print("⚠️ Retraining triggered: accuracy dropped significantly")
</code></pre>
`
  },
  {
    slug: 'mlops-5-project',
    title: 'MLOps Project: End-to-End ML System',
    description: 'Complete MLOps pipeline with DVC, MLflow, FastAPI, Docker, and GitHub Actions CI/CD',
    category: 'MLOps',
    content: `
<h3>Architecture</h3>
<pre><code>Data → DVC → Feature Engineering → MLflow Training → Model Registry → FastAPI → Docker → CI/CD</code></pre>

<h3>DVC Pipeline Definition</h3>
<pre><code class="language-yaml"># dvc.yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps: [src/prepare.py, data/raw/data.csv]
    outs: [data/processed/train.csv, data/processed/test.csv]

  train:
    cmd: python src/train.py
    deps: [src/train.py, data/processed/train.csv]
    params: [params.yaml:model]
    outs: [models/model.pkl]
    metrics: [metrics/scores.json:cache]
</code></pre>

<h3>Training Script with MLflow</h3>
<pre><code class="language-python">import mlflow, mlflow.sklearn
import pandas as pd, yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

with open('params.yaml') as f:
    params = yaml.safe_load(f)['model']

train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')
X_train, y_train = train_df.drop('target', axis=1), train_df['target']
X_test, y_test = test_df.drop('target', axis=1), test_df['target']

mlflow.set_experiment("loan-approval-v2")
with mlflow.start_run():
    mlflow.log_params(params)
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, preds), 'f1': f1_score(y_test, preds)}
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model", registered_model_name="loan-approval")

json.dump(metrics, open('metrics/scores.json', 'w'))
print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
</code></pre>

<h3>GitHub Actions CI/CD</h3>
<pre><code class="language-yaml"># .github/workflows/ml-ci.yml
name: ML CI/CD
on: [push]
jobs:
  train-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      - run: dvc pull && dvc repro
      - name: Build and test Docker
        run: |
          docker build -t ml-api:latest .
          docker run -d -p 8000:8000 ml-api:latest
          sleep 5 && curl -f http://localhost:8000/health
      - if: github.ref == 'refs/heads/main'
        run: docker push myorg/ml-api:latest
</code></pre>
`
  },
];
